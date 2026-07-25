#!/usr/bin/env python3
"""
run_trajectory_v2.py — NSGA-II run with FULL trajectory saved and
DETERMINISTIC replicate seeding (bit-for-bit reproducible).

Changes vs run_trajectory.py (v1, used for the thesis runs):

  [V2-1] Deterministic NetLogo seeds. v1 drew each replicate's NetLogo seed
         from an unseeded NumPy generator inside spawned workers, so re-runs
         were only statistically reproducible. v2 derives every replicate seed
         deterministically from (master seed, evaluation index, replicate index)
         via numpy SeedSequence, computed in the parent process and passed to
         workers explicitly. Same --seed => identical output files.
  [V2-2] Replicate-level log. Every simulation run is recorded in
         replicates_seed<N>.csv (evaluation index, parameters, NetLogo seed,
         per-replicate objectives), enabling Monte Carlo variance analysis
         that v1 discarded.
  [V2-3] Loud failure accounting. v1 silently mapped failed evaluations to
         (1e10, 1e10, 1e10). v2 does the same fallback but counts and prints
         failures, and exits with a warning if any occurred.

Usage:
    python3 run_trajectory_v2.py <config.json> --seed <N>

Output:
    trajectory_seed<N>.csv   — all generations × all individuals
    pareto_seed<N>.csv       — final Pareto front
    replicates_seed<N>.csv   — per-replicate seeds and objectives
"""
import json, os, sys, subprocess
import pandas as pd
import numpy as np
import multiprocessing
from functools import partial

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.callback import Callback

EXPERIMENT_XML = """
<experiments>
  <experiment name="optimization_run" repetitions="1" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="{ticks}"/>
    <metric>total-innovation-output</metric>
    <metric>cultural-diversity-index</metric>
    <metric>gini-coefficient</metric>
    <enumeratedValueSet variable="random-seed">
      <value value="{seed}"/>
    </enumeratedValueSet>
    {enumerated_values}
  </experiment>
</experiments>
"""


class TrajectoryCallback(Callback):
    """Accumulates ALL generation data in memory; periodic checkpoint to disk."""
    def __init__(self, param_names, seed=42):
        super().__init__()
        self.param_names = param_names
        self.history = []
        self.seed = seed

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        df = pd.DataFrame(X, columns=self.param_names)
        df['Obj_Innov_Neg'] = F[:, 0]
        df['Obj_Div_Neg'] = F[:, 1]
        df['Obj_Gini'] = F[:, 2]
        df['Generation'] = gen
        self.history.append(df)
        n_nds = len(algorithm.opt)
        print(f"  Gen {gen:>3d} | Pop {len(X)} | NDS {n_nds} | "
              f"Innov ~{-F[:,0].mean():.0f} | Div ~{-F[:,1].mean():.2f} | Gini ~{F[:,2].mean():.3f}")
        if gen % 5 == 0:
            traj = self.get_trajectory()
            ckpt_file = f"trajectory_seed{self.seed}_checkpoint.csv"
            traj.to_csv(ckpt_file, index=False)
            print(f"    [checkpoint] {len(traj)} rows → {ckpt_file}")

    def get_trajectory(self):
        if not self.history:
            return pd.DataFrame()
        return pd.concat(self.history, ignore_index=True)


def run_single_simulation(params, config, task):
    """Run one NetLogo simulation. `task` = (replicate_id, netlogo_seed),
    both computed deterministically in the parent process [V2-1]."""
    replicate_id, netlogo_seed = task
    pid = os.getpid()
    unique_id = f"{pid}_{replicate_id}_{netlogo_seed}"

    param_xml_lines = ""
    for key, val in params.items():
        param_xml_lines += f'<enumeratedValueSet variable="{key}"><value value="{val}"/></enumeratedValueSet>\n'

    xml_content = EXPERIMENT_XML.format(
        ticks=config["MAX_TICKS"], seed=netlogo_seed, enumerated_values=param_xml_lines)

    xml_filename = f"temp_{unique_id}.xml"
    csv_filename = f"temp_{unique_id}.csv"

    try:
        with open(xml_filename, "w") as f:
            f.write(xml_content)
        cmd = [config["NETLOGO_PATH"], "--headless", "--model", config["MODEL_PATH"],
               "--setup-file", xml_filename, "--table", csv_filename]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=300)
        try:
            df = pd.read_csv(csv_filename, skiprows=6, on_bad_lines='skip')
        except Exception:
            return None
        if df.empty:
            return None
        clean_cols = {c: c.replace('"', '').strip() for c in df.columns}
        df.rename(columns=clean_cols, inplace=True)
        final_state = df.iloc[-1]
        return {
            'replicate': replicate_id,
            'netlogo_seed': netlogo_seed,
            'innovation': float(final_state.get('total-innovation-output', 0)),
            'diversity': float(final_state.get('cultural-diversity-index', 0)),
            'gini': float(final_state.get('gini-coefficient', 0)),
        }
    except Exception:
        return None
    finally:
        if os.path.exists(xml_filename): os.remove(xml_filename)
        if os.path.exists(csv_filename): os.remove(csv_filename)


class NetLogoOptimization(ElementwiseProblem):
    def __init__(self, config, master_seed, n_threads=4):
        self.config = config
        self.params = config["PARAM_BOUNDS"]
        self.param_names = list(self.params.keys())
        self.n_replicates = config.get("N_REPLICATES", 1)
        self.n_threads = n_threads
        self.master_seed = master_seed
        self.eval_index = 0          # incremented once per candidate evaluation
        self.failed_runs = 0
        self.replicate_log = f"replicates_seed{master_seed}.csv"
        with open(self.replicate_log, "w") as f:
            f.write("eval_index," + ",".join(self.param_names) +
                    ",replicate,netlogo_seed,innovation,diversity,gini\n")
        xl = [self.params[k][0] for k in self.param_names]
        xu = [self.params[k][1] for k in self.param_names]
        super().__init__(n_var=len(self.param_names), n_obj=3, xl=xl, xu=xu)

    def _replicate_seeds(self):
        """[V2-1] Deterministic seeds for this evaluation's replicates."""
        ss = np.random.SeedSequence([self.master_seed, self.eval_index])
        # 31-bit positive ints, valid for NetLogo's random-seed
        return [int(s % 2147483646) + 1 for s in ss.generate_state(self.n_replicates, dtype=np.uint64)]

    def _evaluate(self, x, out, *args, **kwargs):
        param_dict = dict(zip(self.param_names, x))
        seeds = self._replicate_seeds()
        tasks = list(enumerate(seeds))
        with multiprocessing.Pool(min(self.n_threads, self.n_replicates)) as pool:
            func = partial(run_single_simulation, param_dict, self.config)
            results = pool.map(func, tasks)

        valid = [r for r in results if r is not None]
        n_failed = len(results) - len(valid)
        if n_failed:
            self.failed_runs += n_failed
            print(f"    [warn] eval {self.eval_index}: {n_failed}/{len(results)} replicate(s) failed")

        with open(self.replicate_log, "a") as f:      # [V2-2]
            for r in valid:
                f.write(f"{self.eval_index}," +
                        ",".join(repr(float(param_dict[k])) for k in self.param_names) +
                        f",{r['replicate']},{r['netlogo_seed']},"
                        f"{r['innovation']},{r['diversity']},{r['gini']}\n")

        self.eval_index += 1

        if not valid:
            out["F"] = [1e10, 1e10, 1e10]
            return
        df_res = pd.DataFrame(valid)
        out["F"] = [-df_res['innovation'].mean(), -df_res['diversity'].mean(), df_res['gini'].mean()]


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    if len(sys.argv) < 2:
        print("Usage: python3 run_trajectory_v2.py <config.json> --seed <N>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        config = json.load(f)

    master_seed = 42
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        master_seed = int(sys.argv[idx + 1])

    n_cpu = multiprocessing.cpu_count()
    print(f"=== Trajectory Run v2 (Seed={master_seed}, CPUs={n_cpu}, deterministic replicate seeding) ===\n")

    problem = NetLogoOptimization(config, master_seed, n_threads=n_cpu)
    trajectory_cb = TrajectoryCallback(problem.param_names, seed=master_seed)

    algorithm = NSGA2(
        pop_size=config.get("POP_SIZE", 50),
        n_offsprings=10,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem, algorithm,
                   ('n_gen', config.get("N_GENERATIONS", 50)),
                   seed=master_seed,
                   callback=trajectory_cb,
                   verbose=False)

    traj = trajectory_cb.get_trajectory()
    traj_file = f"trajectory_seed{master_seed}.csv"
    traj.to_csv(traj_file, index=False)
    n_gens = traj['Generation'].nunique()
    print(f"\n  Trajectory: {len(traj)} rows × {n_gens} generations → {traj_file}")

    pareto_file = f"pareto_seed{master_seed}.csv"
    result_df = pd.DataFrame(res.X, columns=problem.param_names)
    result_df['Obj_Innov_Neg'] = res.F[:, 0]
    result_df['Obj_Div_Neg'] = res.F[:, 1]
    result_df['Obj_Gini'] = res.F[:, 2]
    result_df.to_csv(pareto_file, index=False)
    print(f"  Pareto: {len(result_df)} solutions → {pareto_file}")
    print(f"  Replicate log: {problem.replicate_log}")

    if problem.failed_runs:
        print(f"\n  [WARNING] {problem.failed_runs} simulation run(s) failed during this experiment.")
    print(f"\n=== Done (Seed={master_seed}) ===")
