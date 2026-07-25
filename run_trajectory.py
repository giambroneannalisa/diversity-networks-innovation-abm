#!/usr/bin/env python3
"""
run_trajectory.py — Single NSGA-II run with FULL evolutionary trajectory saved.

Unlike the main optimization script, this accumulates ALL generations in memory
and writes the complete history at the end, avoiding file-append issues.

Usage:
    python3 run_trajectory.py <config.json> --seed <N>

Example:
    python3 run_trajectory.py nsga2_config_v3.json --seed 42

Output:
    trajectory_seed<N>.csv  — all generations × all individuals
    pareto_seed<N>.csv      — final Pareto front only
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

# --- XML TEMPLATE ---
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
    """Accumulates ALL generation data in memory. Writes nothing to disk during run."""
    def __init__(self, param_names, seed=42):
        super().__init__()
        self.param_names = param_names
        self.history = []  # list of DataFrames
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

        # Checkpoint every 5 generations
        if gen % 5 == 0:
            traj = self.get_trajectory()
            ckpt_file = f"trajectory_seed{self.seed}_checkpoint.csv"
            traj.to_csv(ckpt_file, index=False)
            print(f"    [checkpoint] {len(traj)} rows → {ckpt_file}")

    def get_trajectory(self):
        if not self.history:
            return pd.DataFrame()
        return pd.concat(self.history, ignore_index=True)


def run_single_simulation(params, config, replicate_id):
    pid = os.getpid()
    unique_id = f"{pid}_{replicate_id}_{np.random.randint(1000, 9999)}"
    current_seed = int(np.random.randint(0, 2147483647))

    param_xml_lines = ""
    for key, val in params.items():
        param_xml_lines += f'<enumeratedValueSet variable="{key}"><value value="{val}"/></enumeratedValueSet>\n'

    xml_content = EXPERIMENT_XML.format(
        ticks=config["MAX_TICKS"], seed=current_seed, enumerated_values=param_xml_lines)

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
        except:
            return None
        if df.empty: return None
        clean_cols = {c: c.replace('"', '').strip() for c in df.columns}
        df.rename(columns=clean_cols, inplace=True)
        final_state = df.iloc[-1]
        return {
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
    def __init__(self, config, n_threads=4):
        self.config = config
        self.params = config["PARAM_BOUNDS"]
        self.param_names = list(self.params.keys())
        self.n_replicates = config.get("N_REPLICATES", 1)
        self.n_threads = n_threads
        xl = [self.params[k][0] for k in self.param_names]
        xu = [self.params[k][1] for k in self.param_names]
        super().__init__(n_var=len(self.param_names), n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        param_dict = dict(zip(self.param_names, x))
        with multiprocessing.Pool(self.n_threads) as pool:
            func = partial(run_single_simulation, param_dict, self.config)
            results = pool.map(func, range(self.n_replicates))
        valid = [r for r in results if r is not None]
        if not valid:
            out["F"] = [1e10, 1e10, 1e10]
            return
        df_res = pd.DataFrame(valid)
        out["F"] = [-df_res['innovation'].mean(), -df_res['diversity'].mean(), df_res['gini'].mean()]


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    if len(sys.argv) < 2:
        print("Usage: python3 run_trajectory.py <config.json> --seed <N>")
        sys.exit(1)

    with open(sys.argv[1], 'r') as f:
        config = json.load(f)

    pymoo_seed = 42
    if '--seed' in sys.argv:
        idx = sys.argv.index('--seed')
        pymoo_seed = int(sys.argv[idx + 1])

    n_cpu = multiprocessing.cpu_count()
    print(f"=== Trajectory Run (Seed={pymoo_seed}, CPUs={n_cpu}) ===\n")

    problem = NetLogoOptimization(config, n_threads=n_cpu)
    trajectory_cb = TrajectoryCallback(problem.param_names, seed=pymoo_seed)

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
                   seed=pymoo_seed,
                   callback=trajectory_cb,
                   verbose=False)

    # === Save trajectory (ALL generations) ===
    traj = trajectory_cb.get_trajectory()
    traj_file = f"trajectory_seed{pymoo_seed}.csv"
    traj.to_csv(traj_file, index=False)
    n_gens = traj['Generation'].nunique()
    print(f"\n  Trajectory: {len(traj)} rows × {n_gens} generations → {traj_file}")

    # === Save Pareto front ===
    pareto_file = f"pareto_seed{pymoo_seed}.csv"
    result_df = pd.DataFrame(res.X, columns=problem.param_names)
    result_df['Obj_Innov_Neg'] = res.F[:, 0]
    result_df['Obj_Div_Neg'] = res.F[:, 1]
    result_df['Obj_Gini'] = res.F[:, 2]
    result_df.to_csv(pareto_file, index=False)
    print(f"  Pareto: {len(result_df)} solutions → {pareto_file}")

    print(f"\n=== Done (Seed={pymoo_seed}) ===")
