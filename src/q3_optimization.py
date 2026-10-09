"""Q3: feasible particle swarm, fixed/free recipes, and SLSQP comparison.

Inputs are Q1/Q2 exports and C7 metadata; no original A data is reread.
The separate p effect is an explicit scenario assumption, not identified by B.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from common import BASE, C, MIX_DOMAINS
from scaling_models import optimal_mixture

ETA, Q0 = 2e-4, 0.4
N_MIN, N_MAX, D_MIN, D_MAX = .005, 1e5, .2, 1e5
FORMS = ("additive", "interaction")
COSTS = ("exp", "power", "log")
BUDGETS = (1e19, 1e22, 1e24)
KEYS = ["model", "cost_form", "budget_FLOPs", "L_ctx", "recipe_mode"]


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_inputs():
    root = Path(BASE)
    q1_path = root / "results/q1/mixture_models.json"
    q2_path = root / "results/q2/model_results.json"
    parameters_path = root / "results/q2/model_parameters.csv"
    selection_path = root / "results/q2/selected_model.json"
    c7_path = Path(C) / "model_architecture_metadata.csv"
    manifest = read_json(q2_path)
    mixture = optimal_mixture(q1_path)
    if not np.allclose(mixture["p_star"], manifest["mixture"]["p_star"], atol=1e-12):
        raise ValueError("Q1 optimum changed: rerun q2_scaling.py first")
    if not np.isclose(mixture["H_star"], manifest["mixture"]["H_star"], atol=1e-12):
        raise ValueError("Q1 H_star changed: rerun Q2 comparison")
    for source in (q1_path, root / "results/q2/classical_baseline.json"):
        expected = manifest["sources_sha256"][str(source.relative_to(root))]
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Stale Q2 inputs: {source}")
    table = pd.read_csv(parameters_path)
    params = {}
    for form in FORMS:
        rows = table[(table.model == f"{form}_mixture") & (table.evaluation == "fit_unique")]
        if len(rows) != 1:
            raise ValueError(f"Expected one Q2 full-data fit for {form}")
        row = rows.iloc[0]
        names = ["A", "a", "B", "b", "C", "g", "lambda"] + (["d"] if form == "interaction" else [])
        params[form] = {name: float(row[name]) for name in names}
        params[form]["E"] = float(row.E_fixed)
        if min(params[form].values()) <= 0:
            raise ValueError("Expected positive scaling coefficients and lambda")
    c7 = pd.read_csv(c7_path)
    supported = sorted(c7.max_position_embeddings.dropna().astype(int).unique().tolist())
    if not {2048, 4096}.issubset(supported):
        raise ValueError("Requested primary windows not present in C7")
    selection = read_json(selection_path)
    fitted = selection["fit"]
    chosen = fitted["form"]
    main_key = chosen if fitted["mode"] == "mixture" else chosen + "_fixed_E"
    if main_key not in params:
        params[main_key] = {**fitted["params"], "E": fitted["E"],
                            "lambda": fitted["params"].get("lambda", 0.)}
    manifest_oof = pd.read_csv(root / "results/q2/model_performance.csv")
    ranked = manifest_oof[manifest_oof.evaluation == "group_cv_oof"].sort_values(["RMSE", "model"])
    if ranked.iloc[0].model != selection["model"]:
        raise ValueError("Q2 selection disagrees with OOF metrics")
    sources = [q1_path, q2_path, parameters_path, selection_path, c7_path]
    mixture["main_model"] = main_key
    mixture["selection"] = selection
    return params, mixture, c7, supported, {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def quality_cost(Q, form):
    Q = np.asarray(Q)
    if form == "exp":
        return 1e7 * np.exp(6 * Q)
    if form == "power":
        return 5e9 * Q**4
    if form == "log":
        return 2e9 * np.log1p(10 * Q)
    raise ValueError(form)


def project_simplex(points):
    """Euclidean projection onto p>=0, sum(p)=1, including exact zeros."""
    points = np.atleast_2d(points)
    ordered = np.sort(points, axis=1)[:, ::-1]
    sums = np.cumsum(ordered, axis=1) - 1
    indexes = np.arange(1, points.shape[1] + 1)
    rho = np.sum(ordered - sums / indexes > 0, axis=1) - 1
    theta = sums[np.arange(len(points)), rho] / (rho + 1)
    return np.maximum(points - theta[:, None], 0)


class Problem:
    def __init__(self, model, params, mixture, cost_form, budget, lctx, mode):
        self.model, self.params, self.mixture = model, params, mixture
        self.cost_form, self.budget, self.lctx, self.mode = cost_form, budget, lctx, mode
        self.dimension = 19 if mode == "free" else 2
        self.resource_dimensions = 2
        self.p_star = np.asarray(mixture["p_star"])
        self.b = np.asarray(mixture["mean_coefficients"])
        self.tau = mixture["tau"]
        self.h_ref = np.log1p(np.asarray(mixture["p_ref"]) / self.tau) @ self.b
        self.train_coefficient = 1e18 * (6 + ETA * lctx)

    def h(self, p):
        return np.log1p(np.maximum(p, 0) / self.tau) @ self.b - self.h_ref

    def decode(self, points):
        z = np.atleast_2d(points)
        Q = Q0 + (1 - Q0) * z[:, 1]
        cq = 1e9 * np.maximum(quality_cost(Q, self.cost_form) - quality_cost(Q0, self.cost_form), 0)
        # Exact feasible coordinates: D is chosen maximally because dL/dD<0.
        # Q-dependent N upper bound ensures D>=D_MIN for EVERY search point.
        n_upper = np.minimum(N_MAX, (self.budget / D_MIN - cq) / self.train_coefficient)
        if np.any(n_upper < N_MIN):
            raise ValueError("Budget cannot support the requested quality range")
        N = np.exp(np.log(N_MIN) + z[:, 0] * np.log(n_upper / N_MIN))
        D = np.minimum(D_MAX, self.budget / (self.train_coefficient * N + cq))
        p = z[:, 2:] if self.mode == "free" else np.broadcast_to(self.p_star, (len(z), 17))
        return N, D, Q, p

    def loss_physical(self, N, D, Q, p):
        t = self.params
        if self.model.startswith("quality_power"):
            quality = t["eta_Q"] * (1 - Q**t["kappa"])
        else:
            quality = t["C"] * np.maximum(1 - Q, 0)**t["g"]
        if self.model.startswith("interaction"):
            quality *= D**(-t["d"])
        return t["E"] + t["A"] * N**(-t["a"]) + t["B"] * D**(-t["b"]) + quality + t["lambda"] * self.h(p)

    def objective(self, points):
        return self.loss_physical(*self.decode(points))

    def repair(self, points):
        z = np.array(points, copy=True)
        z[:, :2] = np.clip(z[:, :2], 0, 1)
        if self.mode == "free":
            z[:, 2:] = project_simplex(z[:, 2:])
        return z

    def population(self, rng, size):
        z = rng.uniform(0, 1, (size, self.dimension))
        if self.mode == "free":
            z[:, 2:] = rng.dirichlet(np.ones(17), size)
        # Common, non-oracle boundary starts for both solvers.
        z[0, :2] = [.5, 0]
        z[1, :2] = [.5, 1]
        if self.mode == "free":
            z[:2, 2:] = 1 / 17
        return z

    def record(self, z):
        N, D, Q, p = self.decode(z)
        N, D, Q, p = float(N[0]), float(D[0]), float(Q[0]), p[0]
        ct = 6e18 * N * D
        cq = D * 1e9 * max(float(quality_cost(Q, self.cost_form) - quality_cost(Q0, self.cost_form)), 0)
        ca = ETA * 1e18 * N * D * self.lctx
        total = ct + cq + ca
        return {"N_B": N, "D_B": D, "Q": Q, "Loss": float(self.objective(z)[0]),
                "H_p": float(self.h(p)), "C_train_FLOPs": ct, "C_quality_FLOPs": cq,
                "C_attn_FLOPs": ca, "budget_utilization": total / self.budget,
                "s_train": ct / total, "s_quality": cq / total, "s_attn": ca / total,
                "attention_to_train": ca / ct,
                "p_L1_from_star": float(np.abs(p - self.p_star).sum()),
                "simplex_error": float(abs(p.sum() - 1)), "min_p": float(p.min()),
                **{f"p_{name}": float(value) for name, value in zip(MIX_DOMAINS, p)}}


def solve_pso(problem, seed, particles=128, iterations=800):
    """Pure projected global-best PSO; no SLSQP polishing or known p* starts."""
    started = perf_counter()
    rng = np.random.default_rng(seed)
    x = problem.population(rng, particles)
    v = np.zeros_like(x)
    personal = x.copy()
    values = problem.objective(x)
    evaluations = particles
    best = personal[np.argmin(values)].copy()
    best_value = float(values.min())
    history = [{"iteration": 0, "evaluations": evaluations, "best_Loss": best_value}]
    for iteration in range(iterations):
        inertia = .9 - .5 * iteration / max(iterations - 1, 1)
        v = inertia * v + 1.7 * rng.random(x.shape) * (personal - x) + 1.7 * rng.random(x.shape) * (best - x)
        resource_dimensions = problem.resource_dimensions
        v[:, :resource_dimensions] = np.clip(v[:, :resource_dimensions], -.2, .2)
        if problem.mode == "free":
            v[:, resource_dimensions:] = np.clip(v[:, resource_dimensions:], -.06, .06)
        new_x = problem.repair(x + v)
        v = new_x - x
        x = new_x
        current = problem.objective(x)
        evaluations += particles
        better = current < values
        personal[better], values[better] = x[better], current[better]
        candidate = int(np.argmin(values))
        if values[candidate] < best_value:
            best_value, best = float(values[candidate]), personal[candidate].copy()
        # Diversification is stochastic, uses no derivative or reference solution.
        if (iteration + 1) % 100 == 0 and iteration + 1 < iterations:
            count = max(2, particles // 10)
            worst = np.argsort(values)[-count:]
            trial = problem.population(rng, count)
            local = count // 2
            trial[:local] = problem.repair(best[None, :] + rng.normal(0, .02, (local, problem.dimension)))
            x[worst], v[worst] = trial, 0
            personal[worst], values[worst] = trial, problem.objective(trial)
            evaluations += count
            candidate = int(np.argmin(values))
            if values[candidate] < best_value:
                best_value, best = float(values[candidate]), personal[candidate].copy()
        history.append({"iteration": iteration + 1, "evaluations": evaluations, "best_Loss": best_value})
    return best, {"seconds": perf_counter() - started, "evaluations": evaluations,
                  "iterations": iterations, "solver_success": True, "successful_starts": 1}, history


def solve_slsqp(problem, seed, starts=24, maxiter=600, population_size=128):
    """Same feasible resource coordinates and initial-start generator as PSO."""
    started = perf_counter()
    rng = np.random.default_rng(seed)
    x0 = problem.population(rng, population_size)
    if starts > population_size:
        x0 = np.vstack([x0, problem.population(rng, starts - population_size)])
    x0 = x0[:starts]
    constraints = [] if problem.mode == "fixed" else [
        {"type": "eq", "fun": lambda z: z[2:].sum() - 1,
         "jac": lambda z: np.r_[np.zeros(2), np.ones(17)]}]
    evaluations, iterations, successes = 0, 0, 0
    best, best_value = None, np.inf
    for start in x0:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Values in x were outside bounds")
            sol = minimize(lambda z: float(problem.objective(z)[0]), start,
                           method="SLSQP", bounds=[(0, 1)] * problem.dimension,
                           constraints=constraints, options={"maxiter": maxiter, "ftol": 1e-11})
        evaluations += sol.nfev
        iterations += sol.nit
        feasible = (np.isfinite(sol.fun) and np.all(sol.x >= -1e-9) and np.all(sol.x <= 1 + 1e-9)
                    and (problem.mode == "fixed" or abs(sol.x[2:].sum() - 1) < 1e-8))
        successes += int(sol.success and feasible)
        if sol.success and feasible and sol.fun < best_value:
            best_value, best = float(sol.fun), sol.x.copy()
    if best is None:
        raise RuntimeError("SLSQP found no successful feasible solution")
    return best, {"seconds": perf_counter() - started, "evaluations": evaluations,
                  "iterations": iterations, "solver_success": True, "successful_starts": successes}, []


def reference_solution(problem, grid_size=161):
    """Independent nested bounded search, endpoints and all detected Q minima.

    At fixed Q, the objective is convex in log N (positive posynomial),
    including the D_MAX cap. Q is scanned and every detected basin refined.
    This is a high-accuracy numerical reference, not a global Q proof.
    """
    def at_quality(q):
        upper = min(N_MAX, (problem.budget / D_MIN - 1e9 * float(
            quality_cost(q, problem.cost_form) - quality_cost(Q0, problem.cost_form))) / problem.train_coefficient)
        def evaluate(lnn):
            n = np.exp(lnn)
            cq = 1e9 * max(float(quality_cost(q, problem.cost_form) - quality_cost(Q0, problem.cost_form)), 0)
            d = min(D_MAX, problem.budget / (problem.train_coefficient * n + cq))
            return float(problem.loss_physical(n, d, q, problem.p_star))
        lo, hi = np.log(N_MIN), np.log(upper)
        sol = minimize_scalar(evaluate, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        candidates = [(evaluate(lo), lo), (evaluate(hi), hi), (float(sol.fun), float(sol.x))]
        loss, lnn = min(candidates)
        return loss, (lnn - lo) / (hi - lo)
    grid = np.linspace(Q0, 1, grid_size)
    sampled = np.array([at_quality(q)[0] for q in grid])
    candidates = [(sampled[0], grid[0]), (sampled[-1], grid[-1])]
    for i in range(1, len(grid) - 1):
        if sampled[i] <= sampled[i - 1] and sampled[i] <= sampled[i + 1]:
            sol = minimize_scalar(lambda q: at_quality(q)[0], bounds=(grid[i - 1], grid[i + 1]),
                                  method="bounded", options={"xatol": 1e-12})
            candidates.append((float(sol.fun), float(sol.x)))
    loss, q = min(candidates)
    _, n_coordinate = at_quality(q)
    z = np.r_[n_coordinate, (q - Q0) / (1 - Q0)]
    if problem.mode == "free":
        z = np.r_[z, problem.p_star]
    return z, loss


def summarize(runs):
    summary = runs.groupby(KEYS + ["algorithm"], as_index=False).agg(
        repetitions=("Loss", "size"), best_Loss=("Loss", "min"), mean_Loss=("Loss", "mean"),
        std_Loss=("Loss", "std"), median_seconds=("seconds", "median"),
        mean_evaluations=("evaluations", "mean"), max_gap=("gap_to_reference", "max"),
        mean_gap=("gap_to_reference", "mean"), reference_hit_rate=("reference_hit", "mean"))
    best = runs.loc[runs.groupby(KEYS + ["algorithm"])["Loss"].idxmin()].copy()
    pair_keys = ["model", "cost_form", "budget_FLOPs", "L_ctx", "algorithm", "seed"]
    fixed = runs[runs.recipe_mode == "fixed"]
    free = runs[runs.recipe_mode == "free"]
    pair = free.merge(fixed, on=pair_keys, suffixes=("_free", "_fixed"))
    comparison = pair[pair_keys].copy()
    comparison["Loss_fixed"] = pair.Loss_fixed
    comparison["Loss_free"] = pair.Loss_free
    comparison["free_minus_fixed_Loss"] = pair.Loss_free - pair.Loss_fixed
    comparison["N_relative_change"] = pair.N_B_free / pair.N_B_fixed - 1
    comparison["D_relative_change"] = pair.D_B_free / pair.D_B_fixed - 1
    comparison["Q_change"] = pair.Q_free - pair.Q_fixed
    comparison["H_excess_over_star"] = pair.H_p_free - pair.H_p_fixed
    comparison["p_L1_from_star"] = pair.p_L1_from_star_free
    comparison["theoretical_improvement"] = 0.0
    return summary, best, comparison


def make_figures(out, runs, refs, best, history, supported):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, form in zip(axes, FORMS):
        for window, marker in [(2048, "o"), (4096, "s")]:
            frame = refs[(refs.model == form) & (refs.cost_form == "power") & (refs.L_ctx == window)]
            ax.plot(frame.budget_FLOPs, frame.Loss, marker=marker, label=f"L_ctx={window}")
        ax.set(xscale="log", xlabel="Budget (FLOPs)", ylabel="Predicted loss", title=form.capitalize())
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / "01_model_comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    free = runs[runs.recipe_mode == "free"]
    for ax, form in zip(axes, FORMS):
        for algorithm, marker in [("PSO", "o"), ("SLSQP", "s")]:
            frame = free[(free.model == form) & (free.cost_form == "power") & (free.L_ctx == 4096)]
            frame = frame[frame.algorithm == algorithm].groupby("budget_FLOPs").gap_to_reference.mean()
            ax.plot(frame.index, np.maximum(frame.values, 1e-14), marker=marker, label=algorithm)
        ax.set(xscale="log", yscale="log", xlabel="Budget (FLOPs)", ylabel="Mean loss gap to reference", title=form.capitalize())
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / "02_solver_gap.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode in zip(axes, ["fixed", "free"]):
        frame = runs[(runs.recipe_mode == mode) & (runs.L_ctx == 4096)]
        for i, algorithm in enumerate(["PSO", "SLSQP"]):
            values = frame[frame.algorithm == algorithm].seconds.to_numpy()
            ax.scatter(np.full(len(values), i), values, alpha=.35, s=16)
            ax.scatter(i, np.median(values), color="black", marker="_", s=250, zorder=3)
        ax.set(xticks=[0, 1], xticklabels=["PSO", "SLSQP"], yscale="log", ylabel="Seconds per complete run", title=f"Recipe: {mode}")
        ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "03_solver_runtime.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    selection = best[(best.model == "interaction") & (best.cost_form == "power") & (best.budget_FLOPs == 1e22)
                     & (best.L_ctx == 4096) & (best.recipe_mode == "free")]
    pcols = [f"p_{domain}" for domain in MIX_DOMAINS]
    ax.plot(np.arange(17), refs.iloc[0][pcols].to_numpy(float), "o-", label="Previously optimal p*")
    for _, row in selection.iterrows():
        ax.plot(np.arange(17), row[pcols].to_numpy(float), ".--", label=f"Free p: {row.algorithm}")
    ax.set(xticks=np.arange(17), xticklabels=MIX_DOMAINS, ylabel="Domain fraction")
    ax.tick_params(axis="x", rotation=65); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "04_recipe_comparison.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    windows = sorted(set(supported + [30000]))
    ax.plot(windows, ETA * np.array(windows) / 6, "o-")
    ax.axhline(1, color="gray", ls="--")
    ax.axvline(30000, color="#b04a45", ls=":", label="Theoretical equality: 30000")
    ax.set(xscale="log", yscale="log", xlabel="Context length", ylabel="Attention / training cost")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "05_context_cost.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    frame = history[(history.model == "interaction") & (history.cost_form == "power")
                    & (history.budget_FLOPs == 1e22) & (history.L_ctx == 4096) & (history.seed == 20261009)]
    for mode in ["fixed", "free"]:
        curve = frame[frame.recipe_mode == mode]
        ax.plot(curve.evaluations, np.maximum(curve.gap_to_reference, 1e-14), label=mode)
    ax.set(yscale="log", xlabel="Objective evaluations", ylabel="Loss gap to reference", title="PSO convergence: interaction, 1e22, power cost")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(out / "06_pso_convergence.png", dpi=180); plt.close(fig)


def write_report(out, config, refs, summary, comparison, runs):
    def md(frame):
        headers = list(frame.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(f"{x:.8g}" if isinstance(x, (float, np.floating)) else str(x) for x in row) + " |")
        return "\n".join(lines)
    table = refs[(refs.cost_form == "power") & (refs.L_ctx == 4096)][["model", "budget_FLOPs", "N_B", "D_B", "Q", "Loss"]]
    solver_table = runs.groupby(["recipe_mode", "algorithm"], as_index=False).agg(
        mean_gap=("gap_to_reference", "mean"), max_gap=("gap_to_reference", "max"),
        hit_rate=("reference_hit", "mean"), median_seconds=("seconds", "median"), mean_evaluations=("evaluations", "mean"))
    mixture_table = comparison.groupby("algorithm", as_index=False).agg(
        mean_free_minus_fixed=("free_minus_fixed_Loss", "mean"), max_H_excess=("H_excess_over_star", "max"),
        mean_p_L1=("p_L1_from_star", "mean"))
    text = f"""# 第三问：粒子群、配比和隐含条件对照

## 采用的公式与来源

读取第二问固定E、450条去重B6/B7样本上拟合的additive_mixture和interaction_mixture参数，不重新拟合。E={config['fixed_E']:.12g}。参数保存为model_parameters.json。B6–B8为半合成补充，优化输出是模型预测，不是直接实验观测。

- additive: L=E+A*N^(-a)+B*D^(-b)+C*(1-Q)^g+lambda*H(p)。
- interaction: L=E+A*N^(-a)+B*D^(-b)+C*(1-Q)^g*D^(-d)+lambda*H(p)。
- 固定配比使用H(p_star)=H_star={config['mixture']['H_star']:.12g}，自由配比使用第一问导出的完整H(p)。H(p)=mean_j[F_j(p)]-mean_j[F_j(p_ref)]，p_ref为17域均匀配比，M2的tau={config['mixture']['tau']}。仅读第一问导出，不读原始A。

## 条件与上下文核查

N、D均用十亿单位。基础训练和注意力总成本严格为10^18*N*D*(6+eta*L_ctx)，eta=2e-4。质量成本为10^9*D*[f(Q)-f(0.4)]_+，三种f沿用原脚本：1e7*exp(6Q)、5e9*Q^4、2e9*ln(1+10Q)。预算约束是三项成本总和不超过预算。N在[0.005,100000]、D在[0.2,100000]、Q在[0.4,1]。自由配比p_k>=0且sum(p)=1。上下文是外生场景，不以缩短上下文获取无任务收益的最优值。

主对照按截图保守采用2048、4096，预算10^19、10^22、10^24。截图声称30000超出整张C7，但原表实际观测窗口为{config['supported_contexts']}，所以不能将4096称为整张C7上限。2048/4096分别对应注意力/基础训练成本6.8267%/13.6533%。理论相等点L_ctx^crit=6/eta=30000，是成本公式推导值，不是C7直接记录窗口；32768和131072等架构最大窗口已超过该点。特定架构能否使用某窗口仍取决于该架构自己的max_position_embeddings；连续优化N不等于挑选某个具体架构。其余C7窗口的模型敏感性另存context_sensitivity.csv，并明确其有限的可实现性。原实现已经使用正确的10^18换算，故这次没有额外乘一次10^18。

Q仍解释为固定领域构成下可通过域内筛选改变的整体质量。本次未引入未经观测的七到十七域质量映射、域级供应量或配比成本，故没有实现Q(p)。这些条件如要加入，需要额外数据和校准，不能凭空设定。

## 求解方法与比较协议

主求解器为纯投影粒子群PSO：{config['particles']}个粒子、{config['iterations']}轮、惯性0.9线性降至0.4、个体/群体系数均1.7，速度截断，100轮进行少量随机重启。配比通过欧氏单纯形投影处理，可得到零份额。没有SLSQP润色，没有将已知p_star放进自由配比初值。

对照SLSQP每次{config['slsqp_starts']}个初值、最多600步、ftol=1e-11。两算法同一seed生成相同起始候选，SLSQP取前若干个，PSO使用全群；工作量并不相等。每个场景重复{config['repetitions']}次，分别记录损失、耗时、目标函数调用量和可行性。matched_evaluations.csv另外在SLSQP总调用量以内截取PSO已完成批次的最好损失，属于调用量匹配诊断，并非严格相同墙钟预算。

由于所有模型在固定N,Q,p时损失随D严格下降，先精确消去D：D=min(D_max,预算/[10^18*N*(6+eta*L_ctx)+10^9*质量单token增量成本])。N上界随Q设置，确保D>=D_min。两算法使用相同可行参数化，避免原始10^24量级不等式造成数值条件差异。若D上界生效，允许预算不耗尽。优化自由度为固定配比2维、自由配比18维（19个坐标含一个配比和约束）。因此比较反映算法在同一重参数化问题上的表现，不能将改善全归因于替换旧求解器。

独立参考采用给定Q时对log N的一维有界优化，再扫描161个Q点、细化每个检测到的局部极小，并检查端点。给定Q的目标在log N上凸；Q方向仍是数值扫描，参考不宣称严格全局证明。参考命中标准为Loss差<=1e-6。

## 两公式的优化结果

下表为独立高精度参考（幂成本、上下文4096、固定p_star）。实际PSO/SLSQP结果和每次波动见optimization_runs.csv与optimization_summary.csv。

{md(table)}

不同公式的预测Loss只能比较各自模型下的最优策略，不能据此证明哪种策略在真实训练中更好。第二问分组验证中additive_mixture的预测表现略优于interaction_mixture。

## 重新优化p是否改善

本次lambda>0，Q与p可分离，成本也不依赖p。故min_p L=L_scale+lambda*min_p H(p)，自由配比理论最优与固定p_star完全一致，N、D、Q最优也相同。任何自由配比损失更高都是有限迭代/局部收敛的数值误差；极小的反向差可能来自资源子问题误差，不能当作配比收益。

{md(mixture_table)}

单独优化H的分离式PSO控制也已导出separable_pso_control.csv，展示利用问题结构降低维度后能否恢复同一解。它不参与上面的联合PSO/SLSQP计时排名。若引入Q(p)、领域供给或配比筛选成本，最优p才可能与先前不同，但本次公式并无这些机制。

## PSO与SLSQP的实测表现

{md(solver_table)}

PSO的潜在优势是不需梯度、批量计算方便、适合非光滑或复杂非凸扩展；对本次平滑且可分离问题，必须据实比较精度、波动和成本，不预设PSO更优。全局搜索特征也不构成全局最优保证。耗时为本机顺序运行，包含优化内的初始化/可行化，排除输入加载/参考求解/绘图，不能直接推广到其他机器。

## 输出与复现

运行python src/q3_optimization.py，默认使用本实验。可通过--repetitions、--particles、--iterations、--slsqp-starts控制工作量，通过--output-dir指定结果目录。SLSQP在同一问题上作为对照求解器运行。主输出在results/q3，保留逐次解、完整17域配比、成本份额、逐轮PSO收敛曲线及来源SHA256。

主模型由第二问分组OOF RMSE最小选择，本次为{config['Q2_selection']['model']}，OOF RMSE={config['Q2_selection']['OOF_RMSE']:.8f}。main_optimal_configurations.csv只保留该模型的实际PSO最优重复结果（固定/自由配比均列出），不按优化预测Loss高低选择公式。窗口条件引入前后PSO情景对照见constraints_before_after.csv与constraint_comparison.md。
"""
    (out / "optimization_analysis.md").write_text(text, encoding="utf-8")


def constraint_comparison(out, config, runs, params, mixture):
    """Context scenario ablation; never treat incorrect units as a valid baseline.

    Before: unconstrained theoretical 30000. After: screenshot's 2048/4096
    scenarios. This changes the task context and is NOT same-task performance.
    Both sides retain physical units, attention and identical resource bounds.
    """
    rows, pairs = [], []
    for model, t in params.items():
        for cost in COSTS:
            for budget in BUDGETS:
                problem = Problem(model, t, mixture, cost, budget, 30000, "fixed")
                ref_z, ref_loss = reference_solution(problem)
                for repetition in range(config["repetitions"]):
                    seed = config["seed_base"] + repetition
                    z, stats, _ = solve_pso(problem, seed, config["particles"], config["iterations"])
                    rows.append({"model": model, "cost_form": cost, "budget_FLOPs": budget,
                                 "L_ctx": 30000, "recipe_mode": "fixed", "algorithm": "PSO",
                                 "seed": seed, "constraint_stage": "before_theoretical_context",
                                 "reference_Loss": ref_loss, **stats, **problem.record(z)})
                    before = rows[-1]
                    after = runs[(runs.model == model) & (runs.cost_form == cost) &
                                 (runs.budget_FLOPs == budget) & (runs.recipe_mode == "fixed") &
                                 (runs.algorithm == "PSO") & (runs.seed == seed)]
                    for row in after.to_dict("records"):
                        rows.append({**row, "constraint_stage": "after_primary_context_scenario"})
                        pairs.append({"model": model, "cost_form": cost, "budget_FLOPs": budget,
                                      "seed": seed, "before_L_ctx": 30000, "after_L_ctx": row["L_ctx"],
                                      **{f"before_{k}": before[k] for k in ["N_B", "D_B", "Q", "Loss", "s_attn"]},
                                      **{f"after_{k}": row[k] for k in ["N_B", "D_B", "Q", "Loss", "s_attn"]},
                                      "delta_Loss_after_minus_before": row["Loss"] - before["Loss"]})
    save_csv(pd.DataFrame(rows), out / "constraint_scenario_runs.csv")
    save_csv(pd.DataFrame(pairs), out / "constraints_before_after.csv")
    summary = pd.DataFrame(pairs).groupby(["model", "budget_FLOPs", "after_L_ctx"], as_index=False).agg(
        mean_delta_Loss=("delta_Loss_after_minus_before", "mean"))
    fig, ax = plt.subplots(figsize=(8, 4))
    for (model, window), frame in summary.groupby(["model", "after_L_ctx"]):
        ax.plot(frame.budget_FLOPs, frame.mean_delta_Loss, "o-", label=f"{model}: 30000 -> {window}")
    ax.set(xscale="log", xlabel="Budget (FLOPs)", ylabel="Loss after - before",
           title="Context condition scenario comparison (PSO)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "08_constraints_before_after.png", dpi=180)
    plt.close(fig)
    (out / "constraint_comparison.md").write_text(
        "# 隐含条件引入前后对照\n\n"
        "前：允许理论临界窗口30000；后：主情景限定为2048/4096。相同公式、预算、固定p*、PSO参数和seed。"
        "两侧均使用正确的原始参数/token单位换算，并计入训练、质量与注意力成本。"
        "原代码已有正确单位，不能将错误单位的解当成合法的‘引入前’解。\n\n"
        "缩短窗口会减少成本，同时改变任务上下文，因此损失下降是模型内情景差异，不能当作同任务性能提升。"
        "30000为理论等成本点，未在C7直接记录；C7另有32768/131072等窗口，4096不是整张表的上限。"
        "连续N没有对应唯一架构，2048/4096属于保守主情景，而非所有架构必须遵守的统一硬上限。\n\n"
        "成对明细：constraints_before_after.csv；逐次配置：constraint_scenario_runs.csv。\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--slsqp-starts", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(BASE) / "results/q3")
    args = parser.parse_args()
    if min(args.repetitions, args.iterations, args.slsqp_starts) < 1 or args.particles < 4:
        parser.error("Positive counts and at least four particles are required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    params, mixture, c7, supported, hashes = load_inputs()
    forms = tuple(params)
    config = {"fixed_E": params["additive"]["E"], "mixture": mixture, "parameters": params,
              "supported_contexts": supported, "primary_contexts": [2048, 4096], "theoretical_critical_context": 6 / ETA,
              "particles": args.particles, "iterations": args.iterations, "repetitions": args.repetitions,
              "slsqp_starts": args.slsqp_starts, "seed_base": 20261009,
              "sources_sha256": hashes, "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "assumptions": ["Q can vary within a fixed recipe", "A mixture response transfers to B up to lambda",
                              "No p-dependent costs or supply limits", "D loss derivative is negative",
                              "C7 max windows constrain specific architectures, not every continuous N"],
              "default_solver": "PSO", "main_model": mixture["main_model"],
              "Q2_selection": mixture["selection"], "reference_tolerance": 1e-6}
    (out / "model_parameters.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    save_csv(c7, out / "context_architecture_audit.csv")
    runs, references, histories, matched = [], [], [], []
    for form in forms:
        for cost_form in COSTS:
            for budget in BUDGETS:
                for window in [2048, 4096]:
                    fixed = Problem(form, params[form], mixture, cost_form, budget, window, "fixed")
                    ref_z, ref_loss = reference_solution(fixed)
                    meta = {"model": form, "cost_form": cost_form, "budget_FLOPs": budget, "L_ctx": window}
                    references.append({**meta, **fixed.record(ref_z)})
                    for mode in ["fixed", "free"]:
                        problem = Problem(form, params[form], mixture, cost_form, budget, window, mode)
                        for repetition in range(args.repetitions):
                            seed = 20261009 + repetition
                            pso_z, pso_stats, history = solve_pso(problem, seed, args.particles, args.iterations)
                            sqp_z, sqp_stats, _ = solve_slsqp(problem, seed, args.slsqp_starts,
                                                             population_size=args.particles)
                            for algorithm, z, stats in [("PSO", pso_z, pso_stats), ("SLSQP", sqp_z, sqp_stats)]:
                                record = problem.record(z)
                                gap = record["Loss"] - ref_loss
                                runs.append({**meta, "recipe_mode": mode, "algorithm": algorithm, "seed": seed,
                                             **stats, **record, "reference_Loss": ref_loss,
                                             "gap_to_reference": gap, "reference_hit": abs(gap) <= 1e-6})
                            for entry in history:
                                histories.append({**meta, "recipe_mode": mode, "seed": seed, **entry,
                                                  "gap_to_reference": entry["best_Loss"] - ref_loss})
                            eligible = [entry for entry in history if entry["evaluations"] <= sqp_stats["evaluations"]]
                            equal = eligible[-1] if eligible else None
                            matched.append({**meta, "recipe_mode": mode, "seed": seed,
                                            "SLSQP_evaluations": sqp_stats["evaluations"],
                                            "SLSQP_Loss": float(problem.objective(sqp_z)[0]),
                                            "PSO_evaluations_at_match": equal["evaluations"] if equal else np.nan,
                                            "PSO_Loss_at_match": equal["best_Loss"] if equal else np.nan})
                    print(f"{form} {cost_form} C={budget:.0e} ctx={window}: reference={ref_loss:.8f}", flush=True)
                    # Preserve useful partial progress for longer reruns.
                    save_csv(pd.DataFrame(runs), out / "optimization_runs.csv")
    runs, references, histories = pd.DataFrame(runs), pd.DataFrame(references), pd.DataFrame(histories)
    summary, best, comparison = summarize(runs)
    main_runs = runs[(runs.model == config["main_model"]) & (runs.algorithm == "PSO")]
    selected_runs = main_runs.loc[main_runs.groupby(KEYS).Loss.idxmin()].copy()
    save_csv(selected_runs, out / "main_optimal_configurations.csv")
    for filename, table in [("optimization_summary.csv", summary), ("best_configurations.csv", best),
                            ("reference_optima.csv", references), ("fixed_vs_free_recipe.csv", comparison),
                            ("pso_convergence.csv", histories), ("matched_evaluations.csv", pd.DataFrame(matched))]:
        save_csv(table, out / filename)
    # Exploit separability as a transparent additional control, with pure PSO.
    separated = []
    class MixtureOnly:
        mode, dimension = "free", 17
        resource_dimensions = 0
        def population(self, rng, size):
            return rng.dirichlet(np.ones(17), size)
        def objective(self, points):
            return fixed.h(points)
        def repair(self, points):
            return project_simplex(points)
    for repetition in range(args.repetitions):
        seed = 20261009 + repetition
        recipe, stats, _ = solve_pso(MixtureOnly(), seed, args.particles, args.iterations)
        separated.append({"seed": seed, **stats, "H_p": float(fixed.h(recipe)),
                          "H_excess": float(fixed.h(recipe) - mixture["H_star"]),
                          "p_L1_from_star": float(np.abs(recipe - fixed.p_star).sum()),
                          **{f"p_{domain}": float(value) for domain, value in zip(MIX_DOMAINS, recipe)}})
    save_csv(pd.DataFrame(separated), out / "separable_pso_control.csv")
    sensitivity = []
    for form in forms:
        for budget in BUDGETS:
            for window in sorted(set(supported + [30000])):
                problem = Problem(form, params[form], mixture, "power", budget, window, "fixed")
                z, _ = reference_solution(problem)
                sensitivity.append({"model": form, "budget_FLOPs": budget, "L_ctx": window,
                                    "window_recorded_in_C7": window in supported,
                                    "window_role": "theoretical_equality" if window == 30000 else "observed_architecture_maximum",
                                    **problem.record(z)})
    save_csv(pd.DataFrame(sensitivity), out / "context_sensitivity.csv")
    constraint_comparison(out, config, runs, params, mixture)
    make_figures(out, runs, references, best, histories, supported)
    write_report(out, config, references, summary, comparison, runs)
    print(summary.groupby(["recipe_mode", "algorithm"])[["mean_gap", "max_gap", "median_seconds", "reference_hit_rate"]].mean().to_string(), flush=True)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
