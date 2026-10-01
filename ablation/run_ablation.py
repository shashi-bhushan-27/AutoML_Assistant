#!/usr/bin/env python
"""
RAG-AutoML ablation study — entry point.

The harness imports the application unmodified and only wraps/subclasses/
monkeypatches it inside this process (see README.md in this folder).

    python ablation/run_ablation.py prepare                 # download datasets
    python ablation/run_ablation.py units --workers 4       # brute force + ablations B and C
    python ablation/run_ablation.py llm --calls 10          # ablation A (Groq calls)
    python ablation/run_ablation.py checks                  # targeted code-behaviour checks
    python ablation/run_ablation.py aggregate               # results/*.csv|json + LaTeX table
    python ablation/run_ablation.py figures                 # figures/*.pdf|png
    python ablation/run_ablation.py all --workers 4 --calls 10

Requires GROQ_API_KEY in the environment for ``llm`` (never logged).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --out NAME (or ABLATION_OUT=NAME) writes to ablation/NAME and ablation/figures<suffix>; must be set before
# common.py is imported. Subprocesses inherit it through the environment.
if "--out" in sys.argv:
    _i = sys.argv.index("--out")
    os.environ["ABLATION_OUT"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]

from common import CACHE_DIR, DATASETS, RAW_DIR, SEEDS, dump_json, environment_info, load_dataset  # noqa: E402

try:  # Windows: the app package loads torch before anything imports scikit-learn (see app_backend/__init__.py)
    import app_backend  # noqa: E402,F401
except ImportError:
    pass

LOG_DIR = os.path.join(CACHE_DIR, "logs")


def cmd_prepare(args):
    for ds in DATASETS:
        df = load_dataset(ds)
        print(ds, df.shape)
    dump_json(environment_info(), os.path.join(RAW_DIR, "environment.json"))


def cmd_preproc_only(args):
    from unit_runner import rerun_preproc

    for s in args.seeds:
        rerun_preproc(args.dataset, s)


def cmd_shap_rerun(args):
    """Re-run SHAP jobs whose worker was killed by SIGKILL (container OOM), at most --workers at a time."""
    from unit_runner import find_killed_shap_jobs

    todo = find_killed_shap_jobs()
    print("killed SHAP jobs:", {f"{d}_s{s}": j for (d, s), j in todo.items()}, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    queue = list(todo.items())
    running = []
    while queue or running:
        while queue and len(running) < args.workers:
            (ds, s), jobs = queue.pop(0)
            code = (f"import sys; sys.path.insert(0, {HERE!r}); from unit_runner import rerun_shap_jobs; "
                    f"rerun_shap_jobs({ds!r}, {s}, {jobs!r}, shap_timeout={args.shap_timeout})")
            running.append(subprocess.Popen([sys.executable, "-c", code], env=env))
        running = [p for p in running if p.poll() is None]
        time.sleep(2)


def cmd_unit(args):
    from unit_runner import run_unit

    run_unit(args.dataset, args.seed, do_preproc=not args.no_preproc, do_shap=not args.no_shap,
             shap_timeout=args.shap_timeout)


def _unit_done(ds, seed, need_shap=True):
    import json

    p = os.path.join(RAW_DIR, f"unit_{ds}_s{seed}.json")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        u = json.load(f)
    return ("shap" in u or not need_shap) and "preproc" in u


def cmd_units(args):
    """Run (dataset, seed) units as parallel single-threaded subprocesses."""
    os.makedirs(LOG_DIR, exist_ok=True)
    datasets = args.datasets or DATASETS
    seeds = args.seeds if args.seeds is not None else SEEDS
    # seed-0 units first (the LLM stage needs their brute-force results), longest dataset first
    order = sorted(((ds, s) for ds in datasets for s in seeds),
                   key=lambda t: (t[1] != 0, t[0] != "credit", t[1], t[0]))
    queue = [t for t in order if args.force or not _unit_done(*t, need_shap=not args.no_shap)]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    running = {}
    while queue or running:
        while queue and len(running) < args.workers:
            ds, s = queue.pop(0)
            cmd = [sys.executable, os.path.abspath(__file__), "unit", "--dataset", ds, "--seed", str(s),
                   "--shap-timeout", str(args.shap_timeout)]
            if args.no_shap:
                cmd.append("--no-shap")
            if args.no_preproc:
                cmd.append("--no-preproc")
            log = open(os.path.join(LOG_DIR, f"unit_{ds}_s{s}.log"), "w")
            running[(ds, s)] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print(f"[{time.strftime('%H:%M:%S')}] started {ds} seed {s}", flush=True)
        for key, (proc, log) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                print(f"[{time.strftime('%H:%M:%S')}] finished {key[0]} seed {key[1]} (exit {proc.returncode})",
                      flush=True)
                del running[key]
        time.sleep(5)


def cmd_llm(args):
    from rag_ablation import collect_llm_calls

    try:  # the app reads the key from .env; do the same (the key is never printed)
        from dotenv import load_dotenv

        load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))
    except ImportError:
        pass
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set")
    datasets = args.datasets or DATASETS
    # wait for the seed-0 brute-force checkpoints (priors for meta-learning)
    while not all(os.path.exists(os.path.join(RAW_DIR, f"unit_{ds}_s0.json")) for ds in DATASETS):
        print("[llm] waiting for seed-0 brute-force results ...", flush=True)
        time.sleep(60)
    collect_llm_calls(datasets, args.calls, args.model, configs=args.configs, pause_s=args.pause)


def cmd_checks(args):
    from checks import run_checks

    run_checks()


def cmd_aggregate(args):
    from aggregate import aggregate_all

    aggregate_all()


def cmd_figures(args):
    from figures import make_all_figures

    make_all_figures()


def cmd_all(args):
    cmd_prepare(args)
    cmd_checks(args)
    # training units in the background, LLM calls in the foreground (they wait for seed 0)
    units = subprocess.Popen([sys.executable, os.path.abspath(__file__), "units", "--workers", str(args.workers),
                              "--shap-timeout", str(args.shap_timeout)])
    cmd_llm(args)
    units.wait()
    cmd_aggregate(args)
    cmd_figures(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare")

    p = sub.add_parser("unit")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-preproc", action="store_true")
    p.add_argument("--shap-timeout", type=float, default=300.0)

    p = sub.add_parser("preproc-only", help="re-run ablation B for existing units")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--seeds", nargs="+", type=int, required=True)

    p = sub.add_parser("shap-rerun", help="re-run SHAP jobs killed by SIGKILL (OOM)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--shap-timeout", type=float, default=300.0)

    p = sub.add_parser("units")
    p.add_argument("--datasets", nargs="*", choices=DATASETS)
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-preproc", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--shap-timeout", type=float, default=300.0)

    for name in ("llm", "all"):
        p = sub.add_parser(name)
        p.add_argument("--datasets", nargs="*", choices=DATASETS)
        p.add_argument("--calls", type=int, default=10)
        p.add_argument("--model", default="openai/gpt-oss-120b",
                       help="substitute Groq model (the app's llama-3.3-70b-versatile is no longer served)")
        p.add_argument("--configs", nargs="*")
        p.add_argument("--pause", type=float, default=8.0, help="seconds between LLM calls (rate limits)")
        if name == "all":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--shap-timeout", type=float, default=300.0)

    sub.add_parser("checks")
    sub.add_parser("aggregate")
    sub.add_parser("figures")

    args = ap.parse_args()
    {"prepare": cmd_prepare, "unit": cmd_unit, "preproc-only": cmd_preproc_only, "shap-rerun": cmd_shap_rerun, "units": cmd_units, "llm": cmd_llm, "checks": cmd_checks,
     "aggregate": cmd_aggregate, "figures": cmd_figures, "all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
