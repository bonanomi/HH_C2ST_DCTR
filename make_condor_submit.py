#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HTCondor wrapper and submit files for C2ST/DCTR training."
    )
    parser.add_argument("--task", choices=("main", "closure"), required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--conda-base", default="~/miniforge3")
    parser.add_argument("--conda-env", default="c2st")
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="24GB")
    parser.add_argument("--disk", default="10GB")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--runtime-hours", type=float, default=None)
    parser.add_argument("--job-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--extra-classad", action="append", default=[])
    parser.add_argument("--environment", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--submit", action="store_true")

    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def shell_join(items):
    return " ".join(shlex.quote(str(x)) for x in items)


def build_command(args, extra):
    if args.task == "main":
        if args.channels is not None or args.folds is not None:
            raise ValueError("--channels/--folds are only valid for --task closure.")
        cmd = ["python", "-u", "c2st_nn.py"]
    else:
        cmd = ["python", "-u", "-m", "c2st_final_closure.train_dctr_crossfit_closure"]
        if args.channels:
            cmd += ["--channels", *args.channels]
        if args.folds is not None:
            if args.folds < 2:
                raise ValueError("--folds must be >= 2.")
            cmd += ["--folds", str(args.folds)]
    return cmd + extra


def main():
    args, extra = parse_args()

    if args.cpus < 1:
        raise ValueError("--cpus must be >= 1.")
    if args.gpus < 0:
        raise ValueError("--gpus must be >= 0.")
    if args.runtime_hours is not None and args.runtime_hours <= 0:
        raise ValueError("--runtime-hours must be > 0.")

    repo = args.repo_dir.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)

    for item in args.environment:
        if "=" not in item:
            raise ValueError(f"Expected NAME=VALUE, got {item!r}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = args.job_name or args.task
    job_dir = (
        args.job_dir.expanduser().resolve()
        if args.job_dir
        else repo / "condor_jobs" / f"{job_name}_{timestamp}"
    )
    logs = job_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    cmd = build_command(args, extra)

    if args.conda_base.startswith("~/"):
        conda_assignment = f'CONDA_BASE="$HOME/{args.conda_base[2:]}"'
    else:
        conda_assignment = f"CONDA_BASE={shlex.quote(args.conda_base)}"

    exports = "\n".join(
        f"export {shlex.quote(k)}={shlex.quote(v)}"
        for k, v in (item.split("=", 1) for item in args.environment)
    )

    wrapper = job_dir / "run.sh"
    wrapper_text = f'''#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "C2ST HTCondor job"
echo "started: $(date)"
echo "host:    $(hostname)"
echo "============================================================"

{conda_assignment}
if [[ ! -f "$CONDA_BASE/bin/activate" ]]; then
    echo "ERROR: missing $CONDA_BASE/bin/activate"
    exit 2
fi

source "$CONDA_BASE/bin/activate"
conda activate {shlex.quote(args.conda_env)}

cd {shlex.quote(str(repo))}

echo
echo "Repository: $(pwd)"
echo "Git revision:"
git rev-parse HEAD 2>/dev/null || true

echo
echo "Python:"
which python
python --version

echo
echo "TensorFlow devices:"
python -c 'import tensorflow as tf; print("TensorFlow:", tf.__version__); print("GPUs:", tf.config.list_physical_devices("GPU"))'

if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    nvidia-smi || true
fi

{exports}

echo
echo "Command:"
echo {shlex.quote(shell_join(cmd))}
echo "============================================================"

{shell_join(cmd)}
status=$?

echo "============================================================"
echo "finished: $(date)"
echo "exit code: $status"
echo "============================================================"

exit "$status"
'''
    wrapper.write_text(wrapper_text)
    wrapper.chmod(0o755)

    submit = job_dir / "job.sub"
    lines = [
        "universe = vanilla",
        f"executable = {wrapper}",
        "",
        "should_transfer_files = NO",
        "",
        f"request_cpus = {args.cpus}",
        f"request_memory = {args.memory}",
        f"request_disk = {args.disk}",
    ]

    if args.gpus > 0:
        lines.append(f"request_gpus = {args.gpus}")

    if args.runtime_hours is not None:
        seconds = int(round(args.runtime_hours * 3600))
        lines += [
            "",
            "# Common runtime ClassAd; verify this name for your local HTCondor setup.",
            f"+RequestRuntime = {seconds}",
        ]

    if args.extra_classad:
        lines += ["", "# User-supplied submit directives"]
        lines.extend(args.extra_classad)

    lines += [
        "",
        f"output = {logs}/$(ClusterId).$(ProcId).out",
        f"error  = {logs}/$(ClusterId).$(ProcId).err",
        f"log    = {logs}/$(ClusterId).log",
        "",
        "getenv = True",
        "",
        "queue 1",
        "",
    ]
    submit.write_text("\n".join(lines))

    print(f"Generated wrapper: {wrapper}")
    print(f"Generated submit:  {submit}")
    print(f"Log directory:     {logs}")
    print()
    print("Training command:")
    print(" ", shell_join(cmd))
    print()
    print("Submit with:")
    print(" ", f"condor_submit {shlex.quote(str(submit))}")

    if args.submit:
        exe = shutil.which("condor_submit")
        if exe is None:
            raise RuntimeError("condor_submit is not available in PATH.")
        subprocess.run([exe, str(submit)], check=True)


if __name__ == "__main__":
    main()
