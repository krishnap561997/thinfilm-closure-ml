#!/usr/bin/env python3
"""Generate per-case extraction JSON files and Slurm submission scripts.

Run on HiPerGator:
  python3 generate_extraction_jobs.py
  /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/datasets/BasC/DNS/param_space/ClosureData/run_extraction

Use --extractor PATH if the extraction script is not in PARAM_SPACE's parent
repository root. Generation does not submit jobs; run_extraction does.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path


DEFAULT_ROOT = Path(
    "/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/"
    "thinfilm-closure-ml/datasets/BasC/DNS/param_space"
)
CASE_PATTERN = re.compile(r"Re\d{3}_Ca\d+_\d{3}_theta\d+(?:_\d+)?\Z")
SNAPSHOT_PATTERN = re.compile(r"snapshot_(\d{6})\.h5\Z")


def write_if_changed(path: Path, content: str, executable: bool = False) -> None:
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--extractor", type=Path, help="Path to extract_closure_terms_latest.py")
    parser.add_argument(
        "--python-env",
        type=Path,
        default=Path("/blue/bala1s/krishnap.kalivel/TLFHydrodynamics/"
                     "thinfilm-closure-ml/software/envs/thinfilm-ml-py311/bin"),
        help="Directory containing python3 for extraction",
    )
    parser.add_argument("--step", type=int, default=10, help="Snapshot label interval")
    parser.add_argument("--max-time", type=int, default=2000, help="Largest snapshot label to process")
    parser.add_argument("--walltime", default="04:00:00", help="Slurm wall time per case")
    parser.add_argument("--memory", default="16G", help="Slurm memory per case")
    parser.add_argument("--account", default="bala1s")
    parser.add_argument("--qos", default="bala1s")
    parser.add_argument("--partition", default="hpg-dev")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Simulation root does not exist: {root}")
    if args.step <= 0 or args.max_time < 0:
        parser.error("--step must be positive and --max-time nonnegative")

    extractor = (args.extractor or root.parents[2] / "extract_closure_terms_latest.py")
    extractor = extractor.expanduser().resolve()
    if not extractor.is_file():
        parser.error(f"Extractor not found: {extractor}; supply --extractor PATH")

    destination = root / "ClosureData"
    configs = destination / "configs"
    logs = destination / "logs"
    configs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    ready_configs: list[Path] = []
    for case in sorted(root.iterdir()):
        if not case.is_dir() or not CASE_PATTERN.fullmatch(case.name):
            continue

        output_dir = case / "output"
        labels = sorted(
            int(match.group(1))
            for path in output_dir.glob("snapshot_*.h5")
            if (match := SNAPSHOT_PATTERN.fullmatch(path.name)) is not None
        ) if output_dir.is_dir() else []
        selected = [label for label in labels if 10 <= label <= args.max_time]
        if not selected:
            print(f"SKIP {case.name}: no snapshots at or below {args.max_time}")
            skipped += 1
            continue

        expected = list(range(selected[0], selected[-1] + 1, args.step))
        if selected != expected:
            missing = sorted(set(expected) - set(selected))
            print(
                f"SKIP {case.name}: {len(selected)} snapshots within cap, "
                f"nonuniform labels or missing expected labels "
                f"(first 10 missing: {missing[:10]})."
            )
            skipped += 1
            continue

        config = {
            "snapshot_dir": str(output_dir),
            "output_file": str(destination / f"thinfilm_closure_data_{case.name}.h5"),
            "start_time": selected[0],
            "end_time": selected[-1],
            "delta_time": args.step,
            "snapshot_prefix": "snapshot_",
            "snapshot_suffix": ".h5",
            "filename_width": 6,
            "filename_decimals": 0,
            "wall_shear_mu": 1.0,
            "lower_boundary": 0.0,
            "compression": "gzip",
            "compression_level": 4,
            # Informational; the extractor ignores unknown configuration keys.
            "available_snapshot_count": len(labels),
            "selected_snapshot_count": len(selected),
        }
        config_path = configs / f"{case.name}.json"
        write_if_changed(config_path, json.dumps(config, indent=4) + "\n")
        ready_configs.append(config_path)
        print(f"READY {case.name}: {len(selected)} snapshots, {selected[0]}–{selected[-1]}")
        generated += 1

    write_if_changed(
        destination / "ready_configs.txt",
        "".join(f"{path}\n" for path in ready_configs),
    )

    run_sh = destination / "run.sh"
    write_if_changed(run_sh, f'''#!/usr/bin/env bash
#SBATCH --job-name=extract
#SBATCH --output={logs}/%x_%j.out
#SBATCH --error={logs}/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={args.memory}
#SBATCH --time={args.walltime}
#SBATCH --account={args.account}
#SBATCH --qos={args.qos}
#SBATCH --partition={args.partition}

set -euo pipefail

# Mirror the original job environment while allowing submission from anywhere.
cd "${{SLURM_SUBMIT_DIR}}"
export PATH={shlex.quote(str(args.python_env.expanduser()))}:"$PATH"
PYTHON_SCRIPT={shlex.quote(str(extractor))}
CONFIG_FILE="${{1:-}}"

echo "============================================================"
echo "Thin-film closure extraction"
echo "============================================================"
echo "Job ID        : ${{SLURM_JOB_ID}}"
echo "Node          : $(hostname)"
echo "Working dir   : $(pwd)"
echo "Python        : $(command -v python3)"
echo "Config        : ${{CONFIG_FILE}}"
echo "Start time    : $(date)"
echo "============================================================"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: Cannot find $PYTHON_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Cannot find $CONFIG_FILE" >&2
    exit 1
fi

echo "Config file: $(realpath "$CONFIG_FILE")"
echo "Config contents:"
cat "$CONFIG_FILE"

python3 -u "$PYTHON_SCRIPT" "$CONFIG_FILE"
echo "Finished successfully"
echo "End time: $(date)"
''', executable=True)

    launcher = destination / "run_extraction"
    write_if_changed(launcher, '''#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -s "$here/ready_configs.txt" ]]; then
    echo "No ready case JSON files found in $here/ready_configs.txt" >&2
    exit 1
fi
while IFS= read -r config; do
    echo "Submitting $(basename "$config")"
    sbatch "$here/run.sh" "$config"
done < "$here/ready_configs.txt"
''', executable=True)
    print(f"\nGenerated {generated} configs; skipped {skipped} cases.")
    print(f"To submit: {launcher}")
    print("Each run of run_extraction submits all generated configs again.")


if __name__ == "__main__":
    main()
