import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run af3score_pipeline.py for multiple input directories.")
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_parent_dir", required=True)
    parser.add_argument("--pipeline_script", default="af3score_pipeline.py", help="Path to pipeline runner script.")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--db_dir", action="append", default=None, help="Repeatable database directory argument.")
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--num_workers", type=int, default=4)
    args, passthrough = parser.parse_known_args()

    output_parent = Path(args.output_parent_dir)
    output_parent.mkdir(parents=True, exist_ok=True)

    for input_dir in args.input_dirs:
        name = Path(input_dir).name
        out_dir = output_parent / f"{name}_af3score"
        cmd = [
            args.python_exec,
            args.pipeline_script,
            "--input_pdb_dir", input_dir,
            "--output_dir", str(out_dir),
            "--num_workers", str(args.num_workers),
            *passthrough,
        ]
        if args.model_dir:
            cmd.extend(["--model_dir", args.model_dir])
        for db_dir in args.db_dir or []:
            cmd.extend(["--db_dir", db_dir])
        print("▶", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
