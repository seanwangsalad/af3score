import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run af3score_pipeline.py for multiple input directories.")
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_parent_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--db_dir", required=True)
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
            "af3score_pipeline.py",
            "--input_pdb_dir", input_dir,
            "--output_dir", str(out_dir),
            "--model_dir", args.model_dir,
            "--db_dir", args.db_dir,
            "--num_workers", str(args.num_workers),
            *passthrough,
        ]
        print("▶", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
