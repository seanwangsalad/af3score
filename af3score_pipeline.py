import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()


def str2bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_cmd(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only AF3Score pipeline runner.")
    parser.add_argument("--input", required=True, help="Directory containing input PDB files.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable for subprocess calls.")
    parser.add_argument("--weights", default=None, help="Absolute path to AlphaFold3 model weights file.")
    parser.add_argument("--db_dir", action="append", default=None, help="AlphaFold3 database directory. Repeatable.")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker count for preprocessing scripts.")
    parser.add_argument("--run_data_pipeline", type=str2bool, default=False)
    parser.add_argument("--run_inference", type=str2bool, default=True)
    parser.add_argument("--resume", type=str2bool, default=False, help="Skip completed AF3 outputs and resume interrupted runs.")

    # Script paths are configurable; defaults resolve relative to this file.
    parser.add_argument("--extract_script", default=str(_SCRIPT_DIR / "1_extract_chains.py"))
    parser.add_argument("--pdb2jax_script", default=str(_SCRIPT_DIR / "2_pdb2jax.py"))
    parser.add_argument("--json_script", default=str(_SCRIPT_DIR / "3_generate_json.py"))
    parser.add_argument("--af3score_script", default=str(_SCRIPT_DIR / "run_af3score.py"))
    parser.add_argument("--metrics_script", default=str(_SCRIPT_DIR / "04_get_metrics.py"))

    # Output locations are configurable.
    parser.add_argument("--cif_dir", default=None, help="Override chain CIF output directory.")
    parser.add_argument("--sequence_csv", default=None, help="Override chain sequence CSV path.")
    parser.add_argument("--h5_dir", default=None, help="Override H5 output directory.")
    parser.add_argument("--json_dir", default=None, help="Override JSON output directory.")
    parser.add_argument("--af3_output_dir", default=None, help="Override AF3 output directory.")
    parser.add_argument("--metric_csv", default=None, help="Override metrics CSV output path.")

    args, passthrough = parser.parse_known_args()

    if args.run_inference and not args.weights:
        parser.error("--weights is required when --run_inference is true.")

    weights_path = Path(args.weights).expanduser().resolve() if args.weights else None
    if args.run_inference and (not weights_path or not weights_path.is_file()):
        parser.error("--weights must point to an existing weight file.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_dir = Path(args.cif_dir) if args.cif_dir else output_dir / "complex_chain_cifs"
    seq_csv = Path(args.sequence_csv) if args.sequence_csv else output_dir / "complex_chain_sequences.csv"
    h5_dir = Path(args.h5_dir) if args.h5_dir else output_dir / "complex_h5"
    json_dir = Path(args.json_dir) if args.json_dir else output_dir / "complex_json_files"
    af3_output_dir = Path(args.af3_output_dir) if args.af3_output_dir else output_dir / "af3score_outputs"
    metric_csv = Path(args.metric_csv) if args.metric_csv else output_dir / "af3score_metrics.csv"

    run_cmd([
        args.python_exec,
        args.extract_script,
        "--input_dir", args.input,
        "--output_dir_cif", str(cif_dir),
        "--save_csv", str(seq_csv),
        "--num_workers", str(args.num_workers),
    ])

    run_cmd([
        args.python_exec,
        args.pdb2jax_script,
        "--pdb_dir", args.input,
        "--output_dir", str(h5_dir),
        "--num_workers", str(args.num_workers),
    ])

    run_cmd([
        args.python_exec,
        args.json_script,
        "--sequence_csv", str(seq_csv),
        "--cif_dir", str(cif_dir),
        "--output_dir", str(json_dir),
    ])

    af3_cmd = [
        args.python_exec,
        args.af3score_script,
        f"--batch_json_dir={json_dir}",
        f"--batch_h5_dir={h5_dir}",
        f"--output_dir={af3_output_dir}",
        f"--run_data_pipeline={str(args.run_data_pipeline).lower()}",
        f"--run_inference={str(args.run_inference).lower()}",
        f"--resume={str(args.resume).lower()}",
        *passthrough,
    ]

    if weights_path:
        af3_cmd.append(f"--model_dir={weights_path.parent}")
    for db_dir in args.db_dir or []:
        af3_cmd.append(f"--db_dir={db_dir}")

    run_cmd(af3_cmd)

    run_cmd([
        args.python_exec,
        args.metrics_script,
        "--input_pdb_dir", args.input,
        "--af3score_output_dir", str(af3_output_dir),
        "--save_metric_csv", str(metric_csv),
    ])


if __name__ == "__main__":
    main()
