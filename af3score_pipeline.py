import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only AF3Score pipeline runner.")
    parser.add_argument("--input_pdb_dir", required=True, help="Directory containing input PDB files.")
    parser.add_argument("--output_dir", required=True, help="Base output directory.")
    parser.add_argument("--python_exec", default=sys.executable, help="Python executable for subprocess calls.")
    parser.add_argument("--model_dir", required=True, help="AlphaFold3 model directory.")
    parser.add_argument("--db_dir", required=True, help="AlphaFold3 database directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="Worker count for preprocessing scripts.")
    parser.add_argument("--run_data_pipeline", default="False")
    parser.add_argument("--run_inference", default="true")

    # Script paths are configurable (no hardcoded absolute paths).
    parser.add_argument("--extract_script", default="1_extract_chains.py")
    parser.add_argument("--pdb2jax_script", default="2_pdb2jax.py")
    parser.add_argument("--json_script", default="3_generate_json.py")
    parser.add_argument("--af3score_script", default="run_af3score.py")
    parser.add_argument("--metrics_script", default="04_get_metrics.py")

    # Output locations are configurable.
    parser.add_argument("--cif_dir", default=None, help="Override chain CIF output directory.")
    parser.add_argument("--sequence_csv", default=None, help="Override chain sequence CSV path.")
    parser.add_argument("--h5_dir", default=None, help="Override H5 output directory.")
    parser.add_argument("--json_dir", default=None, help="Override JSON output directory.")
    parser.add_argument("--af3_output_dir", default=None, help="Override AF3 output directory.")
    parser.add_argument("--metric_csv", default=None, help="Override metrics CSV output path.")

    args, passthrough = parser.parse_known_args()

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
        "--input_dir", args.input_pdb_dir,
        "--output_dir_cif", str(cif_dir),
        "--save_csv", str(seq_csv),
        "--num_workers", str(args.num_workers),
    ])

    run_cmd([
        args.python_exec,
        args.pdb2jax_script,
        "--pdb_dir", args.input_pdb_dir,
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

    run_cmd([
        args.python_exec,
        args.af3score_script,
        f"--db_dir={args.db_dir}",
        f"--model_dir={args.model_dir}",
        f"--batch_json_dir={json_dir}",
        f"--batch_h5_dir={h5_dir}",
        f"--output_dir={af3_output_dir}",
        f"--run_data_pipeline={args.run_data_pipeline}",
        f"--run_inference={args.run_inference}",
        *passthrough,
    ])

    run_cmd([
        args.python_exec,
        args.metrics_script,
        "--input_pdb_dir", args.input_pdb_dir,
        "--af3score_output_dir", str(af3_output_dir),
        "--save_metric_csv", str(metric_csv),
    ])


if __name__ == "__main__":
    main()
