import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("▶", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Python-only AF3Score pipeline runner.")
    parser.add_argument("--input_pdb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--python_exec", default=sys.executable)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--db_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--run_data_pipeline", default="False")
    parser.add_argument("--run_inference", default="true")
    args, passthrough = parser.parse_known_args()

    output_dir = Path(args.output_dir)
    cif_dir = output_dir / "complex_chain_cifs"
    seq_csv = output_dir / "complex_chain_sequences.csv"
    h5_dir = output_dir / "complex_h5"
    json_dir = output_dir / "complex_json_files"
    af3_output_dir = output_dir / "af3score_outputs"
    metric_csv = output_dir / "af3score_metrics.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cmd([
        args.python_exec,
        "1_extract_chains.py",
        "--input_dir", args.input_pdb_dir,
        "--output_dir_cif", str(cif_dir),
        "--save_csv", str(seq_csv),
        "--num_workers", str(args.num_workers),
    ])

    run_cmd([
        args.python_exec,
        "2_pdb2jax.py",
        "--pdb_dir", args.input_pdb_dir,
        "--output_dir", str(h5_dir),
        "--num_workers", str(args.num_workers),
    ])

    run_cmd([
        args.python_exec,
        "3_generate_json.py",
        "--sequence_csv", str(seq_csv),
        "--cif_dir", str(cif_dir),
        "--output_dir", str(json_dir),
    ])

    run_cmd([
        args.python_exec,
        "run_af3score.py",
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
        "04_get_metrics.py",
        "--input_pdb_dir", args.input_pdb_dir,
        "--af3score_output_dir", str(af3_output_dir),
        "--save_metric_csv", str(metric_csv),
    ])


if __name__ == "__main__":
    main()
