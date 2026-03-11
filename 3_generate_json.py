import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def format_msa_sequence(sequence: str) -> str:
    return f">query\n{sequence}\n"


def get_chain_sequences_from_row(row: pd.Series):
    chain_sequences = []
    for col in row.index:
        if col.startswith("chain_") and col.endswith("_seq") and pd.notna(row[col]) and row[col] != "":
            chain_sequences.append((col.split("_")[1], row[col]))
    return chain_sequences


def generate_json(row: pd.Series, cif_dir: Path, output_dir: Path):
    complex_name = row["complex"]
    chain_sequences = get_chain_sequences_from_row(row)

    sequences = []
    for chain_id, sequence in chain_sequences:
        cif_path = cif_dir / f"{complex_name}_chain_{chain_id}.cif"
        if not cif_path.exists():
            continue
        sequences.append(
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": format_msa_sequence(sequence),
                    "pairedMsa": format_msa_sequence(sequence),
                    "templates": [{
                        "mmcifPath": str(cif_path),
                        "queryIndices": list(range(len(sequence))),
                        "templateIndices": list(range(len(sequence))),
                    }],
                }
            }
        )

    if not sequences:
        return False

    payload = {
        "dialect": "alphafold3",
        "version": 1,
        "name": complex_name,
        "sequences": sequences,
        "modelSeeds": [10],
        "bondedAtomPairs": None,
        "userCCD": None,
    }

    out_path = output_dir / f"{complex_name}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AF3 JSON files from chain sequence CSV.")
    parser.add_argument("--sequence_csv", default="./complex_chain_sequences.csv", help="CSV produced by 1_extract_chains.py")
    parser.add_argument("--cif_dir", default="./complex_chain_cifs", help="Directory containing per-chain CIF files")
    parser.add_argument("--output_dir", default="./complex_json_files", help="Directory for output JSON files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.sequence_csv)
    success = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating JSON"):
        success += int(generate_json(row, Path(args.cif_dir), output_dir))

    print(f"✅ Generated {success}/{len(df)} JSON files in {output_dir}")


if __name__ == "__main__":
    main()
