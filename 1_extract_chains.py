import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict

import pandas as pd
from Bio import PDB
from Bio.PDB import MMCIFIO, Model, Structure
from tqdm import tqdm

PROTEIN_LETTERS_3TO1: Dict[str, str] = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def get_sequence_from_chain(chain) -> str:
    sequence = ""
    for residue in chain:
        if residue.id[0] == " ":
            sequence += PROTEIN_LETTERS_3TO1.get(residue.get_resname().upper(), "X")
    return sequence


def process_single_pdb(task):
    input_pdb, output_dir_cif = task
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb)
        base_name = Path(input_pdb).stem

        chain_sequences = {}
        merged_sequence = ""

        for chain in structure[0]:
            chain_id = chain.id
            sequence = get_sequence_from_chain(chain)
            chain_sequences[chain_id] = sequence
            merged_sequence += sequence

            new_structure = Structure.Structure("new_structure")
            new_model = Model.Model(0)
            new_structure.add(new_model)
            new_model.add(chain.copy())

            cif_io = MMCIFIO()
            cif_io.set_structure(new_structure)
            cif_io.save(str(Path(output_dir_cif) / f"{base_name}_chain_{chain_id}.cif"))

        return base_name, chain_sequences, len(merged_sequence)
    except Exception as exc:
        print(f"Error processing {input_pdb}: {exc}")
        return None, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-chain CIF files and sequence CSV from PDB files.")
    parser.add_argument("--input_dir", default="./pdb", help="Directory containing input .pdb files.")
    parser.add_argument("--output_dir_cif", default="./complex_chain_cifs", help="Directory for per-chain CIF files.")
    parser.add_argument("--save_csv", default="./complex_chain_sequences.csv", help="Output CSV path for extracted sequences.")
    parser.add_argument("--num_workers", type=int, default=max(1, mp.cpu_count() - 2), help="Worker process count.")
    args = parser.parse_args()

    Path(args.output_dir_cif).mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(Path(args.input_dir).glob("*.pdb"))

    tasks = [(str(pdb_path), args.output_dir_cif) for pdb_path in pdb_files]
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_pdb, tasks), total=len(tasks), desc="Extracting chains"))

    sequences_dict = {}
    for base_name, chain_sequences, length in results:
        if base_name is not None:
            sequences_dict[base_name] = {"sequences": chain_sequences, "length": length}

    all_chain_ids = sorted({chain_id for e in sequences_dict.values() for chain_id in e["sequences"].keys()}, key=str)
    rows = []
    for complex_name, entry in sequences_dict.items():
        row = {"complex": complex_name, "total_length": entry["length"]}
        for chain_id in all_chain_ids:
            row[f"chain_{chain_id}_seq"] = entry["sequences"].get(chain_id, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["complex", "total_length"] + [c for c in df.columns if c not in ["complex", "total_length"]]
        df = df[cols]
    df.to_csv(args.save_csv, index=False)
    print(f"✅ Wrote {len(df)} rows to {args.save_csv}")


if __name__ == "__main__":
    main()
