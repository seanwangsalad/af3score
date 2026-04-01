# AF3Score — CLAUDE.md

## Project overview

Minimal Python pipeline for evaluating protein structure quality using AlphaFold3 (AF3Score). Upstream source: `google-deepmind/alphafold3` (CC BY-NC-SA 4.0). The repo is pruned to the smallest usable workflow — no bash wrappers, all steps exposed as `argparse` CLIs.

## Environment

- Conda env: `af3score`, Python 3.12+
- Key deps: JAX, Haiku, biopython, h5py, pandas, RDKit
- Install: `pip install --no-deps -e .` then `build_data`

## Pipeline steps (in order)

| Script | Input | Output |
|---|---|---|
| `1_extract_chains.py` | `--input_dir` (PDB dir) | chain CIFs + sequence CSV |
| `2_pdb2jax.py` | `--pdb_dir` | H5 files |
| `3_generate_json.py` | sequence CSV + CIF dir | AF3 JSON configs |
| `run_af3score.py` | JSON + H5 + model weights | AF3 output per complex |
| `04_get_metrics.py` | PDB dir + AF3 output | metrics CSV |

Orchestrator: `af3score_pipeline.py` — runs all steps in sequence via `subprocess`.

## Output metrics

All metrics are written to a single CSV by `04_get_metrics.py`. Per-chain columns use the chain letter as suffix (e.g. `chain_A_plddt`). Per-pair columns use `{chain1}_{chain2}` (e.g. `ipsae_A_B`).

- **pTM** / **ipTM**: global and interface topology confidence
- **pLDDT**: per-chain average; computed from `atom_plddts` in `confidences.json`
- **PAE**: intra-chain and inter-chain predicted aligned error
- **ipSAE** (`ipsae_{c1}_{c2}`): directional interface score — implements `ipsae_d0res` from Dunbrack et al. 2025 (biorxiv). d0 is per-residue, based on the count of chain2 residues with PAE < 10 Å for that residue. Final score = max over chain1 residues. All ordered pairs reported (A→B ≠ B→A).
- **min_ipsae**: minimum `ipsae_*` value across all chain pairs for the complex
- **pDOCKQ** (`pdockq_{c1}_{c2}`): Bryant et al. 2022. Symmetric — only unordered pairs reported. Formula: `x = avg_if_pLDDT × log10(n_contacts)`, `pDOCKQ = 0.724/(1+exp(-0.052×(x−152.611)))+0.018`. Contacts = Cβ–Cβ (CA for Gly) within 8 Å. pLDDT taken from the CB atom directly (not per-residue average), matching the reference implementation.
- **min_pdockq**: minimum `pdockq_*` value across all chain pairs

## Key files

- `af3score_pipeline.py` — one-command orchestrator; use `--weights` to point at the model weights file (parent dir is forwarded as `--model_dir`)
- `run_af3score.py` — AF3 inference entry point (absl flags, not argparse)
- `model_manager_correct.py` — model loading utilities
- `ipsae_calculator.py` — `calculate_ipsae` and `calculate_pdockq`; both imported by `04_get_metrics.py`. Reference: `ipsae.py` (Dunbrack, Fox Chase Cancer Center, MIT license) — kept in repo for auditing.
- `04_get_metrics.py` — aggregates all metrics from AF3 JSON outputs in parallel

## Conventions

- No hardcoded absolute paths anywhere — all paths flow through CLI args
- `--run_data_pipeline=false` skips MSA/DB search (standard for scoring existing structures)
- `--weights` in `af3score_pipeline.py` takes a file path; the parent directory is passed as `--model_dir` to `run_af3score.py`
- All output subdirectories default under `--output_dir` but can be overridden individually
- `src/alphafold3/` contains upstream AF3 source — avoid modifying these files unless necessary
- `atom_plddts` / `atom_chain_ids` in `confidences.json` are per-atom arrays in the same order as ATOM/HETATM records in the input PDB — this assumption underlies both ipSAE token masking and pDOCKQ pLDDT lookup
