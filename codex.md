# AF3Score Codebase Notes

This document is a maintainer-oriented map of the repository. It is meant to help a new engineer understand:

- what each top-level script does
- how files move through the pipeline
- what the important output formats look like
- where the expensive and failure-prone logic lives
- how resume works in `run_af3score.py`


## Repository Purpose

This repository wraps AlphaFold 3 inference so it can score an input set of protein complex PDBs. The pipeline takes each input complex, builds AlphaFold 3 JSON input plus a traced coordinate tensor, runs the AF3 confidence head / inference path, and then extracts downstream metrics such as pTM, ipTM, pLDDT, PAE, ipSAE, and pDOCKQ.

At a high level:

1. Read input `.pdb` complexes.
2. Split them into per-chain CIF templates and collect per-chain sequences.
3. Convert each original complex PDB into an `.h5` tensor file used as the initial structure input.
4. Generate one AlphaFold 3 JSON file per complex.
5. Run `run_af3score.py` in batch mode over matching JSON/H5 pairs.
6. Extract metrics from the AF3 outputs.


## Top-Level Entry Points

### `af3score_pipeline.py`

This is the Python orchestration wrapper. It sequentially runs:

1. `1_extract_chains.py`
2. `2_pdb2jax.py`
3. `3_generate_json.py`
4. `run_af3score.py`
5. `04_get_metrics.py`

Important properties:

- It is a convenience wrapper, not the core inference implementation.
- It always reruns steps 1 to 3 unless the user manually skips them by calling scripts directly.
- It forwards `--resume` to `run_af3score.py`, so only the AF3 inference stage is resumable.


### `run_af3score.py`

This is the critical runtime script. It contains the expensive AlphaFold 3 inference path and is the place where resume matters.

It supports:

- single-input mode via `--json_path` or `--input_dir`
- batch mode via `--batch_json_dir` and `--batch_h5_dir`

In this repository, batch mode is the normal scoring path.


## Pipeline Stages

### 1. `1_extract_chains.py`

Purpose:

- parses each input PDB
- extracts each chain into its own single-chain CIF
- collects one-letter amino-acid sequences
- writes a CSV describing each complex

Key naming rule:

- the stable complex identifier is `Path(input_pdb).stem`
- this basename becomes the `complex` column in the CSV
- that same name later becomes the JSON filename, H5 filename, output directory name, and metrics row key

Outputs:

- `complex_chain_cifs/<complex>_chain_<chain_id>.cif`
- `complex_chain_sequences.csv`

Important implementation detail:

- chain IDs come directly from the parsed PDB structure
- merged complex length is just the concatenated protein-chain sequence length


### 2. `2_pdb2jax.py`

Purpose:

- converts each full input complex PDB into a padded coordinate tensor
- saves the tensor and metadata into an HDF5 file

Core logic:

- parse the full PDB
- convert residues into a fixed atom representation
- pad to a bucket size
- wrap as a JAX array
- save as `.h5`

Important implementation detail:

- bucket size is inferred from the input directory name suffix if possible, otherwise defaults to `3072`
- the script currently runs with `num_copies=1`
- output filename is `<complex>.h5`, matching the original PDB stem

Outputs:

- `complex_h5/<complex>.h5`

HDF5 layout:

- dataset `coordinates`: padded coordinate array
- dataset `seq_length`: original sequence length before padding
- dataset `shape`: saved array shape
- group `metadata` attributes:
  - `pdb_file`
  - `chain_ids`
  - `num_copies`
  - `original_length`
  - `padded_length`


### 3. `3_generate_json.py`

Purpose:

- turns the sequence CSV plus per-chain CIF files into AlphaFold 3 JSON inputs

Important implementation detail:

- it reads the `complex` column from the CSV and uses that as the stable target name
- it sets `"name": complex_name`
- it currently hardcodes `"modelSeeds": [10]`
- each chain is represented as a protein with a single template pointing at its chain CIF

Outputs:

- `complex_json_files/<complex>.json`

JSON shape:

```json
{
  "dialect": "alphafold3",
  "version": 1,
  "name": "<complex>",
  "sequences": [
    {
      "protein": {
        "id": "A",
        "sequence": "...",
        "modifications": [],
        "unpairedMsa": ">query\n...\n",
        "pairedMsa": ">query\n...\n",
        "templates": [
          {
            "mmcifPath": "/path/to/<complex>_chain_A.cif",
            "queryIndices": [...],
            "templateIndices": [...]
          }
        ]
      }
    }
  ],
  "modelSeeds": [10],
  "bondedAtomPairs": null,
  "userCCD": null
}
```


### 4. `run_af3score.py`

Purpose:

- loads AF3 JSON inputs
- optionally runs the AF3 data pipeline
- runs the model / confidence head
- writes selected AF3 outputs

This is the longest-running step and the main interruption target.

Important internal blocks:

- flag definitions:
  - input/output paths
  - batch mode paths
  - write controls
  - `--resume`
- `predict_structure(...)`
  - featurises one fold input
  - runs model inference
  - extracts structures
- `write_selective_output(...)`
  - writes only the chosen output files for each sample
- `write_outputs(...)`
  - creates `seed-*_sample-*` directories
  - writes outputs for each sample
  - writes the completion marker
- batch-mode loop in `main(...)`
  - matches `<complex>.json` with `<complex>.h5`
  - reuses one model runner across all complexes
  - performs the batch-level resume skip


### 5. `04_get_metrics.py`

Purpose:

- reads AF3 outputs
- computes downstream scoring metrics
- writes one summary CSV row per complex

Important implementation detail:

- it currently reads only `seed-10_sample-0` for each complex
- this matches the current JSON generator, which uses `modelSeeds: [10]`
- if seed logic changes in the future, this script may need updating

Outputs:

- `af3score_metrics.csv`
- optionally `failed_records.txt` under the AF3 output directory when extraction failures occur


## Naming Conventions and Data Flow

The key stable identifier is the original input PDB basename:

- input PDB: `<complex>.pdb`
- chain CIFs: `<complex>_chain_<chain>.cif`
- traced tensor: `<complex>.h5`
- AF3 JSON: `<complex>.json`
- AF3 output directory: `<complex>/`
- metrics row key: `description = <complex>`

This matters because batch-mode inference relies on matching JSON/H5 pairs by basename.


## Important Output Layouts

### Input directory

```text
input_pdb_dir/
  complex1.pdb
  complex2.pdb
```


### Intermediate outputs

```text
run_dir/
  complex_chain_cifs/
    complex1_chain_A.cif
    complex1_chain_B.cif
  complex_chain_sequences.csv
  complex_h5/
    complex1.h5
  complex_json_files/
    complex1.json
```


### AF3 output directory

For one complex:

```text
af3score_outputs/
  complex1/
    .af3score_complete
    seed-10_sample-0/
      model.cif
      summary_confidences.json
      confidences.json
```

If `num_samples > 1`, more `seed-10_sample-N/` directories appear.

If more seeds are added later, directories become:

```text
seed-<seed>_sample-<sample_idx>/
```


### `summary_confidences.json`

This is the lightweight summary produced by AF3 post-processing. It is expected to contain fields used by metrics extraction such as:

- `ptm`
- `iptm`
- `chain_ptm`
- `chain_iptm`
- `chain_pair_iptm`


### `confidences.json`

This is the full confidence output. It is expected to contain fields used by metrics extraction such as:

- `pae`
- `atom_plddts`
- `atom_chain_ids`


### Metrics CSV

`04_get_metrics.py` writes one row per complex with columns such as:

- `description`
- `input_pdb_path`
- `ptm`
- `iptm`
- `chain_<chain>_plddt`
- `chain_<chain>_pae`
- `chain_<chain>_ptm`
- `chain_<chain>_iptm`
- `ipsae_<chainpair>`
- `min_ipsae`
- `pdockq_<chainpair>`
- `min_pdockq`
- `iptm_<chain1>_<chain2>`


## Resume Behavior

Resume is implemented in `run_af3score.py` and only applies to the AF3 inference stage.

What resume does:

- checks batch-mode outputs before processing each packed structure
- uses the complex basename from the JSON filename as the key
- looks in `output_dir/<complex>/`
- skips the complex if either:
  - `.af3score_complete` exists, or
  - enough complete `seed-*_sample-*` directories already exist

What resume does not do:

- it does not skip `1_extract_chains.py`
- it does not skip `2_pdb2jax.py`
- it does not skip `3_generate_json.py`
- it does not skip metrics extraction

Startup summary:

- in batch mode, the script now prints:
  - total packed structures found
  - remaining packed structures after applying resume logic


## Important Constraints and Assumptions

### Seed assumption

The current JSON generator uses:

```json
"modelSeeds": [10]
```

Because of that, downstream code often implicitly assumes outputs under:

```text
seed-10_sample-0
```

If you change `modelSeeds`, check:

- `run_af3score.py` output layout assumptions
- `04_get_metrics.py` hardcoded `seed-10_sample-0`


### Batch-mode pairing assumption

Batch mode assumes:

- every JSON file is named `<complex>.json`
- every traced tensor file is named `<complex>.h5`
- the correct pair is matched by shared basename

If one side is missing, the complex is skipped with a warning.


### Protein-only assumptions in preprocessing

`1_extract_chains.py` and `2_pdb2jax.py` are primarily written for protein chains and CA/backbone logic. Non-protein or heavily modified inputs may need extra handling.


## Files Worth Reading First

If a new engineer needs to get productive quickly, read these in order:

1. `README.md`
2. `af3score_pipeline.py`
3. `1_extract_chains.py`
4. `2_pdb2jax.py`
5. `3_generate_json.py`
6. `run_af3score.py`
7. `04_get_metrics.py`


## Practical Debugging Guide

### If a complex is missing from AF3 outputs

Check:

- does `complex_json_files/<complex>.json` exist?
- does `complex_h5/<complex>.h5` exist?
- did `run_af3score.py` print a missing-H5 warning?


### If resume is not skipping a completed complex

Check:

- does `af3score_outputs/<complex>/` exist?
- does it contain `.af3score_complete`?
- does it contain enough complete `seed-*_sample-*` directories?
- is `--resume=true` actually being passed?


### If metrics extraction fails

Check:

- does `af3score_outputs/<complex>/seed-10_sample-0/` exist?
- are `model.cif`, `summary_confidences.json`, and `confidences.json` present?
- does the input PDB still exist as `input_pdb_dir/<complex>.pdb`?


## Container Notes

The Apptainer definition is in:

- `apptainer/af3score_sherlock.def`

Important operational notes:

- build from the repository root so `%files . /opt/af3score` copies the entire repo
- AF3 model weights are expected at runtime, not baked into the image
- large builds may require setting `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR`


## Short Summary

If you remember only three things:

1. The entire pipeline is keyed by the original PDB basename.
2. `run_af3score.py` is the expensive step and the only stage with resume support.
3. `04_get_metrics.py` currently assumes outputs live under `seed-10_sample-0`.
