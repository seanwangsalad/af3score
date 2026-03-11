# AF3Score Pipeline

A Python-first pipeline for evaluating protein structure quality with AF3Score.

## Environment Setup

### 1. Create and activate a Conda environment
```bash
conda create -n af3score python=3.11
conda activate af3score
conda install gxx_linux-64 gxx_impl_linux-64 gcc_linux-64 gcc_impl_linux-64=13.2.0
```

### 2. Install AF3Score and dependencies
```bash
git clone https://github.com/Mingchenchen/AF3Score.git
cd AF3Score
pip install -r dev-requirements.txt
pip install --no-deps -e .
build_data
conda install -c conda-forge biopython h5py pandas
```

## Python CLI workflow (no bash wrappers)

The pipeline is split into explicit Python steps with `argparse` CLIs.

### Step 1: extract chains + sequences
```bash
python 1_extract_chains.py
```
Input:
- `./pdb/*.pdb`

Output:
- `./complex_chain_cifs/*.cif`
- `./complex_chain_sequences.csv`

### Step 2: convert PDB to JAX/H5 inputs
```bash
python 2_pdb2jax.py
```
Input:
- `./pdb/*.pdb`

Output:
- `./complex_h5/*.h5`

### Step 3: generate AF3 JSON configs
```bash
python 3_generate_json.py
```
Input:
- `./complex_chain_sequences.csv`

Output:
- `./complex_json_files/*.json`

### Step 4: run AF3Score inference
```bash
python run_af3score.py \
  --model_dir=/path/to/alphafold3_model_parameters \
  --db_dir=/path/to/databases \
  --batch_json_dir=./complex_json_files \
  --batch_h5_dir=./complex_h5 \
  --output_dir=./af3score_outputs \
  --run_data_pipeline=False \
  --run_inference=true
```

### Optional: run all steps in one Python command
```bash
python af3score_pipeline.py \
  --input_pdb_dir ./pdb \
  --output_dir ./run_001 \
  --model_dir /path/to/alphafold3_model_parameters \
  --db_dir /path/to/databases
```

### Optional: run multiple datasets
```bash
python af3score_multidir.py \
  --input_dirs ./set_a ./set_b \
  --output_parent_dir ./multi_runs \
  --model_dir /path/to/alphafold3_model_parameters \
  --db_dir /path/to/databases
```

In that mode, `--db_dir` is not required.

- **pTM**: global/per-chain topology confidence.
- **ipTM**: interface confidence between chains.
- **pLDDT**: per-residue confidence.
- **PAE**: expected aligned error.
- **ipSAE**: interface-focused score from aligned errors.

## Reference

For AlphaFold3 details, see the official repository:
https://github.com/google-deepmind/alphafold3
