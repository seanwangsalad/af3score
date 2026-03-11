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
python 1_extract_chains.py \
  --input_dir ./pdb \
  --output_dir_cif ./complex_chain_cifs \
  --save_csv ./complex_chain_sequences.csv
```

### Step 2: convert PDB to JAX/H5 inputs
```bash
python 2_pdb2jax.py \
  --pdb_dir ./pdb \
  --output_dir ./complex_h5
```
Input:
- `./pdb/*.pdb`

### Step 3: generate AF3 JSON configs
```bash
python 3_generate_json.py \
  --sequence_csv ./complex_chain_sequences.csv \
  --cif_dir ./complex_chain_cifs \
  --output_dir ./complex_json_files
```

### Step 4: run AF3Score inference
```bash
python run_af3score.py \
  --model_dir=/absolute/path/to/alphafold3_model_parameters \
  --batch_json_dir=./complex_json_files \
  --batch_h5_dir=./complex_h5 \
  --output_dir=./af3score_outputs \
  --run_data_pipeline=False \
  --run_inference=true
```
Input:
- `./pdb/*.pdb`

## One-command Python orchestration

> Note: the pipeline wrapper flag is `--weights` (it forwards to `run_af3score.py --model_dir`).


```bash
python af3score_pipeline.py \
  --input ./pdb \
  --output_dir ./run_001 \
  --weights /absolute/path/to/alphafold3_model_parameters
```

### Full path control (no hardcoded internal paths)

All generated output locations and called script paths are configurable from CLI:

```bash
python af3score_pipeline.py \
  --input /data/my_pdbs \
  --output_dir /runs/exp_001 \
  --cif_dir /runs/exp_001/custom_cifs \
  --sequence_csv /runs/exp_001/custom_sequences.csv \
  --h5_dir /runs/exp_001/custom_h5 \
  --json_dir /runs/exp_001/custom_json \
  --af3_output_dir /runs/exp_001/custom_af3 \
  --metric_csv /runs/exp_001/custom_metrics.csv \
  --extract_script /opt/pipeline/1_extract_chains.py \
  --pdb2jax_script /opt/pipeline/2_pdb2jax.py \
  --json_script /opt/pipeline/3_generate_json.py \
  --af3score_script /opt/pipeline/run_af3score.py \
  --metrics_script /opt/pipeline/04_get_metrics.py \
  --weights /models/af3
```
Input:
- `./complex_chain_sequences.csv`

### Do I need `--db_dir`?

No, it is optional. If omitted, `run_af3score.py` uses its internal default DB search path.

Provide `--db_dir` only when you want to override where AF3 looks for databases:

```bash
--db_dir /path/to/databases
```

You can still disable MSA/database search entirely with:

```bash
--run_data_pipeline=false
```

## Output Metrics

- **pTM**: global/per-chain topology confidence.
- **ipTM**: interface confidence between chains.
- **pLDDT**: per-residue confidence.
- **PAE**: expected aligned error.
- **ipSAE**: interface-focused score from aligned errors.

## Reference

For AlphaFold3 details, see the official repository:
https://github.com/google-deepmind/alphafold3
