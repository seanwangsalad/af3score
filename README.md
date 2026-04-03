# AF3Score Pipeline

A Python-first pipeline for evaluating protein structure quality with AF3Score.

## Prerequisites

- Linux x86_64
- CUDA 12 compatible GPU

## Installation

Choose either **conda** or **uv** below.

---

### Option A — conda

#### 1. Create a conda environment with Python 3.11

```bash
conda create -n af3score python=3.11
```

#### 2. Activate the environment

```bash
conda activate af3score
```

#### 3. Install the C++ compiler toolchain (required to build the C extensions)

```bash
conda install gxx_linux-64 gxx_impl_linux-64 gcc_linux-64 gcc_impl_linux-64=13.2.0
```

#### 4. Clone this repository

```bash
git clone https://github.com/Mingchenchen/AF3Score.git
cd AF3Score
```

#### 5. Install pinned Python dependencies (JAX, Haiku, Triton, etc.)

```bash
pip install -r dev-requirements.txt
```

#### 6. Install AF3Score itself (editable, without re-resolving dependencies)

```bash
pip install --no-deps -e .
```

#### 7. Build the compiled data assets (chemical components database, etc.)

```bash
build_data
```

> This step writes pre-processed data files used at inference time. It must complete before running the pipeline.

#### 8. Install additional Python dependencies for the scoring pipeline

```bash
conda install -c conda-forge biopython h5py pandas tqdm
```

---

### Option B — uv

#### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Install the system C++ compiler toolchain (required to build the C extensions)

```bash
sudo apt-get install -y gcc g++
```

#### 3. Clone this repository

```bash
git clone https://github.com/Mingchenchen/AF3Score.git
cd AF3Score
```

#### 4. Create a virtual environment pinned to Python 3.11

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

#### 5. Install pinned Python dependencies (JAX, Haiku, Triton, etc.)

```bash
uv pip install -r dev-requirements.txt
```

#### 6. Install AF3Score itself (editable, without re-resolving dependencies)

```bash
uv pip install --no-deps -e .
```

#### 7. Build the compiled data assets (chemical components database, etc.)

```bash
build_data
```

> This step writes pre-processed data files used at inference time. It must complete before running the pipeline.

#### 8. Install additional Python dependencies for the scoring pipeline

```bash
uv pip install biopython h5py pandas tqdm
```

## Python CLI workflow (no bash wrappers)

The pipeline scripts live inside the cloned repo directory. You can run them from any working directory by pointing Python at their absolute paths. Set a shell variable once to avoid repetition:

```bash
export AF3SCORE=/path/to/AF3Score   # e.g. /home/user/af3score_old
```

The pipeline is split into explicit Python steps with `argparse` CLIs.

### Step 1: extract chains + sequences
```bash
python $AF3SCORE/1_extract_chains.py \
  --input_dir ./pdb \
  --output_dir_cif ./complex_chain_cifs \
  --save_csv ./complex_chain_sequences.csv
```

### Step 2: convert PDB to JAX/H5 inputs
```bash
python $AF3SCORE/2_pdb2jax.py \
  --pdb_dir ./pdb \
  --output_dir ./complex_h5
```

### Step 3: generate AF3 JSON configs
```bash
python $AF3SCORE/3_generate_json.py \
  --sequence_csv ./complex_chain_sequences.csv \
  --cif_dir ./complex_chain_cifs \
  --output_dir ./complex_json_files
```

### Step 4: run AF3Score inference
```bash
python $AF3SCORE/run_af3score.py \
  --model_dir=/absolute/path/to/alphafold3_model_parameters \
  --batch_json_dir=./complex_json_files \
  --batch_h5_dir=./complex_h5 \
  --output_dir=./af3score_outputs \
  --run_data_pipeline=False \
  --run_inference=true
```

### Step 5: extract metrics (pLDDT, PAE, ipSAE, pDOCKQ, ipTM, pTM)
```bash
python $AF3SCORE/04_get_metrics.py \
  --input_pdb_dir ./pdb \
  --af3score_output_dir ./af3score_outputs \
  --save_metric_csv ./af3score_metrics.csv \
  --num_workers 16
```

## One-command Python orchestration

> Note: the pipeline wrapper flag is `--weights` and expects a single weight file path; it forwards that file's parent directory to `run_af3score.py --model_dir`.

```bash
python $AF3SCORE/af3score_pipeline.py \
  --input ./pdb \
  --output_dir ./run_001 \
  --weights /absolute/path/to/alphafold3_model_parameters/weights.bin
```

### Full path control (no hardcoded internal paths)

All generated output locations and called script paths are configurable from CLI:

```bash
python $AF3SCORE/af3score_pipeline.py \
  --input /data/my_pdbs \
  --output_dir /runs/exp_001 \
  --cif_dir /runs/exp_001/custom_cifs \
  --sequence_csv /runs/exp_001/custom_sequences.csv \
  --h5_dir /runs/exp_001/custom_h5 \
  --json_dir /runs/exp_001/custom_json \
  --af3_output_dir /runs/exp_001/custom_af3 \
  --metric_csv /runs/exp_001/custom_metrics.csv \
  --extract_script $AF3SCORE/1_extract_chains.py \
  --pdb2jax_script $AF3SCORE/2_pdb2jax.py \
  --json_script $AF3SCORE/3_generate_json.py \
  --af3score_script $AF3SCORE/run_af3score.py \
  --metrics_script $AF3SCORE/04_get_metrics.py \
  --weights /models/af3/weights.bin
```

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
- **pLDDT**: per-residue local confidence.
- **PAE**: predicted aligned error between residue pairs.
- **ipSAE**: interface-focused score derived from aligned errors (per chain pair, plus `min_ipsae`).
- **pDOCKQ**: interface quality score based on Cβ contacts and interface pLDDT (Bryant et al. 2022; per chain pair, plus `min_pdockq`).

## Reference

For AlphaFold3 details, see the official repository:
https://github.com/google-deepmind/alphafold3
