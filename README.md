# Test-Time Scaling for Small VLMs on Multilingual Exam QA

Code for the paper *"More Tokens, Fewer Trees: Test-Time Scaling for Small VLMs on Multilingual Exam QA"*

We study test-time scaling (TTS) strategies for open-weight vision-language models (VLMs) on the [EXAMS-V](https://huggingface.co/datasets/MBZUAI/EXAMS-V) benchmark under a single-GPU, 7B-parameter budget, as part of the [ImageCLEF 2026 Multimodal Reasoning](https://www.imageclef.org/2026/multimodal-reasoning) shared task.

## Key Results

| Method | Policy | Accuracy | Scale |
|---|---|---|---|
| Zero-shot | Qwen2.5-VL-7B | 56.8% | val-full |
| Chain-of-thought | Qwen2.5-VL-7B | 63.1% | val-full |
| Self-consistency N=8 | Qwen2.5-VL-7B | 66.4% | val-full |
| Self-consistency N=8, 1k-tok | Qwen3.5-4B | 77.8% | val-full |
| **Self-consistency N=8, 2k-tok** | **Qwen3.5-4B** | **81.5%** | **val-full** |
| Self-consistency N=16, 2k-tok | Qwen3.5-4B | 81.6% | val-full |
| Best (N=16, 2k + guided repair) | Qwen3.5-4B | **84.1%** | test |

Three main findings:
1. The dominant scaling axis is per-chain token budget, not chain count: doubling the budget recovers +3.7 pp by eliminating truncation; doubling chains adds only +0.15 pp.
2. PRM-guided beam search underperforms flat self-consistency (-0.39 pp), with search collapsing to unanimous beams on 72% of questions.
3. Neither a generative critic nor a trained PRM beats majority vote, replicated across two policies.

## Repository Structure

```
tts-small-vlm/
├── src/                    # Core pipeline modules
│   ├── backend_vllm.py     # vLLM inference backend
│   ├── backend.py          # HF Transformers backend (PRM model loading)
│   ├── describe.py         # Stage 1: N-sample image description
│   ├── reason.py           # Stage 2: M-sample text-only reasoning
│   ├── verify.py           # Stage 3: majority vote, generative critic, PRM
│   ├── search.py           # PRM-BAS beam-annealing search
│   ├── pipeline.py         # Pipeline orchestrator
│   └── utils/              # Data classes, answer extraction, logging, subsets
├── scripts/
│   ├── experiment.py       # Main experiment runner
│   ├── analyze.py          # Post-hoc analysis (stratified metrics, scaling curves)
│   ├── score_with_critic.py    # Generative critic rescoring (RQ4)
│   ├── score_with_prm.py      # Qwen-VL-PRM rescoring (RQ4)
│   ├── rescore_dtr_with_prm.py # DTR + PRM rescoring (Appendix)
│   ├── repair_parse_failures.py # Guided parse repair
│   ├── format_submission.py    # ImageCLEF competition submission formatting
│   ├── analyze_parse_fail.py   # Parse failure diagnostics
│   └── serve_vllm.sh          # vLLM server launcher
├── configs/                # YAML configs, one per experiment
│   ├── baselines/          # Zero-shot, CoT, SC (Q2.5 and Q3.5)
│   ├── scaling/            # Token budget, chain count, temperature sweep
│   ├── search/             # DTR, PRM-BAS, DTR+PRM
│   └── test/               # Test-set submission config
├── slurm/                  # SLURM job scripts (Snellius HPC)
├── eval/                   # Stratified evaluation script
├── paper/                  # Figure generation scripts
│   └── figures/            # plot_figures.py, generated PDFs
├── tests/                  # Unit tests (49 tests, fully mocked)
├── notebook.ipynb          # Pipeline walkthrough with search and verification
├── requirements.txt
├── .env.example            # Environment variable template
└── LICENSE
```

## Setup

**Requirements**: Python 3.12+, CUDA 12.x, a GPU with at least 40 GB VRAM (A40/A100).

```bash
# Clone and install
git clone <repo-url> && cd tts-small-vlm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure environment (optional, for W&B tracking)
cp .env.example .env
# Edit .env with your API keys

# Run unit tests (no GPU required)
pytest tests/ -v
```

**Models** (downloaded automatically on first use):
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (baselines, search experiments)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (scaling, best configuration)
- [Qwen-VL-PRM-7B](https://huggingface.co/ob11/Qwen-VL-PRM-7B) (PRM-BAS search, PRM rescoring)

## Reproducing Paper Results

Each experiment is driven by a YAML config and a SLURM job script. All scripts assume execution from the repository root.

### Running an experiment locally

```bash
# Start vLLM server (in a separate terminal)
bash scripts/serve_vllm.sh

# Run a single experiment
python scripts/experiment.py --config configs/scaling/q35_sc_n8_2k.yaml --verbose
```

### Running on a SLURM cluster

Edit the `--account` field in the SLURM scripts, then:

```bash
sbatch slurm/baselines_q25.sbatch      # Table 1, Q2.5 rows
sbatch slurm/baselines_q35.sbatch      # Table 1, Q3.5 rows
sbatch slurm/scaling_q35.sbatch        # Table 2, scaling rows
sbatch slurm/temp_sweep_q35.sbatch     # Table 2, temperature sweep
sbatch slurm/search_q25_dev.sbatch     # Table 3, search comparison
sbatch slurm/prm_bas_q35_val.sbatch    # Section 5.2, PRM-BAS at val scale
sbatch slurm/selectors_q25.sbatch      # Table 4, Q2.5 selectors
sbatch slurm/selectors_q35.sbatch      # Table 4, Q3.5 selectors
sbatch slurm/dtr_prm_rescore_dev.sbatch # Appendix Table 10
sbatch slurm/test_submission.sbatch     # Best config + guided repair on test
```

### Paper result to config mapping

| Paper Table | Result | Config |
|---|---|---|
| Tab 1 | Q2.5 Zero-shot (56.8%) | `baselines/q25_zero_shot.yaml` |
| Tab 1 | Q2.5 CoT (63.1%) | `baselines/q25_cot.yaml` |
| Tab 1 | Q2.5 SC-N=8 (66.4%) | `baselines/q25_sc_n8.yaml` |
| Tab 1 | Q3.5 Zero-shot (57.1%) | `baselines/q35_zero_shot.yaml` |
| Tab 1 | Q3.5 CoT (69.7%) | `baselines/q35_cot.yaml` |
| Tab 1 | Q3.5 SC-N=8 1k (77.8%) | `scaling/q35_sc_n8_1k.yaml` |
| Tab 1 | Q3.5 SC-N=8 2k (81.5%) | `scaling/q35_sc_n8_2k.yaml` |
| Tab 1 | Q3.5 SC-N=16 2k (81.6%) | `scaling/q35_sc_n16_2k.yaml` |
| Tab 2 | T=0.3 (80.8%) | `scaling/q35_sc_n8_2k_t03.yaml` |
| Tab 2 | T=0.5 (81.0%) | `scaling/q35_sc_n8_2k_t05.yaml` |
| Tab 2 | T=0.7 (81.5%) | `scaling/q35_sc_n8_2k.yaml` |
| Tab 2 | T=0.9 (81.0%) | `scaling/q35_sc_n8_2k_t09.yaml` |
| Tab 3 | SC-N=8 dev (65.5%) | `search/q25_sc_n8_dev.yaml` |
| Tab 3 | DTR N=2,M=2 (55.0%) | `search/q25_dtr_n2m2.yaml` |
| Tab 3 | DTR N=4,M=4 (61.5%) | `search/q25_dtr_n4m4.yaml` |
| Tab 3 | PRM-BAS (60.5%) | `search/q25_prm_bas_dev.yaml` |
| Tab 4 | R1/R2 selectors | Pool from `baselines/q25_sc_n8.yaml` or `scaling/q35_sc_n8_2k.yaml`, rescored by `score_with_critic.py` / `score_with_prm.py` |
| App Tab 10 | DTR + PRM modes | Pool from `search/q25_dtr_n4m4.yaml`, rescored by `rescore_dtr_with_prm.py` |
| S-best | Test 84.1% | `test/q35_sc_n16_2k_test.yaml` + `repair_parse_failures.py` |

### Post-hoc analysis and figures

```bash
# Stratified analysis of a completed run
python scripts/analyze.py --run-dir runs/<run_id>

# Generate paper figures (hardcoded data, no GPU needed)
python paper/figures/plot_figures.py
```

## Notebook

`notebook.ipynb` walks through the full pipeline:
1. Baselines (zero-shot, chain-of-thought, self-consistency)
2. Search strategy (PRM-BAS beam-annealing search within a describe-then-reason scaffold)
3. Verification strategies (training-free generative critic, Qwen-VL-PRM discriminative rescoring)
4. Guided parse repair
5. Stratified evaluation

GPU-dependent cells are guarded with a `RUN_INFERENCE` flag for readability without hardware.

## Data

The [EXAMS-V dataset](https://huggingface.co/datasets/MBZUAI/EXAMS-V) is loaded automatically from HuggingFace. It contains approximately 25,000 real school exam questions across 13 languages and 20 subjects. Our experiments use the validation split (4,651 questions) and a 200-question stratified subset for ablations.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
