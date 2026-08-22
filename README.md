# MASPRM

MASPRM is a process reward model for guiding search over routed transcripts in multi agent systems. It supports Monte Carlo Tree Search, stepwise beam search, outcome reward models, policy likelihood baselines, dynamic scheduling, and dynamic routing.

- [Paper](https://arxiv.org/abs/2510.24803)
- [Project page](https://milad1378yz.github.io/MASPRM)
- License: MIT

## Repository Layout

- `configs/` contains the agent prompts, communication graphs, and schedules.
- `src/run_mcts.py` generates MAS MCTS trees for training.
- `src/train/preprocess_data.py` extracts process samples and preference pairs.
- `src/train/trainer.py` trains MASPRM and ORM scorers.
- `src/experiments/compare_sbs_mcts.py` runs the main search and scorer comparisons.
- `aggregator.py` aggregates metrics across seed specific log files.

## Installation

The pinned environment uses Python 3.11 and CUDA capable PyTorch. The full experiments require NVIDIA GPUs with enough memory for the selected generator and scorer models.

```bash
conda create -n masprm python=3.11 -y
conda activate masprm
pip install -r requirements.txt
```

The Docker image provides the same pinned Python dependencies.

```bash
docker build -t masprm .
docker run --rm -it --gpus all -v "$PWD:/app" masprm
```

## Smoke Test

This command generates two small GSM8K trees. Output is written below `results/` using a deterministic directory name that records the search settings.

```bash
python src/run_mcts.py \
  --dataset gsm8k \
  --split train \
  --n 2 \
  --seed 42 \
  --n_rollouts 2 \
  --search_c 4 \
  --n_candidates 3 \
  --mas_config configs/gsm8k.yaml \
  --load_in_4bit \
  --no_compile
```

## Training Data Generation

The paper uses MAS MCTS with 40 simulations, three candidates per expansion, an exploration constant of 4, temperature 0.7, and top p 0.95. The following command generates the GSM8K training trees. Remove `--ray` and its worker options for one process per run.

```bash
python src/run_mcts.py \
  --dataset gsm8k \
  --split train \
  --seed 42 \
  --n_rollouts 40 \
  --search_c 4 \
  --n_candidates 3 \
  --temperature 0.7 \
  --top_p 0.95 \
  --mas_config configs/gsm8k.yaml \
  --load_in_4bit \
  --ray \
  --gpus_per_actor 1 \
  --actors 8
```

The command writes to `results/gsm8k_res_train_Qwen_Qwen2.5-1.5B-Instruct_r40_c4_cand3_t0p7_p0p95`. Extract the Bradley Terry preference pairs with:

```bash
python src/train/preprocess_data.py \
  --input results/gsm8k_res_train_Qwen_Qwen2.5-1.5B-Instruct_r40_c4_cand3_t0p7_p0p95 \
  --target-nodes 5 \
  --make-ppm \
  --ppm-max-pairs 8 \
  --ppm-pos-topk 4 \
  --ppm-neg-topk 4 \
  --view full \
  --config configs/gsm8k.yaml
```

## Scorer Training

The output preference file is next to the tree directory. This command uses the main 1.5B scorer settings from the paper.

```bash
python src/train/trainer.py \
  --mode ppm \
  --data_dir results/ppm_pairs_gsm8k_res_train_Qwen_Qwen2.5-1.5B-Instruct_r40_c4_cand3_t0p7_p0p95.jsonl \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir checkpoints/MASPRM-GSM8K-1.5B \
  --epochs 5 \
  --lr 1e-5 \
  --train-batch-size 8 \
  --grad-accum 16 \
  --lora-r 256 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --seed 42
```

Use `--mode orm` with terminal outcome samples to train an ORM. Use `Qwen/Qwen2.5-7B-Instruct` as the model identifier to train the 7B scorer.

## Evaluation

Run each evaluation with seeds 14, 24, and 42. The command below evaluates the main GSM8K Single Pass, SBS, and MCTS conditions on the full test split.

```bash
python src/experiments/compare_sbs_mcts.py \
  --dataset gsm8k \
  --split test \
  --seed 14 \
  --prm_dir checkpoints/MASPRM-GSM8K-1.5B \
  --prm_base_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --gen_model_id Qwen/Qwen2.5-1.5B-Instruct \
  --mas_config configs/gsm8k.yaml \
  --methods single_pass sbs_logprob sbs_prm mcts_logprob mcts_prm \
  --b1 3 \
  --b2 5 \
  --n_simulations 10 \
  --max_children 3 \
  --c_uct 2 \
  --sbs_temperature 0.7 \
  --mcts_temperature 0.7 \
  --top_p 0.95
```

Repeat the command for seeds 24 and 42, then aggregate the text logs.

```bash
python aggregator.py
```

## Data and Checkpoints

The benchmark datasets are downloaded through the Hugging Face `datasets` package and are not redistributed here. Generated search trees, raw evaluation logs, and model adapters exceed normal GitHub size limits. A versioned archive for these files is still required before the artifact can reproduce every numerical table from the paper. The source repository alone reproduces the pipeline but does not include all raw outputs or trained adapters.

## Citation

```bibtex
@article{yazdani2025masprm,
  title={{MASPRM}: Multi-Agent System Process Reward Model},
  author={Yazdani, Milad and Mostajabdaveh, Mahdi and Zhou, Zirui and Xiong, Ying},
  journal={arXiv preprint arXiv:2510.24803},
  year={2025}
}
```
