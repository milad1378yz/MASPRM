<div align="center">
  <h1>MASPRM</h1>
  <p>Multi-Agent System Process Reward Model</p>
  <p>A lightweight process reward model that guides multi-agent reasoning at search time.</p>
  <p>
    <a href="">Paper</a> ·
    <a href="">PDF</a> ·
    <a href="">Project Page</a>
  </p>
</div>

<p align="center">
  <img
    src=""
    alt="MASPRM training pipeline"
    width="900"
  />
</p>
<p align="center"><em>MASPRM training pipeline (main paper figure).</em></p>

## Highlights
- MASPRM adds a process reward model to guide multi-agent sytem.
- Plugs into MCTS and inference time search for better trajectory selection.
- Improves exact-match on challenging reasoning benchmarks.

## Quickstart
```bash
pip install -r requirements.txt
python src/run_mcts.py --dataset mmlu --split train --load_in_4bit --ray --gpus_per_actor 0.125 --actors 32
```

## Docker
```bash
docker build -t masprm .
docker run --rm -it -v "$PWD:/app" masprm python src/run_mcts.py --help
```
