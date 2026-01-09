<div align="center">
  <h1>MASPRM</h1>
  <p>Multi-Agent System Process Reward Model</p>
  <p>A lightweight process reward model that guides multi-agent reasoning at search time.</p>
  <p>
    <a href="https://arxiv.org/abs/2510.24803">Paper</a> ·
    <a href="https://arxiv.org/pdf/2510.24803">PDF</a> ·
    <a href="https://milad1378yz.github.io/MASPRM">Project Page</a>
  </p>
</div>

<p align="center">
  <img
    src="https://arxiv.org/html/2510.24803v1/figs/PRM_MAS-MCTS2data.drawio.png"
    alt="MASPRM training pipeline"
    width="900"
  />
</p>
<p align="center"><em>MASPRM training pipeline (main paper figure).</em></p>

## Highlights
- MASPRM adds a process reward model to guide multi-agent reasoning.
- Plugs into MCTS and search-time decoding for better trajectory selection.
- Improves exact-match on challenging math reasoning benchmarks.

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

## BibTeX
```bibtex
@article{yazdani2025masprm,
  title={{MASPRM}: Multi-Agent System Process Reward Model},
  author={Yazdani, Milad and Mostajabdaveh, Mahdi and Zhou, Zirui and Xiong, Ying},
  journal={arXiv preprint arXiv:2510.24803},
  year={2025}
}
```
