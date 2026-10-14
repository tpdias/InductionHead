# Induction Heads on Markov Chains

*Work in progress.* A tiny transformer, written from raw PyTorch, trained on data from a Markov chain, with the goal of finding an induction head in the attention weights and explaining why it is there.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

- `src/data.py`: Markov chain generator
- `src/model.py`: transformer
- `src/train.py`: training loop
- `src/viz.py`: attention plots
- `notebooks/`: exploratory plots only
- `results/`: figures and checkpoints
