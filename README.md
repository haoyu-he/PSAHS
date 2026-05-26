# PSAHS

Official implementation of **Progressive Structure Adjustment for Homophily Shift (PSAHS)** for Graph Domain Adaptation (GDA).

PSAHS addresses cross-domain **homophily shift** by:

1. **Source adjustment** — reweight edges and add intra-class links so low-homophily source nodes reach a target homophily level \(h\).
2. **Target adjustment (progressive)** — refine the target graph using pseudo-labels where a structure-aware GNN and an attribute-only MLP agree.
3. **Domain-adversarial alignment** — align representations across domains after each structural update.

## Repository layout

```
.
├── main/                 # Training entry points
│   ├── train_mlp.py      # Step 1: auxiliary MLP pretraining
│   └── main.py           # Step 2: PSAHS (DANN + progressive reweighting)
├── psahs/                # Core library
│   ├── data/datasets.py  # Loaders + graph adjustment
│   ├── synthetic/        # Noncircle synthetic benchmark generator
│   ├── edge_stats.py     # Edge statistics after adjustment
│   ├── paths.py
│   └── training_utils.py
├── scripts/              # Data download, synthetic graphs, figure plotting
├── dataset/              # Place downloaded data here (not tracked in git)
├── outputs/              # Checkpoints and logs (created at runtime)
└── figures/              # Saved plots (created at runtime)
```

## Installation

From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install PyG scatter/sparse wheels for your PyTorch + CUDA build
```

See [DATA.md](DATA.md) for dataset download and directory layout.

## Quick start

All commands assume the repository root as the working directory.

### 1. Prepare data

**Twitch** (automatic download):

```bash
python scripts/download_twitch_musae.py
```

**Noncircle** (synthetic, generated locally):

```bash
python scripts/generate_synthetic_graphs.py --num_nodes 4000
```

Place **Blog**, **DBLP–ACM**, and **Airport** files under `dataset/` following [DATA.md](DATA.md).

### 2. Pretrain auxiliary MLP (Step 1)

```bash
python main/train_mlp.py -d Blog --src_name Blog1 --tgt_name Blog2
python main/train_mlp.py -d Twitch --src_name EN --tgt_name DE
python main/train_mlp.py -d dblp_acm --src_name ACMv9 --tgt_name DBLPv7
python main/train_mlp.py -d Airport --src_name USA --tgt_name Europe
python main/train_mlp.py -d Noncircle --num_nodes 4000
```

Checkpoints are saved to `outputs/<dataset>/best_mlp_model_seed{seed}_*.pt`.

### 3. Train PSAHS (Step 2)

Use method `DANN_rw` (default) to enable progressive target structure adjustment:

```bash
python main/main.py -d Blog --src_name Blog1 --tgt_name Blog2 --seeds 1 2 3 4 5
python main/main.py -d Twitch --src_name EN --tgt_name DE
python main/main.py -d Noncircle --num_nodes 4000
```

Logs are written to `outputs/<dataset>/train.log`. GNN checkpoints: `best_valid_model_{seed}_*.pt`.

### 4. Reproduce paper figures (training curves)

After Step 1 (MLP checkpoint required):

```bash
python scripts/plot_training_metrics.py -d Blog --src_name Blog1 --tgt_name Blog2 --seeds 1
```

PDFs are saved under `figures/<dataset>/` (target test accuracy, GNN–MLP agreement ratio, pseudo-label accuracy, truncated at best target-test epoch).

For **embedding visualizations** (PCA / t-SNE), use helpers in `psahs/training_utils.py` (`plot_domain_scatter`) from your trained model embeddings.

## Key hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--h_threshold` | 0.6 | Intended node homophily level after adjustment |
| `--start_epoch` | 0 | Epoch to begin target adjustment |
| `--rw_freq` | 20 | Reweight target graph every N epochs |
| `--method` | `DANN_rw` | Must contain `rw` to enable structure adjustment |
| `--seeds` | 1–5 | Random seeds (paper: 5 runs) |

Tuned defaults for Blog/Twitch-style benchmarks are applied automatically in `main/args.py`.

## Supported datasets

| Dataset | `--dataset` | Example `--src_name` / `--tgt_name` |
|---------|-------------|-------------------------------------|
| Blog | `Blog` | `Blog1` / `Blog2` |
| Twitch | `Twitch` | `EN` / `DE` |
| DBLP–ACM | `dblp_acm` | `ACMv9` / `DBLPv7` |
| Airport | `Airport` | `USA` / `Europe` |
| Noncircle (synthetic) | `Noncircle` | (pickles under `dataset/noncircle/`) |

## Citation

If you use this code, please cite the PSAHS paper (*Progressive Graph Structure Adjustment for Homophily Shift Adaptation*).

```bibtex
@inproceedings{psahs2026,
  title={Progressive Graph Structure Adjustment for Homophily Shift Adaptation},
  author={...},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Notes

- Raw data, checkpoints, and result logs are excluded via `.gitignore` (see [DATA.md](DATA.md) for what to download).
- Training-curve plots: `scripts/plot_training_metrics.py` (requires Step 1 MLP checkpoints).
