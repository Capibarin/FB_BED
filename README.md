# scRatio

Flow matching-based density and likelihood ratio estimation.

## Overview

**scRatio** is a Python package that leverages flow matching models to calculate likelihood ratios of various types of data, e.g. single-cell transcriptomics data. Beyond standard flow matching capabilities, scRatio provides specialized tools for:

- **Density Ratio Estimation**: Compute density ratios between different cell populations or conditions
- **Neural ODE Integration**: Advanced neural differential equation-based architectures

The package is built on PyTorch Lightning for efficient training and supports integration with popular single-cell analysis frameworks like Scanpy and AnnData.

## Installation

### From source (development mode)

Clone the repository and install in editable mode with dependencies:

```bash
git clone <repository-url>
cd scRatio
pip install -e .
```

### With development tools

To install with additional development dependencies (testing, linting, Jupyter):

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import anndata as ad
import scRatio as scr

# Load your single-cell data
adata = ad.read_h5ad("your_data.h5ad")

# Create a data module
data_module = scr.AnnDataDataModule(adata, batch_size=32)

# Initialize flow matching model
model = scr.ConditionalFlowMatchingWithScore(
    input_dim=adata.n_vars,
    hidden_dim=256
)

# Train using PyTorch Lightning
from lightning import Trainer
trainer = Trainer(max_epochs=100, accelerator="gpu")
trainer.fit(model, data_module)
```

## Features

- 🧬 **Single-cell focused**: Optimized for high-dimensional transcriptomic data
- ⚡ **Efficient training**: Built on PyTorch Lightning with GPU support
- 🔄 **Flow matching**: State-of-the-art generative modeling
- 📊 **Density ratios**: Calculate likelihood ratios between cell populations
- 🧠 **Neural ODEs**: Advanced NODE-based architectures for complex dynamics
- 📦 **AnnData compatible**: Seamless integration with the Scanpy ecosystem

## Requirements

- Python >= 3.9
- PyTorch >= 2.5.1
- Lightning >= 2.4.0
- Scanpy >= 1.11.2
- AnnData >= 0.11.4

See [requirements.txt](requirements.txt) for the complete list of dependencies.

## Project Structure

```
src/
├── scRatio/
│   ├── __init__.py
│   ├── utils.py
│   ├── datamodules/      # Data loading and preprocessing
│   │   ├── datamodule.py
│   │   └── dataset.py
│   └── models/           # Flow matching and NODE models
│       ├── flow_matching.py
│       └── node_wrappers.py
```

## Usage Examples

> **Note**: For the PBMC and ComboSciplex examples below, you need to download the respective datasets and place them in the corresponding data folders within the notebook directories.

### 1) Standard workflow with AnnData

This uses AnnData objects and the built-in datamodule. The example below is adapted from [notebooks/pbmc10m/scRatio_pbmc10m.ipynb](notebooks/pbmc10m/scRatio_pbmc10m.ipynb), which contains the PBMC experiment comparing our likelihood ratios with cytokine efficiencies reported in a prior paper.

```python
import lightning as L
import numpy as np
import torch

from scRatio.datamodules import AnnDataDataModule
from scRatio.models import ConditionalFlowMatchingWithScore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
dim = 50

datamodule = AnnDataDataModule(
    adata_path="./data/pbmc10m.h5ad",
    conditions=["donor", "cytokine"],
    num_features=dim,
    train_batch_size=batch_size,
    val_batch_size=batch_size,
    test_batch_size=batch_size,
    val_split=0,
    test_split=0.1,
    num_workers=5,
    seed=22,
    representation_key="X_pca",
)
datamodule.setup()

cond_dims = datamodule.condition_dims
n_steps = 100_000
sigma = 0
sigma_min = 1e-3

lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2

model = ConditionalFlowMatchingWithScore(
    input_dim=dim,
    cond_dims=cond_dims,
    hidden_dims=[1024, 1024, 1024],
    encoder_hidden_dims=[256],
    encoder_out_dim=64,
    lambda_t=lambda_t,
    lambda_sp_t=lambda_sp_t,
    betas=[0, 0],
    lr=1e-4,
    time_feature_dim=32,
    encoder_out_dim_cond=32,
)

trainer = L.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_steps=n_steps,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    val_check_interval=None,
    limit_val_batches=0,
)
trainer.fit(model, datamodule=datamodule)
model = model.to(device)
```

### 2) Custom DataLoader with Lightning training

This example uses a custom dataset and DataLoader, but still trains via Lightning. The snippet is taken from [runs/scripts/make_comparison.py](runs/scripts/make_comparison.py). The script compares different schedules against a baseline method, and the resulting plots are in [notebooks/gaussian_tests](notebooks/gaussian_tests).

```python
import lightning as L
import torch

from torch.utils.data import DataLoader
from scRatio.datamodules import ArrayDataset
from scRatio.models import ConditionalFlowMatchingWithScore

def build_train_loader(x_train, c_train, batch_size):
    dataset = ArrayDataset(x_train, c_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

train_loader = build_train_loader(X_train, C_train, batch_size=512)

model = ConditionalFlowMatchingWithScore(
    input_dim=n,
    cond_dims=[cond_dim],
    hidden_dims=[1024, 1024, 1024],
    encoder_hidden_dims=[256],
    encoder_out_dim=latent_dim,
    lambda_t=lambda_t,
    lambda_sp_t=lambda_sp_t,
    betas=[0],
    lr=1e-4,
    time_feature_dim=time_feature_dim,
    encoder_out_dim_cond=cond_latent_dim,
)

trainer = L.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_steps=100_000,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
)
trainer.fit(model, train_dataloaders=train_loader)
```

### 3) Custom training loop

This example uses a fully custom training loop. The snippet is adapted from [notebooks/combosciplex/scRatio_ae_5.ipynb](notebooks/combosciplex/scRatio_ae_5.ipynb), where we apply our method to drug efficiency assessment on the ComboSciplex dataset.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 512
cond_dims = list(np.max(C_both, axis=0) + 1)
num_cond = C_both.shape[1]

sigma = 0
sigma_min = 1e-3

lambda_t = lambda t: torch.sqrt((1 - (1 - sigma_min) * t) ** 2 + sigma * t * (1 - t))
lambda_sp_t = lambda t: (sigma * (1 - 2 * t) - 2 * (1 - sigma_min) * (1 - (1 - sigma_min) * t)) / 2

model = ConditionalFlowMatchingWithScore(
    input_dim=dim,
    cond_dims=cond_dims,
    hidden_dims=[1024, 1024],
    encoder_hidden_dims=[256],
    encoder_out_dim=64,
    lambda_t=lambda_t,
    lambda_sp_t=lambda_sp_t,
    betas=[0, 0],
    lr=1e-4,
    time_feature_dim=32,
    encoder_out_dim_cond=32,
).to(device)
optimizer = model.configure_optimizers()

for k in range(200_000):
    optimizer.zero_grad()
    x1, cond = prepare_both_batch(X_both, C_both, cond_dims, num_cond, batch_size, device)
    loss = model.shared_step(x1, cond)
    loss.backward()
    optimizer.step()
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

MIT License - see LICENSE file for details.

## Citation

If you use scRatio in your research, please cite:

```bibtex
@misc{antipov2026flowbaseddensityratioestimation,
      title={Flow-Based Density Ratio Estimation for Intractable Distributions with Applications in Genomics}, 
      author={Egor Antipov and Alessandro Palma and Lorenzo Consoli and Stephan Günnemann and Andrea Dittadi and Fabian J. Theis},
      year={2026},
      eprint={2602.24201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.24201}, 
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
