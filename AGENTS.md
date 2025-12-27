# Agent Guidelines for NetBelief

This document provides essential information for AI coding agents working in the NetBelief codebase.

## Project Overview

NetBelief is a network belief propagation and forecasting system that simulates VoIP traffic over Mininet network topologies and uses LSTM models (JAX/Flax) to forecast network metrics like jitter, packet loss, bandwidth, and MOS (Mean Opinion Score).

**Tech Stack:** Python 3.12+, JAX, Flax NNX, Mininet, Hydra, Grain, Optax, TensorBoard

## Build, Test & Run Commands

### Installation
```bash
# Install all dependencies
make install

# Or using uv directly
uv sync
```

### Running Simulations
```bash
# Run VoIP simulation with default config
uv run python scripts/voip_simulation.py

# Run with specific config
uv run python scripts/voip_simulation.py --config-name simulation_with_variability

# Convert GraphML topology to Python
make graph2py topo=Arn out=arn
```

### Training Models
```bash
# Train LSTM model with Hydra config
uv run python scripts/train_lstm.py

# With specific config
uv run python scripts/train_lstm.py --config-name train_lstm
```

### Testing
```bash
# Run a single test file
uv run python nn_test.py

# No formal test suite exists yet - tests are standalone scripts
```

### Network Operations
```bash
# Clear Mininet state (required after crashes)
make clear-mininet

# Run topology in Mininet
make run-topology topo=pytopo/airtel.py
```

## Code Style Guidelines

### Import Organization
Organize imports in the following order with blank lines between groups:
1. Standard library (e.g., `typing`, `pathlib`, `dataclasses`)
2. Third-party core libraries (e.g., `jax`, `numpy`, `pandas`)
3. ML/DL frameworks (e.g., `flax`, `optax`, `grain`)
4. Domain-specific (e.g., `mininet`, `hydra`)
5. Local imports from `src/`

**Example:**
```python
import typing as tp
from pathlib import Path
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from loguru import logger
from mininet.net import Mininet

from src.modules.lstm import LSTMForecaster
from src.utils.types import ForecasterOutput
```

### Type Annotations
- **Always use type hints** for function signatures
- Use `typing as tp` for brevity: `tp.Optional`, `tp.List`, `tp.Tuple`, `tp.Callable`
- Use `jax.Array` for JAX arrays, not `np.ndarray`
- Use `|` for unions when appropriate: `str | Path` instead of `tp.Union[str, Path]`
- Use `tp.cast()` when dealing with dynamic configs

**Example:**
```python
def process_data(
    data_path: str | Path,
    exclude_columns: tp.Optional[list[str]] = None,
) -> jax.Array:
    ...
```

### Naming Conventions
- **Functions/variables:** `snake_case` (e.g., `train_step`, `loss_fn`, `grad_norm`)
- **Classes:** `PascalCase` (e.g., `LSTMForecaster`, `TrainingConfig`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DATA_SHARDING`, `VOIP_BW`)
- **Private members:** Prefix with `_` (e.g., `_train_step_fn`, `_eval_step_fn`)
- **Type aliases:** `PascalCase` with underscore prefix (e.g., `_Batch`, `_TrainStepFn`)

### Dataclasses & Configs
- Use `@dataclass` for configuration objects
- Use `unsafe_hash=True, eq=True` for configs that need to be hashable
- Use `HfArgumentParser` for parsing Hydra/YAML configs to dataclasses
- Store configs in `config/` or `configs/` directories

**Example:**
```python
@dataclass(unsafe_hash=True, eq=True)
class LSTMForecasterConfig:
    num_metrics: int
    hidden_features: int
    num_heads: int = 4
    horizon: int = 1
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
```

### JAX/Flax Patterns
- Use `nnx.jit` decorator for JIT compilation: `@nnx.jit`
- Use `nnx.Rngs` for random number generation
- Use `nnx.Module` for neural network modules
- Use `chex.assert_rank()` for shape validation in functions
- Use `einops.rearrange()` for tensor reshaping (preferred over reshape)
- Use `jax.device_put()` for explicit device placement
- Use `NamedSharding` with `PartitionSpec` for distributed training

**Example:**
```python
@nnx.jit
def train_step(
    model: LSTMForecaster,
    batch: tuple[jax.Array, jax.Array],
    optimizer: nnx.Optimizer,
) -> jax.Array:
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    return loss
```

### Error Handling & Logging
- Use `loguru` logger, not `print()` or standard logging
- Import as: `from loguru import logger` or `from loguru import Logger`
- Log at appropriate levels: `logger.info()`, `logger.warning()`, `logger.error()`
- Check file existence before loading: `Path.exists()`
- Use descriptive error messages with context

**Example:**
```python
from loguru import logger

def load_data(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}")
    data = pd.read_csv(path)
    
    if data.empty:
        logger.warning(f"Loaded empty dataset from {path}")
    
    return data.values
```

### Hydra Configuration
- Use `@hydra.main()` decorator with `config_path` and `config_name`
- Access nested configs: `cfg["model"]`, `cfg["trainer"]`
- Use `OmegaConf.to_container(cfg, resolve=True)` to convert to dict
- Parse to dataclasses using `HfArgumentParser`

**Example:**
```python
@hydra.main(
    config_path="../configs",
    config_name="train_lstm",
    version_base="1.2",
)
def main(cfg: DictConfig):
    parser = HfArgumentParser(TrainingConfig)
    args = parser.parse_dict(OmegaConf.to_container(cfg["trainer"], resolve=True))[0]
```

### Function Design
- Keep functions focused and single-purpose
- Use descriptive docstrings with Args/Returns sections
- Prefer pure functions when possible (no side effects)
- Use `functools.partial` for currying repeated patterns

**Example:**
```python
def load_csv_data(
    data_path: str | Path,
    exclude_columns: tp.Optional[list[str]] = None,
) -> np.ndarray:
    """
    Load data from CSV file and convert to numpy array.

    Args:
        data_path: Path to CSV file
        exclude_columns: List of column names to exclude

    Returns:
        numpy array of shape (num_devices, num_metrics, timesteps)
    """
    ...
```

### Module Organization
- **`src/modules/`**: Neural network modules (LSTM, attention, etc.)
- **`src/training/`**: Training infrastructure (trainer, callbacks, metrics)
- **`src/data/`**: Data loading and transformations
- **`src/simulation/`**: Mininet simulation components
- **`src/utils/`**: Utility functions (types, arrays, visualization)
- **`scripts/`**: Executable scripts (training, simulation)
- **`pytopo/`**: Generated Mininet topology files
- **`config/` or `configs/`**: YAML configuration files

### Comments & Documentation
- Write docstrings for all public functions and classes
- Use inline comments sparingly - code should be self-documenting
- Explain *why*, not *what* - the code shows what
- Use `# ty: ignore` for type checker overrides (non-standard, project convention)

## Common Patterns

### Loading Checkpoints
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=hub_url,
    local_dir=save_dir,
    token=token,
)
```

### Progress Bars
```python
from tqdm.auto import tqdm

pbar = tqdm(dataloader, desc="Training", total=num_steps)
for batch in pbar:
    pbar.set_postfix(loss=loss.item())
```

### TensorBoard Logging
```python
reporter.log_scalar("train/loss", loss, step=current_step)
reporter.log_scalars({"train/acc": acc, "train/f1": f1}, step=current_step)
```

## Project-Specific Notes

- **Mininet requires root:** Use `sudo` for Mininet commands
- **Simulation data:** Stored in `dumps/` directory
- **Network events:** Logged to `dumps/network_events.json`
- **Variability system:** Configurable network perturbations for realistic data
- **Quantile forecasting:** Models predict multiple quantiles for uncertainty estimation
- **Mesh parallelism:** Configure with `mesh_shape` and `axis_names` in config

## Common Pitfalls

1. **Mininet state:** Always run `make clear-mininet` after simulation crashes
2. **Shape mismatches:** Use `chex.assert_rank()` to validate tensor shapes early
3. **Device placement:** Explicitly use `jax.device_put()` for data sharding
4. **NaN debugging:** Uncomment `jax.debug.print()` statements in training loops
5. **Config parsing:** Use `tp.cast()` after `HfArgumentParser` to help type checkers

---

*Generated for AI coding agents. Keep this file updated as patterns evolve.*
