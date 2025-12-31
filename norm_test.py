import jax
import jax.numpy as jnp
from flax import nnx


def create_patches(x: jax.Array, patch_size: int, stride: int, pad_mode: str = "edge"):
    """
    Extract patches from time series (PatchTST-style) with optional padding.

    Args:
        x: jax.Array of shape (..., n_vars, seq_len)
        patch_size: int, length of each patch
        stride: int, step between patches
        pad_mode: str, padding mode ('edge', 'constant', etc.)

    Returns:
        patches: jax.Array of shape (..., n_vars, n_patches, patch_size)
    """
    seq_len = x.shape[-1]

    if seq_len < patch_size:
        # need to pad to at least patch_size
        pad_len = patch_size - seq_len
        # only pad time axis
        padding = [(0, 0)] * (x.ndim - 1) + [(0, pad_len)]
        x = jnp.pad(x, pad_width=padding, mode=pad_mode)
        seq_len = x.shape[-1]

    # compute required padding so that (seq_len - patch_size) is divisible by stride
    remainder = (seq_len - patch_size) % stride
    if remainder != 0:
        pad_len = stride - remainder
        padding = [(0, 0)] * (x.ndim - 1) + [(0, pad_len)]
        x = jnp.pad(x, pad_width=padding, mode=pad_mode)
        seq_len = x.shape[-1]

    n_patches = (seq_len - patch_size) // stride + 1
    start_indices = jnp.arange(n_patches) * stride
    patch_offsets = jnp.arange(patch_size)
    indices = start_indices[:, None] + patch_offsets  # (n_patches, patch_size)

    patches = x[..., indices]  # -> (..., n_vars, n_patches, patch_size)

    return patches


if __name__ == "__main__":
    rngs = nnx.Rngs(123)

    batch, seq, metrics = 4, 64, 8
    patch_size = 16
    stride = 8
    dummy = jax.random.normal(rngs(), shape=(batch, metrics, seq))

    print(create_patches(dummy, patch_size, stride).shape)
