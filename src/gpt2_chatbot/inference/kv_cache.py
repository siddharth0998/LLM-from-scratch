from __future__ import annotations
from typing import List, Optional, Tuple
import torch

class KV_Cache:
    # Non-sequence dimensions of the ``[batch, num_heads, seq, head_dim]``
    # layout that must match on every append for a given layer.
    _NON_SEQ_DIMS: Tuple[int, ...] = (0, 1, 3)

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: List[Optional[torch.Tensor]] = [None] * n_layers
        self.values: List[Optional[torch.Tensor]] = [None] * n_layers

    def _check_layer_idx(self, layer_idx: int) -> None:
        if layer_idx < 0 or layer_idx >= self.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} is out of range for a cache with "
                f"{self.n_layers} layers (valid range 0..{self.n_layers - 1})"
            )

    def get(
        self, layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        self._check_layer_idx(layer_idx)

        keys = self.keys[layer_idx]
        values = self.values[layer_idx]
        if keys is None or values is None:
            return None
        return keys, values

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_layer_idx(layer_idx)

        existing_keys = self.keys[layer_idx]
        existing_values = self.values[layer_idx]

        if existing_keys is not None and existing_values is not None:
            self._validate_non_seq_dims(layer_idx, existing_keys, new_keys, "keys")
            self._validate_non_seq_dims(
                layer_idx, existing_values, new_values, "values"
            )
            keys = torch.cat([existing_keys, new_keys], dim=2)
            values = torch.cat([existing_values, new_values], dim=2)
        else:
            keys = new_keys
            values = new_values

        self.keys[layer_idx] = keys
        self.values[layer_idx] = values
        return keys, values

    @staticmethod
    def _validate_non_seq_dims(
        layer_idx: int,
        existing: torch.Tensor,
        new: torch.Tensor,
        name: str,
    ) -> None:
        existing_non_seq = tuple(existing.shape[d] for d in KV_Cache._NON_SEQ_DIMS)
        new_non_seq = tuple(new.shape[d] for d in KV_Cache._NON_SEQ_DIMS)
        if existing_non_seq != new_non_seq:
            raise ValueError(
                f"non-sequence dimensions mismatch for {name} at layer "
                f"{layer_idx}: expected {existing_non_seq} "
                f"(dims {KV_Cache._NON_SEQ_DIMS}) from stored shape "
                f"{tuple(existing.shape)}, received {new_non_seq} from shape "
                f"{tuple(new.shape)}"
            )

    def __len__(self) -> int:
        keys = self.keys[0] if self.n_layers > 0 else None
        if keys is None:
            return 0
        return keys.shape[2]
