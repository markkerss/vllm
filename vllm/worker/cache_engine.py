# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
from typing import List, Dict, Tuple

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
    
from vllm.core.block_manager import BlockTable


logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            layer_kv_cache = torch.zeros(kv_cache_shape,
                                         dtype=self.dtype,
                                         pin_memory=pin_memory,
                                         device=device)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        # NOTE: src_to_dst is a map from CPU block number to GPU block number.
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        # NOTE: src_to_dst is a map from GPU block number to CPU block number.
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        # NOTE: src_to_dsts is a map from source GPU block number to a list of
        # destination GPU block numbers.
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    # ---- KV Cache Export/Import Methods ----

    def get_kv_cache_shape_per_block(self) -> Tuple[int, ...]:
        """Returns the shape of the KV cache for a single block for one layer."""
        # Shape is (2, block_size, num_kv_heads, head_size)
        # Derived from the global shape (2, num_blocks, block_size, ...)
        # Use 1 for num_blocks if it's 0, as get_kv_cache_shape might expect > 0
        num_blocks_for_shape = self.num_gpu_blocks if self.num_gpu_blocks else 1
        global_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks_for_shape,
            self.block_size,
            self.num_kv_heads,
            self.head_size)

        # Expected global shape: (2, num_blocks, block_size, num_kv_heads, head_size)
        if len(global_shape) != 5 or global_shape[0] != 2:
             logger.warning(f"Unexpected KV cache shape from backend: {global_shape}. "
                            "Assuming standard per-block shape: (2, block_size, num_kv_heads, head_size).", exc_info=True)
             # Return the expected shape directly based on parameters as fallback
             return (2, self.block_size, self.num_kv_heads, self.head_size)

        # Extract shape by removing the num_blocks dimension (index 1)
        per_block_shape = (global_shape[0], ) + global_shape[2:]
        # Expected per_block_shape: (2, block_size, num_kv_heads, head_size)
        return per_block_shape

    def export_blocks_to_cpu_buffer(
        self,
        block_table: BlockTable
    ) -> List[List[torch.Tensor]]:
        """Copies the KV cache data from GPU blocks specified by a block table
           to a CPU buffer keyed by logical block index.

        Args:
            block_table: The block table mapping logical indices to physical blocks
                         for the sequence group to export.

        Returns:
            A list of lists of CPU tensors.
            Each list contains one tensor per layer, holding the combined K/V
            data for that block (shape: (2, block_size, ...)).
        """
        cpu_buffer: List[List[torch.Tensor]] = []
        if not self.gpu_cache:
            logger.warning("GPU cache is empty. Cannot export.")
            return cpu_buffer
        if self.num_gpu_blocks == 0:
            logger.warning("No GPU blocks allocated. Cannot export.")
            return cpu_buffer

        # Pre-calculate expected per-block shape for verification
        expected_block_shape = self.get_kv_cache_shape_per_block()

        for logical_idx, block in enumerate(block_table):
            if not block.is_gpu:
                logger.debug(f"Skipping non-GPU block at logical index {logical_idx}")
                continue

            physical_gpu_id = block.block_number
            if physical_gpu_id >= self.num_gpu_blocks:
                 logger.error(f"Physical GPU block ID {physical_gpu_id} for logical "
                              f"index {logical_idx} is out of bounds ({self.num_gpu_blocks}). Skipping.")
                 continue # Skip this block

            combined_tensors_for_block: List[torch.Tensor] = []
            layer_idx = 0 # Define layer_idx before the try block for error logging
            try:
                for layer_idx in range(self.num_attention_layers):
                    # Slice the combined K/V data for the physical block
                    # Assumes FlashAttentionBackend layout [2, num_blocks, block_size, num_heads, head_size]
                    # Indexing the 'num_blocks' dimension (dim 1)
                    combined_gpu_slice = self.gpu_cache[layer_idx][:, physical_gpu_id, ...]

                    # Verify shape before copying
                    if combined_gpu_slice.shape != expected_block_shape:
                         logger.warning(f"Shape mismatch for GPU slice layer {layer_idx} block {physical_gpu_id}. "
                                       f"Expected {expected_block_shape}, Got {combined_gpu_slice.shape}. "
                                       "Copying anyway, but check cache layout.")

                    # Copy to CPU
                    combined_tensors_for_block.append(combined_gpu_slice.cpu())

                cpu_buffer[logical_idx] = combined_tensors_for_block

            except IndexError:
                 logger.error(f"IndexError accessing GPU block {physical_gpu_id} in layer {layer_idx}. "
                              f"GPU cache shape: {self.gpu_cache[layer_idx].shape}. "
                              f"Is num_gpu_blocks ({self.num_gpu_blocks}) correct?", exc_info=True)
                 continue # Skip this block
            except Exception as e:
                 logger.error(f"Error processing block {physical_gpu_id} (logical {logical_idx}): {e}", exc_info=True)
                 continue # Skip this block

        return cpu_buffer

    def import_cpu_buffer_to_gpu(
        self,
        cpu_buffer: List[List[torch.Tensor]],
        block_table: BlockTable # Use string for forward ref if needed
    ) -> None:
        """Copies KV cache data from a CPU buffer (keyed by logical index)
           into target GPU blocks specified by the block_table.

        Args:
            cpu_buffer: A dictionary mapping logical block index to a list of
                        CPU tensors (one per layer, shape (2, block_size,...)).
            block_table: The block table for the target sequence group, mapping
                         logical block numbers to allocated physical GPU blocks.
        """
        if not cpu_buffer:
             logger.debug("Import CPU buffer is empty. Nothing to import.")
             return
        if not self.gpu_cache:
            logger.error("GPU cache is empty. Cannot import.")
            return
        if self.num_gpu_blocks == 0:
            logger.error("No GPU blocks allocated. Cannot import.")
            return

        # Pre-calculate expected per-block shape for verification
        expected_block_shape = self.get_kv_cache_shape_per_block()

        for logical_idx, block in enumerate(block_table):
            if not block.is_gpu:
                # This implies the block wasn't allocated correctly on the target
                logger.error(f"Target block at logical index {logical_idx} is not on GPU. Cannot import.")
                continue
            if logical_idx not in cpu_buffer:
                logger.warning(f"Logical block {logical_idx} not found in cpu_buffer. Skipping import.")
                continue

            target_physical_gpu_id = block.block_number
            if target_physical_gpu_id >= self.num_gpu_blocks:
                 logger.error(f"Target physical block ID {target_physical_gpu_id} "
                              f"is out of bounds ({self.num_gpu_blocks}). Skipping import.")
                 continue

            combined_cpu_layers = cpu_buffer[logical_idx]
            if len(combined_cpu_layers) != self.num_attention_layers:
                 logger.error(f"Mismatch in number of layers for logical block {logical_idx}. "
                              f"Expected {self.num_attention_layers}, got {len(combined_cpu_layers)}. Skipping import.")
                 continue

            layer_idx = 0 # Define layer_idx before try block for error logging
            try:
                for layer_idx in range(self.num_attention_layers):
                    # Get the target combined K/V slice in the GPU cache
                    # Indexing assumes FlashAttentionBackend layout [2, num_blocks, ...]
                    target_combined_slice = self.gpu_cache[layer_idx][:, target_physical_gpu_id, ...]
                    # Get the source combined CPU tensor
                    combined_cpu = combined_cpu_layers[layer_idx]

                    # Verify shapes match before copying
                    shape_match = (target_combined_slice.shape == expected_block_shape and
                                   combined_cpu.shape == expected_block_shape)

                    if not shape_match:
                         logger.error(f"Shape mismatch cannot perform copy for layer {layer_idx}, logical block {logical_idx}. "
                                      f"Target GPU Slice: {target_combined_slice.shape}, Source CPU Tensor: {combined_cpu.shape}, Expected: {expected_block_shape}")
                         continue # Skip copy for this layer if shapes don't match expectation

                    # Perform the single copy operation for the combined K/V tensor
                    target_combined_slice.copy_(combined_cpu, non_blocking=True)

            except IndexError:
                 logger.error(f"IndexError accessing target GPU block {target_physical_gpu_id} in layer {layer_idx}. "
                              f"GPU cache shape: {self.gpu_cache[layer_idx].shape}. "
                              f"Is num_gpu_blocks ({self.num_gpu_blocks}) correct?", exc_info=True)
                 continue # Skip rest of layers for this block
            except Exception as e:
                 logger.error(f"Error copying data for logical block {logical_idx} "
                              f"to physical block {target_physical_gpu_id}: {e}", exc_info=True)
                 continue # Skip rest of layers for this block

        # Ensure copies are synchronized if necessary, depending on stream usage
        # Consider adding torch.cuda.synchronize() here if operations
        # need to be guaranteed complete before proceeding.

    # ---- End KV Cache Export/Import Methods ----

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
