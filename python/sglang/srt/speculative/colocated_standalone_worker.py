"""Colocated Speculative Decoding Worker with SM Partitioning.

This worker pipelines draft and target model execution on the same GPU
using SM partitioning to enable concurrent execution.

Pipeline:
    Time T:   draft(batch_N)   ||  verify(batch_N-1)
    Time T+1: draft(batch_N+1) ||  verify(batch_N)
"""

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import EAGLEWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context, load_token_map
from sglang.srt.utils import empty_context, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class ColocatedStandaloneWorker(EAGLEWorker):
    """Pipelined draft/target execution with SM partitioning.

    This worker extends the standard speculative decoding workflow by:
    1. Running draft and verify phases on separate CUDA streams
    2. Partitioning SMs between draft and target models
    3. Pipelining: draft(N) runs concurrently with verify(N-1)
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        with empty_context(), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            TpModelWorker.__init__(
                self,
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            )

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        # Initialize SM partitioning and pipeline state
        self._init_colocated()

    def _init_colocated(self):
        """Initialize SM partitioning and pipeline state."""
        # Create separate CUDA streams for draft and target
        self.draft_stream = torch.cuda.Stream(device=self.device)
        self.target_stream = torch.cuda.Stream(device=self.device)

        # Initialize SM controller for partitioning
        self.sm_ctrl = None
        self.sm_partition_enabled = False

        # Get SM partition ratio from environment or use default
        draft_ratio = float(os.environ.get("COSPEC_DRAFT_SM_RATIO", "0.2"))

        try:
            from sglang.srt.utils.sm_controller import SMController
            self.sm_ctrl = SMController()

            if self.sm_ctrl.enabled:
                # Partition SMs: draft gets draft_ratio, target gets the rest
                draft_tpcs = int(self.sm_ctrl.total_tpcs * draft_ratio)
                draft_tpcs = max(1, draft_tpcs)  # At least 1 TPC for draft

                self.sm_ctrl.set_stream_mask(self.draft_stream, 0, draft_tpcs)
                self.sm_ctrl.set_stream_mask(
                    self.target_stream, draft_tpcs, self.sm_ctrl.total_tpcs
                )
                self.sm_partition_enabled = True
                logger.info(
                    f"ColocatedWorker: SM partitioning enabled. "
                    f"Draft: TPCs 0-{draft_tpcs}, Target: TPCs {draft_tpcs}-{self.sm_ctrl.total_tpcs}"
                )
        except Exception as e:
            logger.warning(
                f"ColocatedWorker: SM partitioning disabled ({e}). "
                f"Falling back to sequential execution with stream overlap."
            )

        logger.info("ColocatedStandaloneWorker initialized")

    def _run_verify(self, batch: ScheduleBatch, spec_info) -> GenerationBatchResult:
        """Run verification phase."""
        logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
            self.verify(batch, spec_info)
        )

        # Run draft extend after decode if needed
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            if (
                self.server_args.enable_dp_attention
                or batch.spec_info.verified_id.shape[0] > 0
            ):
                self.forward_draft_extend_after_decode(batch)

        # Clone spec_info tensors to detach from CUDA graph output buffers.
        # Without this, the next CUDA graph replay (for a different batch)
        # would overwrite these shared output buffers.
        if batch.spec_info is not None:
            if batch.spec_info.topk_p is not None:
                batch.spec_info.topk_p = batch.spec_info.topk_p.clone()
            if batch.spec_info.topk_index is not None:
                batch.spec_info.topk_index = batch.spec_info.topk_index.clone()
            if batch.spec_info.hidden_states is not None:
                batch.spec_info.hidden_states = batch.spec_info.hidden_states.clone()

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=verify_output.verified_id,
            num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
            accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def forward_dual_batch_decode(self, batch_draft, batch_verify, spec_info_verify):
        """draft(batch_draft) || verify(batch_verify) concurrently on separate streams."""
        default_stream = torch.cuda.current_stream()
        self.target_stream.wait_stream(default_stream)
        self.draft_stream.wait_stream(default_stream)

        # Launch verify on target_stream
        with torch.cuda.stream(self.target_stream):
            verify_result = self._run_verify(batch_verify, spec_info_verify)

        # Launch draft on draft_stream
        with torch.cuda.stream(self.draft_stream):
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                spec_info = self.draft(batch_draft)

        self.draft_stream.synchronize()
        self.target_stream.synchronize()

        return spec_info, verify_result

    def forward_verify_only(self, batch, spec_info):
        """Run verify phase only (when draft queue is empty)."""
        return self._run_verify(batch, spec_info)

    def forward_draft_only(self, batch):
        """Run draft phase only (first iteration for a queue)."""
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            return self.draft(batch)
