/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

// SM90 non-cooperative kernel wrapper for dual-weight standalone GEMM.
// Identical to sm90_gemm_tma_warpspecialized.hpp except matches on
// KernelTmaWarpSpecializedCustom schedule tag (used by the dual-weight
// standalone mainloop). The cooperative path uses the standard cooperative
// kernel wrapper via KernelTmaWarpSpecializedCooperative inheritance.

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cutlass/conv/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/arch/cluster_sm90.hpp"

#include "cutlass/arch/grid_dependency_control.h"


///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<cutlass::gemm::KernelTmaWarpSpecializedCustom, typename CollectiveMainloop_::DispatchPolicy::Schedule>>
>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static constexpr bool IsConvProblemShape = not (cute::is_tuple_v<ProblemShape>|| IsCutlass3ArrayKernel<ProblemShape>::value);
  static_assert( IsConvProblemShape || (cute::rank(ProblemShape{}) == 3 || cute::rank(ProblemShape{}) == 4), "ProblemShape{} should be <M,N,K> or <M,N,K,L> for Gemm");

  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "TMA warp-specialized kernel does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileSchedulerTag, ArchTag, TileShape, ClusterShape>::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Kernel level shared memory storage
  struct SharedStorage {
    union TensorStorage {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      EpilogueTensorStorage epilogue;
    } tensors;

    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
    } pipelines;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 1;
  static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMma{})) + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};

    Arguments() = default;

    Arguments(
        cutlass::gemm::GemmUniversalMode mode_,
        ProblemShape problem_shape_,
        MainloopArguments mainloop_,
        EpilogueArguments epilogue_,
        KernelHardwareInfo hw_info_ = KernelHardwareInfo(),
        TileSchedulerArguments scheduler_ = TileSchedulerArguments())
    : mode(mode_)
      , problem_shape(problem_shape_)
      , mainloop(mainloop_)
      , epilogue(epilogue_)
      , hw_info(hw_info_)
      , scheduler(scheduler_) {}

    Arguments(
        ProblemShape problem_shape_,
        MainloopArguments mainloop_,
        EpilogueArguments epilogue_,
        KernelHardwareInfo hw_info_ = KernelHardwareInfo(),
        TileSchedulerArguments scheduler_ = TileSchedulerArguments())
    : mode(cutlass::gemm::GemmUniversalMode::kGemm)
      , problem_shape(problem_shape_)
      , mainloop(mainloop_)
      , epilogue(epilogue_)
      , hw_info(hw_info_)
      , scheduler(scheduler_) {}

  };

  // Kernel entry point API
  struct Params {
    using ProblemShapeMNKL = decltype(cutlass::conv::detail::get_problem_shape_MNKL_helper<CollectiveMainloop>(ProblemShape{}, cute::conditional_t<IsConvProblemShape, cute::true_type, cute::false_type>{}));
    ProblemShapeMNKL problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
  };

  //
  // Methods
  //

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {

    (void) workspace;
    auto problem_shape_mnkl = cutlass::conv::detail::get_problem_shape_MNKL_helper<CollectiveMainloop>(args.problem_shape, cute::conditional_t<IsConvProblemShape, cute::true_type, cute::false_type>{});
    auto transformed_problem_shape = cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape);

    auto swapped_problem_shape = problem_shape_mnkl;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      get<0>(swapped_problem_shape) = get<1>(problem_shape_mnkl);
      get<1>(swapped_problem_shape) = get<0>(problem_shape_mnkl);
    }
    return {
      swapped_problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(transformed_problem_shape, args.epilogue, workspace)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = true;
    auto transformed_problem_shape = cutlass::conv::detail::get_transformed_problem_shape_MNKL(args.problem_shape);

    if (!implementable) {
        CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
        return implementable;
    }

    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(transformed_problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    auto cluster_shape = ClusterShape{};
    auto tile_shape = TileShape{};
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    return TileScheduler::get_tiled_cta_shape_mnl(
        problem_shape_MNKL, tile_shape, cluster_shape);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the WGMMA ISA is arch conditional to sm90a.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    CUTE_INVALID_CONTROL_PATH("ERROR : Arch conditional MMA instruction used without targeting sm90a compute capability. Aborting.\n");
#else

    enum class WarpGroupRole {
      Producer = 0,
      Consumer = 1,
    };
    enum class ProducerWarpRole {
      MainloopEpilogue = 0,
      Warp1 = 1,
      Warp2 = 2,
      Warp3 = 3
    };

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();


    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    auto cluster_wait_fn = [&] () {
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    static_assert(IsConvProblemShape || cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(IsConvProblemShape || cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    auto blk_shape = TileShape{};
    TiledMma tiled_mma;

    auto problem_shape_MNKL = append<4>(params.problem_shape, cute::Int<1>{});

    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.epilogue);

    // load_init returns (gA2_mkl, gB_nkl, gA3_mkl) for dual-weight mainloop.
    // get<0> is A2 (M-tiled), get<1> is B (N-tiled).
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto m_coord = idx2crd(int(blockIdx.x), shape<2>(gA_mkl));
    auto n_coord = idx2crd(int(blockIdx.y), shape<2>(gB_nkl));
    auto l_coord = [&] (auto const& gB_nkl_) {
      if constexpr (not IsConvProblemShape) {
        return idx2crd(int(blockIdx.z), shape<4>(gB_nkl_));
      }
      else {
        return Int<0>{};
      }
    } (gB_nkl);

    auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

    auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));
    auto k_tile_count = size<3>(gA_mkl);

    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      if (producer_warp_role == ProducerWarpRole::MainloopEpilogue) {
        cutlass::arch::wait_on_dependent_grids();
        collective_mainloop.load(
          params.mainloop,
          mainloop_pipeline,
          mainloop_pipe_producer_state,
          load_inputs,
          blk_coord,
          k_tile_iter, k_tile_count,
          lane_idx,
          block_rank_in_cluster,
          shared_storage.tensors.mainloop
        );
        mainloop_pipe_producer_state.advance(k_tile_count);
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if (collective_epilogue.is_producer_load_needed()) {
          __syncwarp();
          epi_load_pipe_producer_state = collective_epilogue.load(
            epi_load_pipeline,
            epi_load_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            tiled_mma,
            lane_idx,
            shared_storage.tensors.epilogue
          );
          collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        }
      }
    }
    else if (warp_group_role == WarpGroupRole::Consumer) {
      Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));

      collective_mainloop.mma(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        accumulators,
        k_tile_count,
        warp_group_thread_idx,
        shared_storage.tensors.mainloop,
        params.mainloop
      );

      collective_mainloop.mma_tail(
        mainloop_pipeline,
        mainloop_pipe_consumer_state,
        k_tile_count
      );

      cutlass::arch::launch_dependent_grids();

      auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
      collective_epilogue.store(
        epi_load_pipeline,
        epi_load_pipe_consumer_state,
        epi_store_pipeline,
        epi_store_pipe_producer_state,
        problem_shape_MNKL,
        blk_shape,
        blk_coord,
        accumulators,
        tiled_mma,
        warp_group_thread_idx,
        shared_storage.tensors.epilogue
      );

      collective_epilogue.store_tail(
        epi_load_pipeline,
        epi_load_pipe_consumer_state_next,
        epi_store_pipeline,
        epi_store_pipe_producer_state_next
      );
    }
#endif
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
