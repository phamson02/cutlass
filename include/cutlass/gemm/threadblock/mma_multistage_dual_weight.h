/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel that reconstructs FP16 B
           from two FP8 lanes (upper/lower).
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_mixed_input_tensor_op.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"

#include "cutlass/gemm/threadblock/mma_base.h"

#include <cstdint>
#include <cstdio>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock-scoped MMA that streams FP8 B-upper/B-lower, reconstructs FP16 in registers, and
/// feeds a standard tensor core MMA.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory (FP8 upper/lower share this iterator type)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Layout of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// When true, issue MMA(k) before transform(k+1) so FP8->FP16 reconstruction
    /// integer ops can overlap the tensor-core pipeline (k-block interleaving).
    bool kKBlockInterleaved = false,
    /// When true and ElementB is float_e5m2_t, use truncation reconstruction
    /// (upper = fp16[15:8], lower = fp16[7:0]) instead of round-to-nearest-even.
    bool kTruncateE5M2 = false,
    /// When true and ElementB is float_e4m3_t, use FastNumericArrayConverter
    /// reconstruction (sign<<8 | em<<7) matching cutlass::FastNumericArrayConverter.
    bool kFastE4M3 = false,
    /// Used for partial specialization
    typename Enable = bool>
class MmaMultistageDualWeight {
public:
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory (upper/lower FP8)
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Warp-level iterator that loads FP8 B tiles from shared memory.
  using WarpIteratorB = typename Policy::WarpIteratorB;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;

  /// Number of warp-level GEMM operations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the B operand (Global Memory)
  using TensorRefB = TensorRef<typename IteratorB::Element, typename IteratorB::Layout>;

  /// Tensor reference to the B operand (Shared Memory)
  using TensorRefB_Smem = TensorRef<typename IteratorB::Element, typename SmemIteratorB::Layout>;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + kWarpGemmIterations - 1) / kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + kWarpGemmIterations - 1) / kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:

  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename WarpIteratorB::Fragment; // FP8
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_upper_[2];
    WarpLoadedFragmentB warp_loaded_frag_B_lower_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };

 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterators to write threadblock-scoped tiles of B operand to shared memory (upper/lower)
  SmemIteratorB smem_iterator_B_upper_;
  SmemIteratorB smem_iterator_B_lower_;

  // Warp Iterator A
  typename Operator::IteratorA warp_tile_iterator_A_;

  // Warp iterators for B (FP8)
  typename Policy::WarpIteratorB warp_tile_iterator_B_upper_;
  typename Policy::WarpIteratorB warp_tile_iterator_B_lower_;

  static_assert(WarpIteratorB::Fragment::kElements == Operator::FragmentB::kElements,
                "Warp iterator B fragment size must match operator fragment.");

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

 public:

  /// Shared Storage to hold both B_upper and B_lower
  struct SharedStorage {
    // Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;

    // Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;

    using ElementA = typename Operator::ElementA;
    using LayoutA = typename SmemIteratorA::Layout;

    using ElementB = typename IteratorB::Element; // FP8
    using LayoutB = typename SmemIteratorB::Layout;   // FP8 Layout

    // A operand storage
    AlignedBuffer<ElementA, ShapeA::kCount> operand_A;

    // B operand storage (Upper)
    AlignedBuffer<ElementB, ShapeB::kCount> operand_B_upper;

    // B operand storage (Lower)
    AlignedBuffer<ElementB, ShapeB::kCount> operand_B_lower;

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static LayoutA LayoutA_() {
      return LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static LayoutB LayoutB_() {
      return LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }

    CUTLASS_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA_()};
    }

    CUTLASS_DEVICE
    TensorRefB_Smem operand_B_upper_ref() {
      return TensorRefB_Smem{operand_B_upper.data(), LayoutB_()};
    }

    CUTLASS_DEVICE
    TensorRefB_Smem operand_B_lower_ref() {
      return TensorRefB_Smem{operand_B_lower.data(), LayoutB_()};
    }
  };

  /// Construct from tensor references
  CUTLASS_DEVICE
  MmaMultistageDualWeight(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_upper_(shared_storage.operand_B_upper_ref(), thread_idx),
      smem_iterator_B_lower_(shared_storage.operand_B_lower_ref(), thread_idx),
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx),
      warp_tile_iterator_B_upper_(shared_storage.operand_B_upper_ref(), lane_idx),
      warp_tile_iterator_B_lower_(shared_storage.operand_B_lower_ref(), lane_idx),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_idx_k = warp_idx / (WarpCount::kM * WarpCount::kN);

    int warp_idx_m = warp_idx_mn % WarpCount::kM;
    int warp_idx_n = warp_idx_mn / WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, kWarpGemmIterations * warp_idx_k});

    this->warp_tile_iterator_B_upper_.add_tile_offset(
        {kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B_lower_.add_tile_offset(
        {kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -kStages * Policy::kPartitionsK * kWarpGemmIterations});
      warp_tile_iterator_B_upper_.add_tile_offset({-kStages * Policy::kPartitionsK * kWarpGemmIterations, 0});
      warp_tile_iterator_B_lower_.add_tile_offset({-kStages * Policy::kPartitionsK * kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA &iterator_A,
    IteratorB &iterator_B_upper,
    IteratorB &iterator_B_lower)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B_upper.add_tile_offset({1, 0});
    iterator_B_lower.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_upper_.add_tile_offset({1, 0});
    smem_iterator_B_lower_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -kStages});
      smem_iterator_B_upper_.add_tile_offset({-kStages, 0});
      smem_iterator_B_lower_.add_tile_offset({-kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B_upper, IteratorB &iterator_B_lower,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B_upper.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_upper_.set_iteration_index(group_start_B);

    // Async Copy for operand B Upper
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_upper_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B_upper.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B_upper.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B_upper.valid());
          }

          ++iterator_B_upper;
        }
        ++this->smem_iterator_B_upper_;
      }
    }

    iterator_B_lower.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_lower_.set_iteration_index(group_start_B);

    // Async Copy for operand B Lower
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_lower_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B_lower.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B_lower.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B_lower.valid());
          }

          ++iterator_B_lower;
        }
        ++this->smem_iterator_B_lower_;
      }
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B_upper,      ///< [in|out] iterator over B operand in global memory
    IteratorB &iterator_B_lower,      ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B_upper.clear_mask(gemm_k_iterations == 0);
      iterator_B_lower.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B_upper.set_iteration_index(0);
      this->smem_iterator_B_upper_.set_iteration_index(0);

      // Async Copy for operand B Upper
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_upper_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B_upper.get(), iterator_B_upper.valid());

          ++iterator_B_upper;
        }

        ++this->smem_iterator_B_upper_;
      }

      iterator_B_lower.set_iteration_index(0);
      this->smem_iterator_B_lower_.set_iteration_index(0);

      // Async Copy for operand B Lower
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_lower_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B_lower.get(), iterator_B_lower.valid());

          ++iterator_B_lower;
        }

        ++this->smem_iterator_B_lower_;
      }

      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B_upper, iterator_B_lower);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Optionally clear the remaining stages of SMEM.
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory (Upper)
      SmemIteratorB last_smem_iterator_B_upper(this->smem_iterator_B_upper_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B_upper.set_iteration_index(0);

      // Async Copy for operand B Upper
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B_upper.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B_upper;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory (Lower)
      SmemIteratorB last_smem_iterator_B_lower(this->smem_iterator_B_lower_);
      last_smem_iterator_B_lower.set_iteration_index(0);

      // Async Copy for operand B Lower
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B_lower.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B_lower;
      }
    }
  }


  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<kStages - 2>();
    __syncthreads();
  }



  // Helper to reconstruct FP16 from FP8 Upper and Lower
  CUTLASS_DEVICE
  void reconstruct_fp16(
      typename PipeState::WarpTransformedFragmentB &dst_B,
      typename PipeState::WarpLoadedFragmentB const &frag_B_upper,
      typename PipeState::WarpLoadedFragmentB const &frag_B_lower) const {

      using ElementLoad = typename IteratorB::Element;  // fp8
      using ElementMmaB = typename Operator::ElementB;  // fp16
      using MmaIterations = typename Operator::MmaIterations;
      using MmaOperandB = typename Operator::ArchMmaOperator::FragmentB;

      static_assert(Operator::FragmentB::kElements % 4 == 0,
                    "Fragment size must be divisible by 4 for packed reconstruct.");

      // Shuffle fp8 fragments into mma-friendly layout before reconstruction.
      // cutlass::gemm::warp::detail::FragmentShuffler<
      //     ElementMmaB,
      //     ElementLoad,
      //     MmaIterations::kColumn,
      //     Operator::FragmentB::kElements,
      //     MmaOperandB::kElements,
      //     cutlass::gemm::Operand::kB>
      //     shuffler;

      // auto shuffled_upper = shuffler(frag_B_upper);
      // auto shuffled_lower = shuffler(frag_B_lower);

      uint32_t const* upper_u32 = reinterpret_cast<uint32_t const*>(&frag_B_upper);
      uint32_t const *lower_u32 = reinterpret_cast<uint32_t const *>(&frag_B_lower);
      uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(&dst_B);

      int const vec_quads = Operator::FragmentB::kElements / 4;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < vec_quads; ++i) {
        uint32_t a = upper_u32[i];
        uint32_t b = lower_u32[i];

        if constexpr (cutlass::platform::is_same<ElementLoad, cutlass::float_e5m2_t>::value) {
          if constexpr (kTruncateE5M2) {
            // Truncation variant: upper = fp16[15:8], lower = fp16[7:0].
            // No rounding correction needed — just interleave the two bytes.
            dst_u32[2 * i]     = __byte_perm(a, b, 0x1504u);
            dst_u32[2 * i + 1] = __byte_perm(a, b, 0x3726u);
          } else {
            // E5M2 NestedFP round-to-nearest-even reconstruction.
            // upper stored = upper_orig + inc, lower stored = lower_orig | inc (normal only).
            // Detect finite-normal bytes: exponent bits [6:2] != 0.
            // inc is in the LSB of stored_lower for normal values.
            uint32_t exp_bits   = a & 0x7C7C7C7Cu;
            uint32_t normal_ff  = __vcmpne4(exp_bits, 0u);        // 0xFF per normal byte, 0 otherwise
            uint32_t normal_01  = normal_ff & 0x01010101u;         // 1 per normal byte
            uint32_t inc_01     = b & normal_01;                   // extract rounding increment
            uint32_t upper_orig = a - inc_01;                      // undo increment
            uint32_t lower_orig = b & ~normal_01;                  // strip inc bit from lower
            dst_u32[2 * i]     = __byte_perm(upper_orig, lower_orig, 0x1504u);
            dst_u32[2 * i + 1] = __byte_perm(upper_orig, lower_orig, 0x3726u);
          }
        } else if constexpr (kFastE4M3) {
          // Single-weight E4M3: FastNumericArrayConverter formula.
          // FP16 = sign<<8 | em<<7  (em = raw[6:0] = FP16 bits[13:7]).
          // This matches cutlass::FastNumericArrayConverter<half_t, float_e4m3_t>.
          uint32_t lo = __byte_perm(a, 0u, 0x4140u);  // {0, a[1], 0, a[0]}
          uint32_t hi = __byte_perm(a, 0u, 0x4342u);  // {0, a[3], 0, a[2]}
          dst_u32[2 * i]     = (lo & 0x00800080u) << 8 | (lo & 0x007F007Fu) << 7;
          dst_u32[2 * i + 1] = (hi & 0x00800080u) << 8 | (hi & 0x007F007Fu) << 7;
        } else {
          // E4M3 reconstruction.
          // stored upper packs: sign(1) | (exp(4)+mant_hi(3)) into 7 bits with bias adjust.
          uint32_t s         = a & 0x80808080u;          // keep sign bit lanes
          uint32_t sub       = (b & 0x80808080u) >> 7;  // bias adjust from mantissa MSBs
          uint32_t a_sub     = a - sub;                  // undo bias adjust
          uint32_t packed_upper = ((a_sub >> 1) & 0x3f3f3f3fu) | s;
          dst_u32[2 * i]     = __byte_perm(packed_upper, b, 0x1504u);
          dst_u32[2 * i + 1] = __byte_perm(packed_upper, b, 0x3726u);
        }
      }
  }

  /// Transform operand A into MMA layout and reconstruct/convert dual-weight B in one step.
  CUTLASS_DEVICE
  void transform_dual_weight_operands(
      typename PipeState::WarpTransformedFragmentA &dst_A,
      typename PipeState::WarpTransformedFragmentB &dst_B,
      typename PipeState::WarpLoadedFragmentA const &src_A,
      typename PipeState::WarpLoadedFragmentB const &src_B_upper,
      typename PipeState::WarpLoadedFragmentB const &src_B_lower) const {

    using ElementA = typename Operator::ElementA;
    using ElementAMma = typename Operator::ArchMmaOperator::ElementA;
    using MmaIterations = typename Operator::MmaIterations;
    using MmaOperandA = typename Operator::ArchMmaOperator::FragmentA;
    using WarpFragmentA = typename PipeState::WarpLoadedFragmentA;

    // Reconstruct FP16 B from FP8 upper/lower in the MMA-friendly layout
    reconstruct_fp16(dst_B, src_B_upper, src_B_lower);

    // Transform A to MMA operand layout (mirrors warp_mma_.transform)
    cutlass::gemm::warp::detail::FragmentShuffler<ElementAMma, ElementA, MmaIterations::kRow,
             WarpFragmentA::kElements, MmaOperandA::kElements, cutlass::gemm::Operand::kA> shuffler_A;
    cutlass::gemm::warp::detail::FragmentConverter<ElementAMma, ElementA, WarpFragmentA::kElements> convert_A;

    WarpFragmentA tmp_A = shuffler_A(src_A);
    dst_A = convert_A(tmp_A);
  }

  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B_upper,          ///< [in|out] iterator over B operand in global memory
    IteratorB &iterator_B_lower,          ///< [in|out] iterator over B operand in global memory
    int &gemm_k_iterations)         ///< [in|out] number of threadblock mainloop iterations remaining
  {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragments from shared memory (Upper and Lower)
      warp_tile_iterator_B_upper_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      warp_tile_iterator_B_upper_.load(
          pipe_state.warp_loaded_frag_B_upper_[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_B_upper_;

      warp_tile_iterator_B_lower_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      warp_tile_iterator_B_lower_.load(
          pipe_state.warp_loaded_frag_B_lower_[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_B_lower_;

      // Except for the first warp-tile, all warp-tiles convert their incoming
      // shared memory fragments as necessary
      if (warp_mma_k > 0) {
        transform_dual_weight_operands(
            pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
            pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
            pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
            pipe_state.warp_loaded_frag_B_upper_[warp_mma_k % 2],
            pipe_state.warp_loaded_frag_B_lower_[warp_mma_k % 2]);
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(pipe_state.tmp_accum_,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  pipe_state.tmp_accum_);

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(accum, pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2], accum);
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(iterator_A, iterator_B_upper, iterator_B_lower,
                               group_start_iteration_A,
                               group_start_iteration_B);
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == kWarpGemmIterations) {

        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(iterator_A, iterator_B_upper, iterator_B_lower,
                               group_start_iteration_A,
                               group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B_upper, iterator_B_lower);
        advance_smem_read_stage();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B_upper.clear_mask(gemm_k_iterations == 0);
        iterator_B_lower.clear_mask(gemm_k_iterations == 0);
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == kWarpGemmIterations) {
        transform_dual_weight_operands(
            pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_B_upper_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_B_lower_[(warp_mma_k + 1) % 2]);
      }
    }
  }


  /// K-block interleaved variant of mac_loop_iter.
  ///
  /// Issues MMA(k) BEFORE transform(k+1) so that the FP8->FP16 reconstruction
  /// integer ops (__vsub4 / __byte_perm) can overlap with the tensor-core
  /// pipeline executing MMA(k).  Functionally equivalent to mac_loop_iter but
  /// exposes more instruction-level parallelism between INT and TC units.
  CUTLASS_DEVICE
  void mac_loop_iter_kblock_interleaved(
    PipeState &pipe_state,
    FragmentC &accum,
    IteratorA &iterator_A,
    IteratorB &iterator_B_upper,
    IteratorB &iterator_B_lower,
    int &gemm_k_iterations)
  {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; ++warp_mma_k) {

      // 1. Load warp-tile (k+1) fragments from shared memory into registers.
      this->warp_tile_iterator_A_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      warp_tile_iterator_B_upper_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      warp_tile_iterator_B_upper_.load(
          pipe_state.warp_loaded_frag_B_upper_[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_B_upper_;

      warp_tile_iterator_B_lower_.set_kgroup_index(
          (warp_mma_k + 1) % kWarpGemmIterations);
      warp_tile_iterator_B_lower_.load(
          pipe_state.warp_loaded_frag_B_lower_[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_B_lower_;

      // 2. Execute MMA for warp-tile k using pre-transformed fragments.
      //    These fragments were transformed either in gemm_iters (k=0 of the
      //    very first call) or in step 3 of the previous iteration.
      if (Detail::kStagedAccumulation) {
        warp_mma_(pipe_state.tmp_accum_,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  pipe_state.tmp_accum_);

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(accum,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  accum);
      }

      // 3. Transform warp-tile (k+1) from FP8 upper/lower to FP16.
      //    These integer ops (vsub4, byte_perm) execute on ALU units while
      //    the tensor-core pipeline processes MMA(k), hiding their latency.
      transform_dual_weight_operands(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_upper_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_lower_[(warp_mma_k + 1) % 2]);

      // 4. Issue global->shared copies, fence, and stage advance (same as
      //    mac_loop_iter — this section is unchanged).
      if (warp_mma_k < kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(iterator_A, iterator_B_upper, iterator_B_lower,
                               group_start_iteration_A,
                               group_start_iteration_B);
      }

      if (warp_mma_k + 2 == kWarpGemmIterations) {

        int group_start_iteration_A =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(iterator_A, iterator_B_upper, iterator_B_lower,
                               group_start_iteration_A,
                               group_start_iteration_B);

        cutlass::arch::cp_async_fence();
        gmem_wait();
        advance_smem_write_stage(iterator_A, iterator_B_upper, iterator_B_lower);
        advance_smem_read_stage();

        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B_upper.clear_mask(gemm_k_iterations == 0);
        iterator_B_lower.clear_mask(gemm_k_iterations == 0);
      }

      // Note: No separate end-of-loop pre-transform needed.  Step 3 above
      // transforms index (warp_mma_k+1) % kWarpGemmIterations each iteration,
      // so at warp_mma_k == kWarpGemmIterations-1 it naturally produces the
      // pre-transformed k=0 fragment for the next mac_loop_iter call.
    }
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      IteratorB &iterator_B_upper,        ///< [in|out] iterator over B operand in global memory
      IteratorB &iterator_B_lower)        ///< [in|out] iterator over B operand in global memory
  {
    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B_upper.clear_mask(gemm_k_iterations == 0);
    iterator_B_lower.clear_mask(gemm_k_iterations == 0);

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragments from shared memory
    this->warp_tile_iterator_B_upper_.set_kgroup_index(0);
    this->warp_tile_iterator_B_upper_.load(pipe_state.warp_loaded_frag_B_upper_[0]);
    ++this->warp_tile_iterator_B_upper_;

    this->warp_tile_iterator_B_lower_.set_kgroup_index(0);
    this->warp_tile_iterator_B_lower_.load(pipe_state.warp_loaded_frag_B_lower_[0]);
    ++this->warp_tile_iterator_B_lower_;

    // Transform, if necessary, the first warp-tile's shared memory fragments
    transform_dual_weight_operands(
        pipe_state.warp_transformed_frag_A_[0],
        pipe_state.warp_transformed_frag_B_[0],
        pipe_state.warp_loaded_frag_A_[0],
        pipe_state.warp_loaded_frag_B_upper_[0],
        pipe_state.warp_loaded_frag_B_lower_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    // Mainloop — select between standard and k-block interleaved bodies.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-kStages + 1);) {
      if constexpr (kKBlockInterleaved) {
        mac_loop_iter_kblock_interleaved(
          pipe_state,
          accum,
          iterator_A,
          iterator_B_upper,
          iterator_B_lower,
          gemm_k_iterations);
      } else {
        mac_loop_iter(
          pipe_state,
          accum,
          iterator_A,
          iterator_B_upper,
          iterator_B_lower,
          gemm_k_iterations);
      }
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_upper_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_lower_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_upper_;
      ++this->warp_tile_iterator_B_lower_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_upper_.add_tile_offset({(-2 * kStageIters), 0});
      this->warp_tile_iterator_B_lower_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_upper_.add_tile_offset({((kStages - 2) * kStageIters), 0});
      this->warp_tile_iterator_B_lower_.add_tile_offset({((kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B_upper,
      ///< iterator over B operand in global memory
      IteratorB iterator_B_lower,
      ///< initial value of accumulator
      FragmentC const &src_accum) {

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_B_upper, iterator_B_lower, gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // Initialize destination accumulators with source accumulators
    accum = src_accum;

    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B_upper, iterator_B_lower);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
