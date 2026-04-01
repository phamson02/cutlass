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
    \brief Default MMA wiring for dual-weight (FP8 upper/lower -> FP16) mainloop.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/mma_multistage_dual_weight.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/layout/tensor_op_multiplicand_sm80.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

namespace cutlass {
namespace arch {

/// Tag indicating the dual weight operation (FP16 A, FP8 B, FP16 Accum)
struct OpMultiplyAddDualWeight {};

} // namespace arch
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator_ = typename platform::enable_if<
        !cutlass::is_complex<ElementAccumulator_>::value,
        typename device::DefaultGemmConfiguration<
            OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementAccumulator_,
            ElementAccumulator_>::Operator>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// When true, issue MMA(k) before transform(k+1) to expose ILP between
    /// ALU (FP8->FP16 reconstruction) and tensor-core units.
    bool kKBlockInterleaved = false,
    /// When true and ElementB is float_e5m2_t, use truncation reconstruction.
    bool kTruncateE5M2 = false,
    /// When true and ElementB is float_e4m3_t, use FastNumericArrayConverter reconstruction.
    bool kFastE4M3 = false
    >
struct DefaultMmaDualWeight;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Specialization for SM80 tensor cores and row-major C
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename GemmOperator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// K-block interleaved reconstruction
    bool kKBlockInterleaved,
    /// When true and ElementB is float_e5m2_t, use truncation reconstruction.
    bool kTruncateE5M2,
    /// When true and ElementB is float_e4m3_t, use FastNumericArrayConverter reconstruction.
    bool kFastE4M3
    >
struct DefaultMmaDualWeight<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
    ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, Stages, GemmOperator,
    AccumulatorsInRowMajor, SharedMemoryClear, GatherA, GatherB, kKBlockInterleaved,
    kTruncateE5M2, kFastE4M3> {

  using Shape = ThreadblockShape;
  using WarpShape_ = WarpShape;
  using InstructionShape_ = InstructionShape;

  static int const kStages = Stages;
  static int const kAccessSizeInBits = 128;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Global;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = GemmShape<Shape::kM / WarpShape_::kM,
                              Shape::kN / WarpShape_::kN,
                              Shape::kK / WarpShape_::kK>;

  /// Number of threads per warp
  static int const kWarpSize = 32;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //
  using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementA>::value, Shape::kK>;

  using SmemLayoutB = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementB>::value, Shape::kK>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kM>, kThreads,
      layout::PitchLinearShape<8, 4>,
      kAccessSizeInBits / sizeof_bits<ElementA>::value>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 1,
      IteratorThreadMapA>;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<Shape::kK, Shape::kN>, kThreads,
      layout::PitchLinearShape<4, 8>,
      kAccessSizeInBits / sizeof_bits<ElementB>::value>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 0,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  using SmemLayoutB_MmaTensorOp =
      layout::ColumnMajorTensorOpMultiplicandCrosswise<
          sizeof_bits<cutlass::half_t>::value, WarpShape_::kK>;

  using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape_, InstructionShape_, ElementA, SmemLayoutA, cutlass::half_t, SmemLayoutB_MmaTensorOp,
      ElementAccumulator, LayoutC, cutlass::arch::OpMultiplyAdd, WarpCount::kK>::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<MmaTensorOp, MatrixShape<0, 0>,
                              MatrixShape<0, 0>, WarpCount::kK>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, GatherA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, GatherB>;

  // Warp Iterator B (FP8)
  using WarpIteratorB = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
      cutlass::MatrixShape<WarpShape_::kK, WarpShape_::kN>, cutlass::gemm::Operand::kB,
      ElementB, SmemLayoutB,
      cutlass::MatrixShape<InstructionShape_::kK, InstructionShape_::kN>,
      MmaTensorOp::Policy::OpDelta::kRow,
      32,
      MmaPolicy::kPartitionsK>;

  struct Policy {
      using Operator = MmaTensorOp; // FP16 Operator
      using WarpIteratorB = WarpIteratorB; // FP8 Warp Iterator
      using SmemPaddingA = typename MmaPolicy::SmemPaddingA;
      using SmemPaddingB = typename MmaPolicy::SmemPaddingB;
      static int const kPartitionsK = MmaPolicy::kPartitionsK;
  };

  // Define the Mma class
  using Mma = MmaMultistageDualWeight<
      Shape,
      IteratorA,
      SmemIteratorA,
      kCacheOpA,
      IteratorB,
      SmemIteratorB,
      kCacheOpB,
      ElementAccumulator,
      LayoutC,
      Policy,
      Stages,
      SharedMemoryClear,
      kKBlockInterleaved,
      kTruncateE5M2,
      kFastE4M3>;

  using ThreadblockMma = Mma;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
