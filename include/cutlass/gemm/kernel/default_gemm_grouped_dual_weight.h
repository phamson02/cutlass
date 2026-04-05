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
    \brief Default grouped GEMM wiring for dual-weight kernels (FP8 upper/lower -> FP16).
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/threadblock/default_mma_dual_weight.h"
#include "cutlass_extensions/gemm/kernel/moe_cutlass_kernel_dual_weight.h"

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Default GEMM kernel instantiation for the dual-weight mainloop.
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
  /// Element type for C and D matrix operands
  typename ElementC,
  /// Layout type for C and D matrix operands (only RowMajor supported here)
  typename LayoutC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// If true, kernel is configured to support serial reduction in the
  /// epilogue
  bool SplitKSerial,
  /// Operation performed by GEMM
  typename Operator,
  /// Use zfill or predicate for out-of-bound cp.async
  SharedMemoryClearOption SharedMemoryClear,
  /// Gather operand A by using an index array
  bool GatherA,
  /// Gather operand B by using an index array
  bool GatherB,
  /// Scatter operand D by using an index array
  bool ScatterD,
  /// Permute result D
  typename PermuteDLayout,
  /// Permute operand A
  typename PermuteALayout = layout::NoPermute,
  /// Permute operand B
  typename PermuteBLayout = layout::NoPermute,
  /// When true, use k-block interleaved FP8->FP16 reconstruction in mainloop.
  bool kKBlockInterleaved = false,
  /// When true and ElementB is float_e5m2_t, use truncation reconstruction.
  bool kTruncateE5M2 = false,
  /// When true and ElementB is float_e4m3_t, use FastNumericArrayConverter reconstruction.
  bool kFastE4M3 = false
>
struct DefaultGemmDualWeight {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value,
                "Dual-weight default GEMM only supports RowMajor C/D layout.");

  /// Threadblock MMA
  using Mma = typename cutlass::gemm::threadblock::DefaultMmaDualWeight<
      ElementA, LayoutA, kAlignmentA,
      ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC,
      arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, Stages, Operator,
      false, SharedMemoryClear, GatherA, GatherB, kKBlockInterleaved, kTruncateE5M2, kFastE4M3>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ScatterD,
    PermuteDLayout
  >::Epilogue;

  /// Kernel
  using GemmKernel = kernel::Gemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Default grouped GEMM kernel instantiation for the dual-weight mainloop.
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Whether the schedule of problems to visit has been precomputed
    GroupScheduleMode GroupScheduleMode_,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// When true and ElementB is float_e5m2_t, use truncation reconstruction.
    bool kTruncateE5M2 = false
    >
struct DefaultGemmGroupedDualWeight;

template <
    typename ElementA,
    typename LayoutA,
    int kAlignmentA,
    typename ElementB,
    typename LayoutB,
    int kAlignmentB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    GroupScheduleMode GroupScheduleMode_,
    typename Operator,
    SharedMemoryClearOption SharedMemoryClear,
    typename PermuteDLayout,
    bool kTruncateE5M2>
struct DefaultGemmGroupedDualWeight<
    ElementA,
    LayoutA,
    ComplexTransform::kNone,   // transform A
    kAlignmentA,
    ElementB,
    LayoutB,
    ComplexTransform::kNone,   // transform B
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    arch::OpClassTensorOp,
    arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    GroupScheduleMode_,
    Operator,
    SharedMemoryClear,
    PermuteDLayout,
    kTruncateE5M2> {

  /// Define the default GEMM kernel
  using DefaultGemmKernel = typename kernel::DefaultGemmDualWeight<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    true,
    Operator,
    SharedMemoryClear,
    false,
    false,
    false,
    PermuteDLayout,
    layout::NoPermute,
    layout::NoPermute,
    false,             // kKBlockInterleaved
    kTruncateE5M2      // pass through to mainloop
  >::GemmKernel;

  using GemmKernel = kernel::MoeFCGemmDualWeight<
    typename DefaultGemmKernel::Mma,
    typename DefaultGemmKernel::Epilogue,
    ThreadblockSwizzle,
    arch::Sm80,
    GroupScheduleMode_>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
