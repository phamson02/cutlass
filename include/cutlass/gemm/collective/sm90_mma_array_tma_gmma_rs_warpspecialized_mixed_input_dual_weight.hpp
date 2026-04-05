/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/cuda_host_adapter.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

// Helper to dispatch to the correct NestedFP reconstruction based on kernel schedule.
// Default: E4M3 RTN reconstruction (original transform2).
enum class DualWeightReconstructionKind { E4M3_RTN, E5M2_RTN, E5M2_TRUNC };

template <class KernelSchedule>
struct DualWeightReconstructionTrait {
  static constexpr DualWeightReconstructionKind kind =
      cute::is_any_of_v<KernelSchedule,
          KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2Trunc,
          KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2TruncCustom>
        ? DualWeightReconstructionKind::E5M2_TRUNC
        : cute::is_any_of_v<KernelSchedule,
              KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2,
              KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2Custom>
          ? DualWeightReconstructionKind::E5M2_RTN
          : DualWeightReconstructionKind::E4M3_RTN;
};

template <DualWeightReconstructionKind Kind, class T1, class T2, class T3>
CUTE_DEVICE void dual_weight_reconstruct(T1 const& a2, T2 const& a3, T3&& out) {
  if constexpr (Kind == DualWeightReconstructionKind::E5M2_TRUNC) {
    cute::transform2_e5m2_trunc(a2, a3, out);
  } else if constexpr (Kind == DualWeightReconstructionKind::E5M2_RTN) {
    cute::transform2_e5m2_rtn(a2, a3, out);
  } else {
    cute::transform2(a2, a3, out);
  }
}
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <class PointerType>
static constexpr CUTLASS_HOST_DEVICE auto
dual_weight_get_logical_ptr(PointerType const* ptr) {
  return cute::recast_ptr<PointerType const>(ptr);
}

template <int Stages, class LayoutAtom, class TileShape, class Stride>
static constexpr CUTLASS_HOST_DEVICE auto
dual_weight_get_smem_layout(LayoutAtom layout_atom, TileShape const& tile_shape, Stride const& stride) {
  if constexpr (not cute::is_layout<Stride>::value) {
    return tile_to_shape(
      layout_atom,
      append(tile_shape, Int<Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0, Stride>(), Step<_2, _1, _3>, Step<_1, _2, _3>>{});
  } else {
    auto gmem_tile = composition(stride, tile_shape);
    return make_layout_like(append(gmem_tile, make_layout(Int<Stages>{}, 0)));
  }
}

template <class Shape, class Stride>
static constexpr CUTLASS_HOST_DEVICE auto
dual_weight_get_gmem_layout(Shape const& shape, Stride const& stride) {
  if constexpr (not cute::is_layout<Stride>::value) {
    return make_layout(shape, stride);
  } else {
    return stride;
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
// NestedFP dual-weight port: 3 extra template params for the second A operand (A2/A3)
template <
  int Stages,
  class ClusterShape,
  class KernelSchedule_,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class GmemTiledCopyA2_,
  class SmemLayoutAtomA2_,
  class SmemCopyAtomA2_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputDualWeight<Stages, ClusterShape, KernelSchedule_>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    GmemTiledCopyA2_,
    SmemLayoutAtomA2_,
    SmemCopyAtomA2_>
{
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInputDualWeight<Stages, ClusterShape, KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

  using ElementA = ElementA_;
  using ElementB = ElementB_;
  // Dual-weight path supports {E4M3 or E5M2} x FP16 in either operand order.
  // Infer transformed-side from element ordering (builder guarantees plain element types).
  static constexpr bool IsATransformed =
      cute::is_same_v<ElementA, cutlass::float_e4m3_t> ||
      cute::is_same_v<ElementA, cutlass::float_e5m2_t>;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  static_assert(( IsATransformed && (cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) ||
                (!IsATransformed && (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
                "The transformed type must be K-major.");

  static_assert(( IsATransformed && (sizeof(ElementB) == 2)) ||
                (!IsATransformed && (sizeof(ElementA) == 2)) ||
                ((cutlass::gemm::detail::is_k_major<StrideA>() || is_layout<StrideA>::value || is_layout<InternalStrideA>::value) &&
                 (cutlass::gemm::detail::is_k_major<StrideB>() || is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
                "The unscaled element must be 2 bytes OR both inputs must be K-major");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  // NestedFP dual-weight: extra template params for the A2/A3 operand path
  using GmemTiledCopyA2 = GmemTiledCopyA2_;
  using SmemLayoutAtomA2 = SmemLayoutAtomA2_;
  using SmemCopyAtomA2 = SmemCopyAtomA2_;

  // Route the transformed operand through RF so it can be reconstructed before GMMA.
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA = cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB = cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA = cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB = cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA   = cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB   = cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA = cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB = cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutB = decltype(detail::dual_weight_get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomB{}, select<1,2>(TileShape{}), InternalSwappedStrideB{}));
  // NestedFP dual-weight: SmemLayoutA2 for both upper (A2) and lower (A3) weight halves
  using SmemLayoutA2 = decltype(detail::dual_weight_get_smem_layout<DispatchPolicy::Stages>(
      SmemLayoutAtomA2{}, select<0,2>(TileShape{}), InternalSwappedStrideA{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyA2, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA2, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified for A2.");

public:
  static constexpr bool IsDualWeightKernelSchedule = cute::is_any_of_v<
      KernelSchedule,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeDualWeight,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongDualWeight,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeDualWeightE5M2Trunc>;
  static_assert((cute::is_same_v<RealSwappedElementA, cutlass::float_e4m3_t> ||
                 cute::is_same_v<RealSwappedElementA, cutlass::float_e5m2_t>) &&
                    cute::is_same_v<RealSwappedElementB, cutlass::half_t>,
                "Dual-weight RS collective requires reconstructed FP8 (E4M3 or E5M2) x FP16 input pair.");
  static constexpr DualWeightReconstructionKind kReconstructionKind =
      DualWeightReconstructionTrait<KernelSchedule>::kind;
  // Verify e5m2 weight types use e5m2 reconstruction, not e4m3
  static_assert(!(cute::is_same_v<RealSwappedElementA, cutlass::float_e5m2_t> &&
                  kReconstructionKind == DualWeightReconstructionKind::E4M3_RTN),
                "E5M2 weights must use E5M2 reconstruction, not E4M3 — check KernelSchedule type.");
  static constexpr size_t SmemAlignmentA2 = cutlass::detail::alignment_for_swizzle(SmemLayoutA2{});
  static constexpr size_t SmemAlignmentB = cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  // Alias for builder compatibility (CollectiveBuilder checks SmemAlignmentA)
  static constexpr size_t SmemAlignmentA = SmemAlignmentA2;

  static_assert(SmemAlignmentA2 >= 128 and SmemAlignmentB >= 128, "Require at least 128B alignment");

  // NestedFP dual-weight: SharedStorage has smem_B + smem_A2 (upper) + smem_A3 (lower)
  // No smem_A — both weight halves use SmemLayoutA2 with float_e4m3_t element type
  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentB, SmemAlignmentA2), _0> {
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>, SmemAlignmentB> smem_B;
      cute::array_aligned<cutlass::float_e4m3_t, cute::cosize_v<SmemLayoutA2>, SmemAlignmentA2> smem_A2;
      cute::array_aligned<cutlass::float_e4m3_t, cute::cosize_v<SmemLayoutA2>, SmemAlignmentA2> smem_A3;
    } tensors;

    struct TensorMapStorage {
      cute::TmaDescriptor smem_tensormap_A2;
      cute::TmaDescriptor smem_tensormap_B;
      cute::TmaDescriptor smem_tensormap_A3;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  // Uses original field names for launcher compatibility.
  // SwapAB is handled in to_underlying_arguments: ptr_A/ptr_B are in the caller's (builder's) convention.
  // ptr_B_lower always holds the second weight half.
  struct Arguments {
    ElementA const** ptr_A = nullptr;
    StrideA dA{};
    ElementB const** ptr_B = nullptr;
    StrideB dB{};
    ElementB const** ptr_B_lower = nullptr;
  };

  // Device side kernel params
  struct Params {
    // NestedFP dual-weight: A2 layout uses SmemLayoutA2 for upper/lower e4m3 halves
    using LayoutA2 = decltype(detail::dual_weight_get_gmem_layout(
        repeat_like(InternalSwappedStrideA{}, int32_t(0)), InternalSwappedStrideA{}));
    using LayoutB = decltype(detail::dual_weight_get_gmem_layout(
        repeat_like(InternalSwappedStrideB{}, int32_t(0)), InternalSwappedStrideB{}));

    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A2 = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA2{},
        make_tensor(static_cast<cutlass::float_e4m3_t const*>(nullptr),
                    detail::dual_weight_get_gmem_layout(repeat_like(InternalSwappedStrideA{}, int32_t(0)), InternalSwappedStrideA{})),
        SmemLayoutA2{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    using TMA_A3 = TMA_A2;
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::dual_weight_get_logical_ptr(static_cast<SwappedElementB const*>(nullptr)), LayoutB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}))); // mcast along M mode for this N load, if any

    TMA_B tma_load_b;
    TMA_A2 tma_load_a2;
    TMA_A3 tma_load_a3;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t tma_transaction_bytes_mk2 = TmaTransactionBytesMK2;
    uint32_t tma_transaction_bytes_mk3 = TmaTransactionBytesMK3;
    void* tensormaps;
    cutlass::float_e4m3_t const** ptr_A2;
    cutlass::float_e4m3_t const** ptr_A3;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape problem_shapes,
      Arguments const& args,
      void* workspace) {

    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create tensormap/tma desc.
    // These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t mock_L = 1;

    // Determine mock base pointers for TMA descriptor creation.
    // Weight halves (A2/upper, A3/lower) come from the caller's A or B slot
    // depending on SwapAB; the activation (ptr_B_first_batch) from the other slot.
    cutlass::float_e4m3_t const* ptr_A2_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    if constexpr (not SwapAB) {
      // No swap: caller's A slot has narrow weight, B slot has wide activation
      ptr_A2_first_batch = reinterpret_cast<cutlass::float_e4m3_t const*>(
          reinterpret_cast<uint64_t>(args.ptr_A) & 0xFFFFFFFFFFFFFFF0);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(
          reinterpret_cast<uint64_t>(args.ptr_B) & 0xFFFFFFFFFFFFFFF0);
    } else {
      // Swap: caller's B slot has narrow weight, A slot has wide activation
      ptr_A2_first_batch = reinterpret_cast<cutlass::float_e4m3_t const*>(
          reinterpret_cast<uint64_t>(args.ptr_B) & 0xFFFFFFFFFFFFFFF0);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(
          reinterpret_cast<uint64_t>(args.ptr_A) & 0xFFFFFFFFFFFFFFF0);
    }
    // ptr_B_lower always holds the second weight half (no SwapAB)
    cutlass::float_e4m3_t const* ptr_A3_first_batch =
        reinterpret_cast<cutlass::float_e4m3_t const*>(
            reinterpret_cast<uint64_t>(args.ptr_B_lower) & 0xFFFFFFFFFFFFFFF0);

    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      }
      else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(
          transform_leaf(dA.shape(), [](auto x){
            if constexpr (not is_static_v<decltype(x)>) {
              return static_cast<decltype(x)>(1);
            } else {
              return x;
            }
          }),
          dA.stride());
      }
      dB = InternalSwappedStrideB{};
    }
    else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);
      if constexpr (SwapAB) {
        init_M = get<1>(problem_shape_MNK);
        init_N = get<0>(problem_shape_MNK);
      }

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      }
      else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }

    // NestedFP dual-weight: A2/A3 TMA descriptors use GmemTiledCopyA2 + SmemLayoutA2
    Tensor tensor_a2 = make_tensor(ptr_A2_first_batch,
        detail::dual_weight_get_gmem_layout(make_shape(init_M, init_K, mock_L), dA));
    Tensor tensor_a3 = make_tensor(ptr_A3_first_batch,
        detail::dual_weight_get_gmem_layout(make_shape(init_M, init_K, mock_L), dA));
    Tensor tensor_b = make_tensor(ptr_B_first_batch,
        detail::dual_weight_get_gmem_layout(make_shape(init_N, init_K, mock_L), dB));

    typename Params::TMA_A2 tma_load_a2 = make_tma_copy_A_sm90(
        GmemTiledCopyA2{},
        tensor_a2,
        SmemLayoutA2{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_A3 tma_load_a3 = make_tma_copy_A_sm90(
        GmemTiledCopyA2{},
        tensor_a3,
        SmemLayoutA2{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{})); // mcast along M mode for this N load, if any

    void* tensormaps = workspace;
    // args_setup: swap ptr_A/ptr_B based on SwapAB to produce internal pointers
    auto args_setup = [&](auto ptr_weight_upper, auto ptr_activation) -> Params {
      return {
          tma_load_b,
          tma_load_a2,
          tma_load_a3,
          TmaTransactionBytes,
          TmaTransactionBytesNK,
          TmaTransactionBytesMK2,
          TmaTransactionBytesMK3,
          tensormaps,
          reinterpret_cast<cutlass::float_e4m3_t const**>(ptr_weight_upper),  // ptr_A2 (upper weight)
          reinterpret_cast<cutlass::float_e4m3_t const**>(args.ptr_B_lower),  // ptr_A3 (lower weight, always in ptr_B_lower)
          ptr_dA,
          reinterpret_cast<SwappedElementB const**>(ptr_activation),           // ptr_B (activation)
          ptr_dB,
          dA,
          dB
      };
    };
    return SwapAB ? args_setup(args.ptr_B, args.ptr_A)
                  : args_setup(args.ptr_A, args.ptr_B);
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    static_cast<void>(problem_shape);
    static_cast<void>(args);
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);

    // Calculating workspace size
    auto calculate_workspace_size = [SizeOfCuTensorMap, sm_count](uint32_t num_input_tensors) {
        return num_input_tensors * SizeOfCuTensorMap * sm_count;
    };
    // Allocate gmem space for per-SM copies of A/B/A-lower tensormaps.
    return calculate_workspace_size(3);
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    static_cast<void>(problem_shape);
    static_cast<void>(args);
    static_cast<void>(workspace);
    static_cast<void>(stream);
    static_cast<void>(cuda_adapter);
    return cutlass::Status::kSuccess;
  }


  template<class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      ProblemShape problem_shapes,
      Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M,N,K,L] = problem_shape_MNKL;
        auto get_stride = [](auto stride) {
          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {
            return *stride;
          }
          else {
            return stride;
          }
        };
        auto dA = get_stride(args.dA);
        auto dB = get_stride(args.dB);
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
            detail::dual_weight_get_gmem_layout(cute::make_shape(M, K, L), dA));
        implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
            detail::dual_weight_get_gmem_layout(cute::make_shape(N, K, L), dB));
        implementable = implementable && (args.ptr_B_lower != nullptr);
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr uint32_t TmaTransactionBytesNK =
      cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
                             static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementB>));
  // NestedFP dual-weight: A2 (upper) and A3 (lower) each use SmemLayoutA2 with float_e4m3_t
  static constexpr uint32_t TmaTransactionBytesMK2 =
      cutlass::bits_to_bytes(size<0>(SmemLayoutA2{}) * size<1>(SmemLayoutA2{}) *
                             static_cast<uint32_t>(sizeof_bits<cutlass::float_e4m3_t>::value));
  static constexpr uint32_t TmaTransactionBytesMK3 =
      cutlass::bits_to_bytes(size<0>(SmemLayoutA2{}) * size<1>(SmemLayoutA2{}) *
                             static_cast<uint32_t>(sizeof_bits<cutlass::float_e4m3_t>::value));
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesNK + TmaTransactionBytesMK2 + TmaTransactionBytesMK3;

  // NestedFP dual-weight: reconstruction is now done inline via cute::transform2 in the mma() method

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  // The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    static_cast<void>(L);
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(
        shape(detail::dual_weight_get_gmem_layout(make_shape(N, K, mock_L), mainloop_params.dB))); // (n,k,l)
    Tensor mA2_mkl = mainloop_params.tma_load_a2.get_tma_tensor(
        shape(detail::dual_weight_get_gmem_layout(make_shape(M, K, mock_L), mainloop_params.dA))); // (m,k,l)
    Tensor mA3_mkl = mainloop_params.tma_load_a3.get_tma_tensor(
        shape(detail::dual_weight_get_gmem_layout(make_shape(M, K, mock_L), mainloop_params.dA))); // (m,k,l)

    // Make tiled views, defer the slice
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});  // (BLK_N,BLK_K,n,k,l)
    Tensor gA2_mkl = local_tile(mA2_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{}); // (BLK_M,BLK_K,m,k,l)
    Tensor gA3_mkl = local_tile(mA3_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{}); // (BLK_M,BLK_K,m,k,l)

    return cute::make_tuple(gB_nkl, gA2_mkl, gA3_mkl);
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <
    class... Ts,
    class... TMs,
    class KTileIterator, class BlockCoord
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<Ts...> const& load_inputs,
      cute::tuple<TMs...> const& input_tensormaps,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    static_cast<void>(thread_idx);

    static_assert(sizeof... (Ts) == 3, "NestedFP dual-weight needs three load inputs (B, A2, A3)");
    static_assert(sizeof... (TMs) == 3, "NestedFP dual-weight needs three tensormaps (A2, B, A3)");

    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});           // (BLK_N,BLK_K,PIPE)
    Tensor sB  = as_position_independent_swizzle_tensor(sB_);                                       // (BLK_N,BLK_K,PIPE)
    Tensor sA2_ = make_tensor(make_smem_ptr(shared_tensors.smem_A2.data()), SmemLayoutA2{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA2  = as_position_independent_swizzle_tensor(sA2_);                                     // (BLK_M,BLK_K,PIPE)
    Tensor sA3_ = make_tensor(make_smem_ptr(shared_tensors.smem_A3.data()), SmemLayoutA2{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA3  = as_position_independent_swizzle_tensor(sA3_);                                     // (BLK_M,BLK_K,PIPE)

    //
    // Prepare the TMA loads for B, A2, A3
    //

    constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

    Tensor gB_nkl = get<0>(load_inputs);
    Tensor gA2_mkl = get<1>(load_inputs);
    Tensor gA3_mkl = get<2>(load_inputs);

    auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);
    auto block_tma_a2 = mainloop_params.tma_load_a2.get_slice(cluster_local_block_id.y);
    auto block_tma_a3 = mainloop_params.tma_load_a3.get_slice(cluster_local_block_id.y);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    static_cast<void>(k_coord);
    Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)
    Tensor gA2 = gA2_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)
    Tensor gA3 = gA3_mkl(_,_,m_coord,_,l_coord);                                                   // (BLK_M,BLK_K,k)

    // Applies the mapping from block_tma
    Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)
    Tensor tAgA2 = block_tma_a2.partition_S(gA2);                                              // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA2 = block_tma_a2.partition_D(sA2);                                           // (TMA,TMA_M,TMA_K,PIPE)
    Tensor tAgA3 = block_tma_a3.partition_S(gA3);                                              // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA3 = block_tma_a3.partition_D(sA3);                                           // (TMA,TMA_M,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA2, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
      }
    }

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        // TMA loads: B, A2, A3 using per-SM tensormaps
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps), *tma_barrier, mcast_mask_b),
             tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        copy(mainloop_params.tma_load_a2.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_a),
             tAgA2(_,_,_,*k_tile_iter), tAsA2(_,_,_,write_stage));
        copy(mainloop_params.tma_load_a3.with(get<2>(input_tensormaps), *tma_barrier, mcast_mask_a),
             tAgA3(_,_,_,*k_tile_iter), tAsA3(_,_,_,write_stage));
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class FrgTensorC
  >
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params) {
    static_cast<void>(mainloop_params);

    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA2{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutAtomA2{}) == 2, "SmemLayoutAtomA2 must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2, "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SmemCopyAtomA2>,
      "SM90 GMMA mainloops must specify a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    // Obtain warp index
    [[maybe_unused]] int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sB  = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});           // (BLK_N,BLK_K,PIPE)
    Tensor sA2_ = make_tensor(make_smem_ptr(shared_tensors.smem_A2.data()), SmemLayoutA2{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA2  = as_position_independent_swizzle_tensor(sA2_);                                     // (BLK_M,BLK_K,PIPE)
    Tensor sA3_ = make_tensor(make_smem_ptr(shared_tensors.smem_A3.data()), SmemLayoutA2{});        // (BLK_M,BLK_K,PIPE)
    Tensor sA3  = as_position_independent_swizzle_tensor(sA3_);                                     // (BLK_M,BLK_K,PIPE)


    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    auto mma_warpgroup_slice = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCsB = mma_warpgroup_slice.partition_B(sB);                                        // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB = mma_warpgroup_slice.make_fragment_B(tCsB);                                  // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCsA2 = mma_thread_slice.partition_A(sA2);
    Tensor tCrA2 = mma_thread_slice.partition_fragment_A2(sA2(_,_,Int<0>{}));                  // (MMA,MMA_M,MMA_K)
    Tensor tCsA3 = mma_thread_slice.partition_A(sA3);
    Tensor tCrA3 = mma_thread_slice.partition_fragment_A2(sA3(_,_,Int<0>{}));                  // (MMA,MMA_M,MMA_K)

    //
    // Copy Atom A retiling — using SmemCopyAtomA2 for both A2 and A3
    //
    auto smem_tiled_copy_A2 = make_tiled_copy_A(SmemCopyAtomA2{}, tiled_mma);
    auto smem_thr_copy_A2   = smem_tiled_copy_A2.get_thread_slice(thread_idx);
    Tensor tCrA_copy_view2  = smem_thr_copy_A2.retile_D(tCrA2);                                    // (CPY,CPY_M,CPY_K)
    Tensor tCsA_copy_view2  = smem_thr_copy_A2.partition_S(sA2);                                    // (CPY,CPY_M,CPY_K,PIPE)
    auto smem_tiled_copy_A3 = make_tiled_copy_A(SmemCopyAtomA2{}, tiled_mma);
    auto smem_thr_copy_A3   = smem_tiled_copy_A3.get_thread_slice(thread_idx);
    Tensor tCrA_copy_view3  = smem_thr_copy_A3.retile_D(tCrA3);                                    // (CPY,CPY_M,CPY_K)
    Tensor tCsA_copy_view3  = smem_thr_copy_A3.partition_S(sA3);                                    // (CPY,CPY_M,CPY_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA2) == size<1>(tCrA_copy_view2));                                            // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA2) == size<2>(tCrA_copy_view2));                                            // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCsA_copy_view2) == size<1>(tCrA_copy_view2));                                  // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA_copy_view2) == size<2>(tCrA_copy_view2));                                  // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA2) == size<1>(accum));                                                      // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                       // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA2) == size<2>(tCsB));                                                       // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA2) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA2));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                          // PIPE
    CUTE_STATIC_ASSERT_V(size<2>(tCrB) > _2{}, "RS loops require more than 2 MMA k-iterations for correctness.");

    //
    // PIPELINED MAIN LOOP — hardcoded 4-k_block interleaving from NestedFP custom kernel
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

    warpgroup_fence_operand(accum);

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};

    // Reconstruction output fragment (fp16) with same layout as the e4m3 fragment
    Tensor tmp = make_fragment_like<cutlass::half_t>(tCrA2.layout());

    // first k tile
    {
      tiled_mma.accumulate_ = GMMA::ScaleOut::One;

      pipeline.consumer_wait(smem_pipe_read);
      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      // Copy all 4 k_blocks of A2/A3 from smem to registers
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,0,read_stage), tCrA_copy_view2(_,_,0));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,0,read_stage), tCrA_copy_view3(_,_,0));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,1,read_stage), tCrA_copy_view2(_,_,1));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,1,read_stage), tCrA_copy_view3(_,_,1));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,2,read_stage), tCrA_copy_view2(_,_,2));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,2,read_stage), tCrA_copy_view3(_,_,2));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,3,read_stage), tCrA_copy_view2(_,_,3));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,3,read_stage), tCrA_copy_view3(_,_,3));
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,0), tCrA3(_,_,0), tmp(_,_,0));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,0), tCrB(_,_,0,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,1), tCrA3(_,_,1), tmp(_,_,1));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,1), tCrB(_,_,1,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,2), tCrA3(_,_,2), tmp(_,_,2));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,2), tCrB(_,_,2,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,3), tCrA3(_,_,3), tmp(_,_,3));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,3), tCrB(_,_,3,read_stage), accum);
      warpgroup_commit_batch();

      --k_tile_count;
      if (k_tile_count == 0) {
        return;
      }

      // Prefetch next tile (only reached when there are more k tiles)
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,0,smem_pipe_read.index()), tCrA_copy_view2(_,_,0));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,0,smem_pipe_read.index()), tCrA_copy_view3(_,_,0));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,1,smem_pipe_read.index()), tCrA_copy_view2(_,_,1));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,1,smem_pipe_read.index()), tCrA_copy_view3(_,_,1));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,2,smem_pipe_read.index()), tCrA_copy_view2(_,_,2));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,2,smem_pipe_read.index()), tCrA_copy_view3(_,_,2));
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,0), tCrA3(_,_,0), tmp(_,_,0));
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,3,read_stage), tCrA_copy_view2(_,_,3));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,3,read_stage), tCrA_copy_view3(_,_,3));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,0), tCrB(_,_,0,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,1), tCrA3(_,_,1), tmp(_,_,1));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,1), tCrB(_,_,1,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,2), tCrA3(_,_,2), tmp(_,_,2));

      // Wait for first 2 GMMAs of this tile to complete, then release the PREVIOUS tile's stage
      // CRITICAL: release must happen BEFORE consumer_wait to avoid 2-stage pipeline deadlock
      warpgroup_wait<2>();
      pipeline.consumer_release(smem_pipe_release);
      ++smem_pipe_release;

      // Now safe to wait for the next tile (producer can fill the just-released stage)
      pipeline.consumer_wait(smem_pipe_read);
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,0,smem_pipe_read.index()), tCrA_copy_view2(_,_,0));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,0,smem_pipe_read.index()), tCrA_copy_view3(_,_,0));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,1,smem_pipe_read.index()), tCrA_copy_view2(_,_,1));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,1,smem_pipe_read.index()), tCrA_copy_view3(_,_,1));
      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,2,smem_pipe_read.index()), tCrA_copy_view2(_,_,2));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,2,smem_pipe_read.index()), tCrA_copy_view3(_,_,2));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,2), tCrB(_,_,2,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,3), tCrA3(_,_,3), tmp(_,_,3));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,3), tCrB(_,_,3,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,0), tCrA3(_,_,0), tmp(_,_,0));

      warpgroup_fence_operand(accum);
    }

    {
      // Last k_tile — no pipeline wait for next stage
      int read_stage = smem_pipe_read.index();

      copy(smem_tiled_copy_A2, tCsA_copy_view2(_,_,3,read_stage), tCrA_copy_view2(_,_,3));
      copy(smem_tiled_copy_A3, tCsA_copy_view3(_,_,3,read_stage), tCrA_copy_view3(_,_,3));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,0), tCrB(_,_,0,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,1), tCrA3(_,_,1), tmp(_,_,1));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,1), tCrB(_,_,1,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,2), tCrA3(_,_,2), tmp(_,_,2));

      warpgroup_wait<2>();

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,2), tCrB(_,_,2,read_stage), accum);
      warpgroup_commit_batch();
      dual_weight_reconstruct<kReconstructionKind>(tCrA2(_,_,3), tCrA3(_,_,3), tmp(_,_,3));

      warpgroup_arrive();
      cute::gemm(tiled_mma, tmp(_,_,3), tCrB(_,_,3,read_stage), accum);
      warpgroup_commit_batch();
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 2;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto
  tensormaps_init(
      Params const& mainloop_params,
      TensorMapStorage& shared_tensormaps,
      int32_t sm_count,
      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap = reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    cute::TmaDescriptor* tma_desc_aux0 = &gmem_tensormap[sm_idx + 2 * sm_count];

    // Bringing tensormaps from params to smem for modification later
    Tensor pA2_tensormap = make_tensor(mainloop_params.tma_load_a2.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA2_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A2), Int<1>{}, Int<1>{});
    Tensor pB_tensormap = make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sB_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});
    Tensor pA3_tensormap = make_tensor(mainloop_params.tma_load_a3.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA3_tensormap = make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_A3), Int<1>{}, Int<1>{});

    if (cute::elect_one_sync()) {
      copy(recast<uint128_t>(pA2_tensormap), recast<uint128_t>(sA2_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
      copy(recast<uint128_t>(pA3_tensormap), recast<uint128_t>(sA3_tensormap));
    }

    __syncwarp();
    return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_aux0);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_address(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A2,
                                                    mainloop_params.ptr_A2[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_A3,
                                                    mainloop_params.ptr_A3[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = (SwapAB? get<1>(problem_shape_mnkl) : get<0>(problem_shape_mnkl));
    const uint32_t N = (SwapAB? get<0>(problem_shape_mnkl) : get<1>(problem_shape_mnkl));
    const uint32_t K = get<2>(problem_shape_mnkl);

    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0,0,0,0,0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B  = {1,1,1,1,1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0,0,0,0,0};

    SwappedElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(
        ptr_A,
        detail::dual_weight_get_gmem_layout(make_shape(M, K, Int<1>{}), mainloop_params.ptr_dA[next_group]));

    SwappedElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(
        ptr_B,
        detail::dual_weight_get_gmem_layout(make_shape(N, K, Int<1>{}), mainloop_params.ptr_dB[next_group]));

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a2, tensor_a,
                                             prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b,
                                             prob_shape_B, prob_stride_B);
    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A2,
                                                            prob_shape_A,
                                                            prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B,
                                                            prob_stride_B);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_A3,
                                                            prob_shape_A,
                                                            prob_stride_A);
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  void
  tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps,
      Params const& mainloop_params,
      cute::tuple<TMs...> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl,
      int32_t next_batch) {
    static_cast<void>(input_tensormaps);
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps,
          mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_cp_fence_release (
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TMs...> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_A2);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
    tma_descriptor_cp_fence_release(get<2>(input_tensormaps), shared_tensormaps.smem_tensormap_A3);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE
  void
  tensormaps_fence_acquire(cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE
  InputTensors
  tensors_perform_update(
      InputTensors const& input_tensors,
      [[maybe_unused]] Params const& mainloop_params,
      [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
      [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
