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
/** Common algorithms on (hierarchical) tensors */

#pragma once

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>

namespace cute
{

//
// for_each
//

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout> const& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return for_each(tensor, op);
}

//
// transform
//

// Similar to std::transform but does not return number of elements affected
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return transform(tensor, op);
}

// Similar to std::transform transforms one tensors and assigns it to another
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in); ++i) {
    tensor_out(i) = op(tensor_in(i));
  }
}

// Accept mutable temporaries
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          UnaryOp&& op)
{
  return transform(tensor_in, tensor_out, op);
}

// Similar to std::transform with a binary operation
// Takes two tensors as input and one tensor as output.
// Applies the binary_op to tensor_in1 and tensor_in2 and
// assigns it to tensor_out
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          BinaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); ++i) {
    tensor_out(i) = op(tensor_in1(i), tensor_in2(i));
  }
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          BinaryOp&& op)
{
  return transform(tensor_in1, tensor_in2, tensor_out, op);
}

// NestedFP dual-weight reconstruction: transform two e4m3 (8-bit) input tensors
// (upper and lower weight bytes) into one fp16 (16-bit) output tensor using
// packed uint32_t arithmetic and __byte_perm instructions.
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out)
{
  // Batch logic operations
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t3 = recast<uint32_t>(tensor_out);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = (b & 0x80808080) >> 7;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;
    uint32_t c = __byte_perm(t1(i), t2(i), 0x1504);
    uint32_t d = __byte_perm(t1(i), t2(i), 0x3726);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }
}

// Accept mutable temporaries (rvalue tensor views from CuTe slicing)
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform2(tensor_in1, tensor_in2, tensor_out);
}

// NestedFP E5M2 RTN reconstruction: undo round-to-nearest packing for E5M2 format.
// stored_upper = upper_orig + inc, stored_lower = lower_orig | inc (normal only).
// Reconstruction detects normal bytes and reverses the rounding.
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE
void
transform2_e5m2_rtn(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
                    Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
                    Tensor<EngineOut,LayoutOut>      & tensor_out)
{
#if defined(__CUDA_ARCH__)
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t3 = recast<uint32_t>(tensor_out);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    // Detect normal bytes: exponent bits [6:2] != 0 per byte.
    uint32_t exp_bits   = a & 0x7C7C7C7Cu;
    uint32_t normal_ff  = __vcmpne4(exp_bits, 0u);         // 0xFF per normal byte, 0 otherwise
    uint32_t normal_01  = normal_ff & 0x01010101u;          // 1 per normal byte
    uint32_t inc_01     = b & normal_01;                   // extract rounding increment (normals only)
    uint32_t upper_orig = a - inc_01;                      // undo increment
    uint32_t lower_orig = b & ~normal_01;                  // strip inc bit from lower
    uint32_t c = __byte_perm(upper_orig, lower_orig, 0x1504u);
    uint32_t d = __byte_perm(upper_orig, lower_orig, 0x3726u);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }
#endif
}

template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2_e5m2_rtn(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
                    Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
                    Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform2_e5m2_rtn(tensor_in1, tensor_in2, tensor_out);
}

// NestedFP E5M2 truncation reconstruction: upper = fp16[15:8], lower = fp16[7:0].
// No rounding correction — just interleave the two bytes back.
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE
void
transform2_e5m2_trunc(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
                      Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
                      Tensor<EngineOut,LayoutOut>      & tensor_out)
{
#if defined(__CUDA_ARCH__)
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t3 = recast<uint32_t>(tensor_out);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t c = __byte_perm(a, b, 0x1504u);
    uint32_t d = __byte_perm(a, b, 0x3726u);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }
#endif
}

template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2_e5m2_trunc(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
                      Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
                      Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform2_e5m2_trunc(tensor_in1, tensor_in2, tensor_out);
}

namespace lazy {

template <class Engine, class Layout, class Fn>
CUTE_HOST_DEVICE constexpr
auto
transform(cute::Tensor<Engine,Layout> const& t, Fn const& fn)
{
  return cute::make_tensor(cute::make_transform_iter(fn, t.data()), t.layout());
}

} // end namespace lazy

} // end namespace cute
