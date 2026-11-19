// The H8R prompt GEMM on XMX with a wider tile per sub-group (QUEUE 33.4; zero/xe2/engine_xw.h, ARC_GEMM_XW=1):
// C[M][ldc] (+)= A[M][K] * W[N][K]^T at column offset c_off, for the H8R layout only (gemm.cl QTYPE 111: int8
// weights in the W8 values layout with one scale per column, int8 activations with one scale per token, the dot
// product over a split's whole K one integer sum).
//
// gemm.cl's XMX kernel gives a sub-group 16 or 32 tokens x 16 columns: each 64-value chunk of a column group's
// weights (1 KB) is loaded for 16 / 32 tokens only, and the activation rows are loaded again by every column group
// (shared through L1 at best). Here a sub-group holds XW_TM tokens x XW_NC column groups (default 32 x 64): the
// activations of a chunk are loaded once for XW_NC * 16 columns, a column group's weights once for XW_TM tokens,
// twice the DPAS work per byte loaded at 32 x 64. The accumulators are TM / 8 x NC int8 vectors (128 registers at
// 32 x 64: the 256-register build).
//
// Bit for bit gemm.cl's: the integer sums are exact in any order (|sum| <= 127 * 127 * K, as there), split-K pieces
// are the same (the host passes s_begin / s_end as gemm.cl gets them), and the epilogue is gemm.cl's expression,
// (token scale * column scale) * (float)sum, then the same accumulate modes. Checked at start against gemm.cl
// without XMX (engine_xw.h xw_check), bit for bit.
//
// -DXW_EMU: DPAS emulated with DP4a and shuffles (tests on GPUs without XMX: ARC_GEMM_XW_TRY=1).
// -DX2D: the activations by 2D block loads, as gemm.cl -DX2D (8 rows x 64 bytes a load).
// -DXW_2D (with X2D, 2026-11-17, ARC_GEMM_XW_2D=1): fewer messages, the same bytes in the same DPAS operands. A chunk's
// 32 activation rows in one 2D read (32 rows x 64 bytes) instead of four of 8 (XW_TM 32); a column group's 1 KB of
// weights as one 2D read of a 64-byte-wide, 16-row surface instead of two 512-byte block reads (lane l gets column l's
// 16 words, as there); with -DXW_2D_C too (ARC_GEMM_XW_2D=1, not =2), C read and written 8 rows x 16 columns a message
// instead of a row a message (a driver whose start check fails that part keeps the loads alone). ocloc bmg / ptl:
// one message each. The loop's messages a chunk at 32 x 64: 16 -> 9 (with the weight prefetch).
// Grid (N: the launch's columns, from group g_base): (16 * (N / (16 * XW_NC)) * rows, WGY), local (16 * WGX, WGY), rows = M / (XW_TM * WGY); the work-groups sweep
// `rgroup` token rows together column block by column block, as gemm.cl, so a weight slice is reused while in L3.
// WGX (2026-11-15, ARC_GEMM_XW_WGX, as a GEMM of 32 sub-groups over 256 tokens x 128 columns does): the
// work-group's sub-groups side by side on the columns too, WGX column blocks of 16 * XW_NC, so the WGY token sub-groups
// share their weights and the WGX column sub-groups their activations in L1. Each sub-group computes what it did
// before, in the same order: the same results bit for bit.
// -DXW_LOCK (ARC_GEMM_XW_LOCK=1): the work-group in step, a split barrier around each 64-value chunk (arrive before its
// loads, wait after its DPAS), so the sub-groups that share a chunk's lines read them while they are in L1. Changes
// no value.

#ifdef XW_LOCK
#pragma OPENCL EXTENSION cl_intel_split_work_group_barrier : enable
#ifdef XW_EMU
#define XW_ARRIVE()
#define XW_WAIT() barrier(CLK_LOCAL_MEM_FENCE)
#else
#define XW_ARRIVE() intel_work_group_barrier_arrive(CLK_LOCAL_MEM_FENCE)
#define XW_WAIT() intel_work_group_barrier_wait(CLK_LOCAL_MEM_FENCE)
#endif
#else
#define XW_ARRIVE()
#define XW_WAIT()
#endif

#define SG 16
#ifndef XW_TM
#define XW_TM 32
#endif
#ifndef XW_NC
#define XW_NC 4
#endif
// (2026-11-13, for a sweep beyond 32 x 64: XW_TM 64 and XW_NC 6 / 8. ocloc for a B60 at 256 registers: 32 x 96 spills
//  ~2 bytes, 16 x 128 and 64 x 32 none, 32 x 128 and 64 x 64 heavily; 32 x 64 none)
#if XW_TM != 16 && XW_TM != 32 && XW_TM != 64
#error "XW_TM must be 16, 32 or 64"
#endif
#if XW_NC != 1 && XW_NC != 2 && XW_NC != 4 && XW_NC != 6 && XW_NC != 8
#error "XW_NC must be 1, 2, 4, 6 or 8"
#endif
#define XW_TG (XW_TM / 8)  // DPAS row groups of 8 tokens
// -DXW_C_RND (with XW_2D_C): the add mode's product rounded to f32 before `+ old`, not fused with it into a mad (an OR
// with M >> 30, 0 but unknown to the compiler). A compiler that fuses the 2D epilogue's add and not gemm.cl's (a Data
// Center GPU Max 1550, driver 25.18) fails the start check in the add modes by 1-6 ulp; with this the values are
// gemm.cl's again (engine_xw.h xw_init)
// -DXW_PF=1 / 2 / 3 (2026-11-13, B60 prefill plan 13.2): prefetch into L1 the weights (1), the activations (2, with
// X2D) or both (3) of the 64-value chunk XW_PFD chunks ahead (1, 2 or 4). The prefetches go to L1 only: the values
// read, and so the results, are the same bit for bit
#ifndef XW_PF
#define XW_PF 0
#endif
#ifndef XW_WGX
#define XW_WGX 1
#endif
#ifndef XW_PFD
#define XW_PFD 2
#endif
// (a column group's 1 KB chunk of weights (16 words x 16 lanes, 1 KB-aligned) as a 64-byte-wide surface of 16 rows:
//  one 2D prefetch message, where two buffer prefetches of 16 x 8 words made four (ocloc bmg))
#ifdef XW_EMU
#define XW_PREF_W(p) prefetch((p), 16)  // (the GPUs without XMX that emulate it here: the portable built-in)
#else
#define XW_PREF_W(p) intel_sub_group_2d_block_prefetch_32b_16r16x1c((__global const void*)(p), 64, 16, 64, (int2)(0, 0))
#endif
#define BCAST(x, i) intel_sub_group_shuffle((x), (i))

// c + A (8 rows x 32 int8: lane l holds values 2l, 2l + 1 of each row) x B (32 x 16: lane n = column n, its 32 values
// as 8 words of 4), int32, component r = row r, lane = column
inline int8 xw_mad(short8 a, uint8 b, int8 c) {
#ifdef XW_EMU
  const int av[8] = {a.s0, a.s1, a.s2, a.s3, a.s4, a.s5, a.s6, a.s7};
  const uint bw[8] = {b.s0, b.s1, b.s2, b.s3, b.s4, b.s5, b.s6, b.s7};
  int cr[8] = {c.s0, c.s1, c.s2, c.s3, c.s4, c.s5, c.s6, c.s7};
#pragma unroll
  for (int w = 0; w < 8; ++w)
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      // (values 4w .. 4w + 3 of row r: lanes 2w and 2w + 1, two bytes each)
      const uint lo = (uint)intel_sub_group_shuffle(av[r], 2 * w) & 0xFFFFu;
      const uint hi = (uint)intel_sub_group_shuffle(av[r], 2 * w + 1) & 0xFFFFu;
      cr[r] += dot_4x8packed_ss_int(lo | (hi << 16), bw[w]);
    }
  return (int8)(cr[0], cr[1], cr[2], cr[3], cr[4], cr[5], cr[6], cr[7]);
#else
  return intel_sub_group_i8_i8_matrix_mad_k32(a, as_int8(b), c);
#endif
}

__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void gemm_xw(__global const uint* restrict wq, __global const uint* restrict wh,
                      __global const char* restrict aq, __global const float* restrict asc,
                      __global float* restrict C, int M, int N, int K, int ldc, int c_off, int accumulate,
                      int rgroup, int s_begin, int s_end, int g_base) {
  // (g_base: the first 16-column group of this launch; N its columns: XW_NC 6 runs a matrix's first 96 * a columns,
  //  and the 32 x 64 kernel the rest from g_base 6 * a, engine_xw.h xw_matmul; 0 and the whole N otherwise)
  // (wgx column sub-groups a work-group, sx this one's: the host keeps NB a multiple of wgx, and local size 16 * wgx)
  const int wgx = XW_WGX;
#if XW_WGX > 1
  const int sx = (int)get_local_id(0) / SG;
#else
  const int sx = 0;
#endif
  const int NB = N / (SG * XW_NC * wgx), rows = M / (XW_TM * (int)get_local_size(1));
  const int w = get_group_id(0), grp = w / (rgroup * NB), rem = w % (rgroup * NB);
  const int rin = min(rgroup, rows - grp * rgroup);
  const int cb = rem / rin, row = grp * rgroup + rem % rin;
  const int m0 = (row * (int)get_local_size(1) + (int)get_local_id(1)) * XW_TM;
  const int nsb = K / 256, g0 = g_base + (cb * wgx + sx) * XW_NC;
#ifndef X2D
  __global const char* ap = aq + (size_t)m0 * K;
#endif

  int8 xacc[XW_TG][XW_NC];
#pragma unroll
  for (int t = 0; t < XW_TG; ++t)
#pragma unroll
    for (int c = 0; c < XW_NC; ++c) xacc[t][c] = (int8)(0);

  for (int s = s_begin; s < s_end; ++s) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      XW_ARRIVE();
#if XW_PF
      // ---- the chunk XW_PFD ahead into L1 while this one's DPAS run (B60 prefill plan 13.2): bit 1 its weights, bit 2
      //      its activations (2D loads only); a prefetch changes no value. Not past this piece's end (s_end)
      {
        const int qn = s * 4 + j + XW_PFD;
        if (qn < s_end * 4) {
#if XW_PF & 1
#pragma unroll
          for (int c = 0; c < XW_NC; ++c) XW_PREF_W(wq + ((size_t)(g0 + c) * nsb * 4 + qn) * (16 * SG));
#endif
#if (XW_PF & 2) && defined(X2D)
#pragma unroll
          for (int t = 0; t < XW_TG; ++t)  // (the work-group's wgx column sub-groups share these rows: one each)
            if (t % wgx == sx) intel_sub_group_2d_block_prefetch_8b_8r32x2c(aq, K, M, K, (int2)(qn * 64, m0 + 8 * t));
#endif
        }
      }
#endif
      // ---- this 64-value chunk of the XW_TM tokens' rows: al values 0..31, ah 32..63 of 8 rows each
      short8 al[XW_TG], ah[XW_TG];
#if defined(XW_2D) && defined(X2D) && XW_TM == 32
      {  // (one read of the 32 rows: block 0 (values 0..31) rows 0..31 in d[0..31], block 1 (32..63) in d[32..63])
        ushort d[64];
        intel_sub_group_2d_block_read_8b_32r32x2c(aq, K, M, K, (int2)(s * 256 + j * 64, m0), d);
#pragma unroll
        for (int t = 0; t < XW_TG; ++t) {
          al[t] = as_short8((ushort8)(d[8 * t], d[8 * t + 1], d[8 * t + 2], d[8 * t + 3], d[8 * t + 4], d[8 * t + 5],
                                      d[8 * t + 6], d[8 * t + 7]));
          ah[t] = as_short8((ushort8)(d[32 + 8 * t], d[33 + 8 * t], d[34 + 8 * t], d[35 + 8 * t], d[36 + 8 * t],
                                      d[37 + 8 * t], d[38 + 8 * t], d[39 + 8 * t]));
        }
      }
#else
#pragma unroll
      for (int t = 0; t < XW_TG; ++t) {
#ifdef X2D
        ushort d[16];
        intel_sub_group_2d_block_read_8b_8r32x2c(aq, K, M, K, (int2)(s * 256 + j * 64, m0 + 8 * t), d);
        al[t] = as_short8((ushort8)(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]));
        ah[t] = as_short8((ushort8)(d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]));
#else
        short ta[8], tb[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          __global const char* rp = ap + (size_t)(8 * t + p) * K + s * 256 + j * 64;
          ta[p] = as_short(intel_sub_group_block_read_us((__global const ushort*)rp));
          tb[p] = as_short(intel_sub_group_block_read_us((__global const ushort*)(rp + 32)));
        }
        al[t] = (short8)(ta[0], ta[1], ta[2], ta[3], ta[4], ta[5], ta[6], ta[7]);
        ah[t] = (short8)(tb[0], tb[1], tb[2], tb[3], tb[4], tb[5], tb[6], tb[7]);
#endif
      }
#endif
      // ---- each column group's 64 values (lo 0..31, hi 32..63), used by every row group
#pragma unroll
      for (int c = 0; c < XW_NC; ++c) {
        __global const uint* dp = wq + ((size_t)(g0 + c) * nsb * 4 + s * 4 + j) * (16 * SG);
#if defined(XW_2D) && defined(X2D)
        // (the 1 KB chunk as 16 rows of 16 words: lane l gets column l, words 0..15, what the two block reads gave)
        uint wb[16];
        intel_sub_group_2d_block_read_32b_16r16x1c((__global const void*)dp, 64, 16, 64, (int2)(0, 0), wb);
        const uint8 lo = (uint8)(wb[0], wb[1], wb[2], wb[3], wb[4], wb[5], wb[6], wb[7]);
        const uint8 hi = (uint8)(wb[8], wb[9], wb[10], wb[11], wb[12], wb[13], wb[14], wb[15]);
#else
        const uint8 lo = intel_sub_group_block_read8(dp), hi = intel_sub_group_block_read8(dp + 8 * SG);
#endif
#pragma unroll
        for (int t = 0; t < XW_TG; ++t) xacc[t][c] = xw_mad(ah[t], hi, xw_mad(al[t], lo, xacc[t][c]));
      }
      XW_WAIT();
    }
  }

  // ---- the one epilogue, gemm.cl's: the token's scale x the column's x the sum, then the accumulate mode
  float av[XW_TM / 16];
#pragma unroll
  for (int v = 0; v < XW_TM / 16; ++v)
    av[v] = as_float(intel_sub_group_block_read((__global const uint*)(asc + m0 + 16 * v)));
#pragma unroll
  for (int c = 0; c < XW_NC; ++c) {
    const int g = g0 + c;
    const float dcol = as_float(intel_sub_group_block_read(wh + (size_t)g * SG));
#pragma unroll
    for (int t = 0; t < XW_TG; ++t) {
      const int8 x = xacc[t][c];
      const int is[8] = {x.s0, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6, x.s7};
#if defined(XW_2D_C) && defined(X2D)
      // (-DXW_2D_C, ARC_GEMM_XW_2D=1: a driver whose compiler fuses `res + old` differently here than in gemm.cl fails
      //  the start check in the add modes, which then drops this form and keeps the plain epilogue below, gemm.cl's code)
      // (the 8 rows x 16 columns of this row group and column group: one 2D read of C where the mode needs it, the same
      //  expression per value as below, one 2D write. C as a surface of M rows x ldc floats from its start; ldc and
      //  c_off are multiples of 4 floats (engine.h matmul1))
      {
        const int2 at = (int2)(c_off + g * SG, m0 + 8 * t);
        uint old[8], out[8];
        if (accumulate != 0)
          intel_sub_group_2d_block_read_32b_8r16x1c((__global const void*)C, ldc * 4, M, ldc * 4, at, old);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
          const int m = 8 * t + r;
          const float res = BCAST(av[m / 16], m % 16) * dcol * (float)is[r];
          float v = res;
          if (accumulate == 1) {
#ifdef XW_C_RND
            v = as_float(as_uint(res) | (uint)(M >> 30)) + as_float(old[r]);
#else
            v += as_float(old[r]);
#endif
          } else if (accumulate == 2) {
            const float gt = as_float(old[r]);
            v *= gt / (1.0f + exp(-gt));
          }
          out[r] = as_uint(v);
        }
        intel_sub_group_2d_block_write_32b_8r16x1c((__global void*)C, ldc * 4, M, ldc * 4, at, out);
      }
      continue;
#endif
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const int m = 8 * t + r;
        const float res = BCAST(av[m / 16], m % 16) * dcol * (float)is[r];
        __global uint* cp = (__global uint*)(C + c_off + (size_t)(m0 + m) * ldc + g * SG);
        float v = res;
        if (accumulate == 1) {
          v += as_float(intel_sub_group_block_read(cp));
        } else if (accumulate == 2) {  // fused SwiGLU: C holds the gate projection, v the up projection
          const float gt = as_float(intel_sub_group_block_read(cp));
          v *= gt / (1.0f + exp(-gt));
        }
        intel_sub_group_block_write(cp, as_uint(v));
      }
    }
  }
}
