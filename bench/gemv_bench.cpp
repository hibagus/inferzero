// Times and validates the single-token GEMV (zero/kernels/decode.cl "gemv" or a variant with the same arguments) on real
// weights of a GGUF model. A name with %d runs every layer that has the tensor in turn, so the weights stream from
// memory as in token generation instead of staying in the cache.
// usage: gemv_bench model.gguf tensor [defines] [kernel] [lanes_per_rowgroup] [iters] [kernel_file]
//   e.g. gemv_bench g12.gguf token_embd.weight "-DGROWS=2"
//        gemv_bench g12.gguf blk.%d.ffn_gate.weight "-DGROWS=1" gemv 16 20 scratch/decode_try.cl
// A kernel name starting with "gemv_mt" (gemv_mt / gemv_mt_v: T token rows per weight read, speculative
// verification) runs with T = GB_T (default 8) activation rows and the extra arguments (T, ldc = N, c_off = 0,
// acc = 0); defines should carry the matching -DTB. Errors are reported for every token row.
// GB_CMP=kernel[:defines] compares bitwise with another kernel (same argument form), built from GB_CMP_FILE if set.
#include "../src/clrt.h"
#include "../src/quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

// GB_REPACK=1 (2026-11-17, a prototype for the GEMV of generation, IQ3_S only): the kernel under test reads each
// matrix in a planar repack of its padded GPU rows (raw_padded_sb_bytes = 112 B a superblock): per row, over its T =
// nsb / 2 pairs of superblocks, plane by plane: qs (128 B a pair: the 32-block of lane l = (superblock l / 8, block
// l % 8) as words l and 16 + l), signs (64 B: word l), qh (16 B: byte l), meta (16 B: bytes 0-3 / 4-7 the two
// superblocks' scales, 8-9 / 10-11 their d). The same row size. GB_CMP's kernel reads the original rows, so the
// comparison is against today's layout. (A pair needs both superblocks: an odd count is refused.)
static std::vector<uint8_t> repack_iq3s(const std::vector<uint8_t>& pad, int N, int nsb, size_t bp) {
  const int T = nsb / 2;
  std::vector<uint8_t> out(pad.size(), 0);
  for (int n = 0; n < N; ++n) {
    const uint8_t* in = &pad[(size_t)n * nsb * bp];
    uint8_t* o = &out[(size_t)n * nsb * bp];
    for (int tp = 0; tp < T; ++tp) {
      for (int l = 0; l < 16; ++l) {
        const int sb = 2 * tp + l / 8, ib = l % 8;
        const uint8_t* p = in + (size_t)sb * bp;
        memcpy(o + tp * 128 + l * 4, p + 8 * ib, 4);
        memcpy(o + tp * 128 + (16 + l) * 4, p + 8 * ib + 4, 4);
        memcpy(o + (size_t)T * 128 + tp * 64 + l * 4, p + 72 + 4 * ib, 4);
        o[(size_t)T * 192 + tp * 16 + l] = p[64 + ib];
      }
      for (int h = 0; h < 2; ++h) {
        const uint8_t* p = in + (size_t)(2 * tp + h) * bp;
        memcpy(o + (size_t)T * 208 + tp * 16 + 4 * h, p + 104, 4);
        memcpy(o + (size_t)T * 208 + tp * 16 + 8 + 2 * h, p + 108, 2);
      }
    }
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: gemv_bench model.gguf tensor [defines] [kernel] [lanes] [iters] [file]\n"); return 1; }
  const std::string tname = argv[2], defs = argc > 3 ? argv[3] : "", kname = argc > 4 ? argv[4] : "gemv";
  const int lanes = argc > 5 ? atoi(argv[5]) : 16, iters = argc > 6 ? atoi(argv[6]) : 10;
  const std::string file = argc > 7 ? argv[7] : "zero/kernels/decode.cl";
  int grows = 4;
  if (auto p = defs.find("-DGROWS="); p != std::string::npos) grows = atoi(defs.c_str() + p + 8);
  const bool mt = kname.rfind("gemv_mt", 0) == 0;
  const int T = mt ? (getenv("GB_T") ? atoi(getenv("GB_T")) : 8) : 1;
  if (T < 1 || T > 16) { fprintf(stderr, "GB_T must be 1..16\n"); return 1; }
  if (mt) {
    int tb = 2;  // (the kernels' default)
    if (auto p = defs.find("-DTB="); p != std::string::npos) tb = atoi(defs.c_str() + p + 5);
    if (T > tb) fprintf(stderr, "warning: T = %d > TB = %d: tokens beyond TB are not computed\n", T, tb);
  }

  GgufFile g(argv[1]);
  std::vector<const GgufTensor*> ts;
  if (tname.find("%d") != std::string::npos) {
    for (int l = 0; l < 1000; ++l) {
      char nm[256];
      snprintf(nm, sizeof nm, tname.c_str(), l);
      const GgufTensor* tp = g.find(nm);
      if (!tp) { if (l > 64) break; continue; }
      const GgufTensor& t = *tp;
      // (GB_TYPE=n: only tensors of ggml type n, e.g. a mixed-type model's main type)
      if (getenv("GB_TYPE") && (int)t.type != atoi(getenv("GB_TYPE"))) continue;
      if (!ts.empty() && (t.type != ts[0]->type || t.dims != ts[0]->dims)) continue;  // one type and shape per run
      ts.push_back(&t);
    }
  } else {
    ts.push_back(&g.get(tname));
  }
  if (ts.empty()) { fprintf(stderr, "tensor not found\n"); return 1; }
  const uint32_t type = ts[0]->type;
  // (small-block types with K % 256 != 0, Qwen3.8 Flash Next's K = 640 down projections: rows padded with zero
  //  blocks to the next multiple of 256, the activations with zeros, as the engine does)
  const int K0 = (int)ts[0]->dims[0], K = is_small_block_type(type) ? padded_k(K0) : K0;
  // (GB_N=n: only the first n rows of each matrix, the share of one GPU of a model split by rows)
  const int N = getenv("GB_N") ? std::min((int)ts[0]->dims[1], atoi(getenv("GB_N"))) : (int)ts[0]->dims[1];
  const int nsb = K / 256, M = 16;
  std::vector<std::vector<uint8_t>> padded_src(ts.size());
  if (K != K0)
    for (size_t i = 0; i < ts.size(); ++i) {
      int blk = 0, bytes = 0;
      ggml_block_info(type, blk, bytes);
      padded_src[i].resize((size_t)N * (K / blk) * bytes);
      pad_small_block_rows(type, ts[i]->data, N, K0, padded_src[i].data());
    }
  auto src_of = [&](size_t i) { return K != K0 ? padded_src[i].data() : ts[i]->data; };
  const size_t bb = gguf_sb_bytes(type), bp = raw_padded_sb_bytes(type);
  printf("%s: %s [%d x %d], %zu matrices, %.1f MB each\n", tname.c_str(), ggml_type_name(type), N, K, ts.size(),
         ts[0]->nbytes / 1e6);

  ClRt cl;
  if (!cl.init()) return 1;
  if (getenv("TOUCHALL")) {  // read the whole mapped file first, as the engine's load does
    uint64_t sum = 0;
    for (auto& t : g.tensors)
      for (uint64_t i = 0; i < t.nbytes; i += 4096) sum += t.data[i];
    printf("touched the file (%llu)\n", (unsigned long long)sum);
  }
  std::vector<cl_mem> raws, junk;
  std::vector<cl_mem> rawk;  // (what the kernel under test reads: raws, or with GB_REPACK their repack)
  const bool repack = getenv("GB_REPACK") && atoi(getenv("GB_REPACK")) != 0;
  if (repack && (type != GGML_IQ3_S || bp != 112 || nsb % 2)) {
    fprintf(stderr, "GB_REPACK: IQ3_S rows of an even number of superblocks only\n");
    return 1;
  }
  if (const char* e = getenv("BIGALLOC_GB")) {  // simulate the engine's memory footprint (allocated before the weights)
    for (int i = 0; i < atoi(e) * 4; ++i) {
      cl_mem m = cl.buffer(CL_MEM_READ_WRITE, (size_t)256 << 20);
      const uint32_t z = 1;
      CL_CHECK(clEnqueueFillBuffer(cl.queue, m, &z, 4, 0, (size_t)256 << 20, 0, nullptr, nullptr));
      junk.push_back(m);
    }
    clFinish(cl.queue);
  }
  for (size_t ti = 0; ti < ts.size(); ++ti) {
    std::vector<uint8_t> pad((size_t)N * nsb * bp + 256, 0);
    for (size_t i = 0; i < (size_t)N * nsb; ++i) gpu_raw_sb(type, src_of(ti) + i * bb, &pad[i * bp]);
    if (getenv("HOSTPTR")) {  // the engine's upload: CL_MEM_ALLOC_HOST_PTR + map
      cl_int err;
      cl_mem m = clCreateBuffer(cl.ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, pad.size(), nullptr, &err);
      CL_CHECK(err);
      void* dst = clEnqueueMapBuffer(cl.queue, m, CL_TRUE, CL_MAP_WRITE, 0, pad.size(), 0, nullptr, nullptr, &err);
      CL_CHECK(err);
      memcpy(dst, pad.data(), pad.size());
      clEnqueueUnmapMemObject(cl.queue, m, dst, 0, nullptr, nullptr);
      raws.push_back(m);
    } else {
      raws.push_back(cl.buffer(CL_MEM_READ_ONLY, pad.size(), pad.data()));
    }
    if (repack) {
      const std::vector<uint8_t> rp = repack_iq3s(pad, N, nsb, bp);
      rawk.push_back(cl.buffer(CL_MEM_READ_ONLY, rp.size(), rp.data()));
    } else {
      rawk.push_back(raws.back());
    }
  }
  // activations: int8 per 32 with float scales (asc [K/32][M]), per-16 sums (absum [K/16][M]); token rows 0..T-1
  // (row t of aq at t * K), the other rows of M zero
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> x(K);
  std::vector<std::vector<float>> xq(T, std::vector<float>(K));
  std::vector<int8_t> aq((size_t)M * K, 0);
  std::vector<float> asc((size_t)(K / 32) * M, 0.0f);
  std::vector<uint16_t> absum((size_t)(K / 16) * M, 0);
  for (int t = 0; t < T; ++t) {
    for (auto& v : x) v = nd(rng) * (1.0f + 3.0f * (rng() % 97 == 0));
    for (int i = K0; i < K; ++i) x[i] = 0.0f;
    for (int b = 0; b < K / 32; ++b) {
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * b + i]));
      const float sc = amax / 127.0f, inv = amax > 0 ? 127.0f / amax : 0.0f;
      asc[(size_t)b * M + t] = sc;
      for (int h = 0; h < 2; ++h) {
        int sum = 0;
        for (int i = 0; i < 16; ++i) {
          const int k = 32 * b + 16 * h + i, q = (int)std::nearbyint(x[k] * inv);
          aq[(size_t)t * K + k] = (int8_t)q;
          xq[t][k] = q * sc;
          sum += q;
        }
        absum[(size_t)(2 * b + h) * M + t] = (uint16_t)(int16_t)sum;
      }
    }
  }
  auto tabs = build_iq_tables();
  cl_mem d_tabs = cl.buffer(CL_MEM_READ_ONLY, tabs.size() * 4, tabs.data());
  cl_mem d_aq = cl.buffer(CL_MEM_READ_ONLY, aq.size(), aq.data());
  cl_mem d_asc = cl.buffer(CL_MEM_READ_ONLY, asc.size() * 4, asc.data());
  cl_mem d_abs = cl.buffer(CL_MEM_READ_ONLY, absum.size() * 2, absum.data());
  const size_t NC = (size_t)T * N;  // outputs: token t's row at t * ldc, ldc = N
  cl_mem d_c = cl.buffer(CL_MEM_READ_WRITE, NC * 4);

  const std::string src = read_file(file);
  cl_program prog = cl.build(src, "-cl-std=CL3.0 -cl-mad-enable -DQT=" + std::to_string(type) + " " + defs);
  cl_kernel k = cl.kernel(prog, kname.c_str());
  printf("kernel %s %s: spill %zu bytes%s\n", kname.c_str(), defs.c_str(), cl.spill_bytes(k),
         mt ? (" (T = " + std::to_string(T) + ")").c_str() : "");
  const size_t lws = 256, gws = ((size_t)(N + grows - 1) / grows * lanes + lws - 1) / lws * lws;
  const int c_off = 0, acc = 0, ldc = N;
  // (one argument form per kernel family: gemv / gemv_v, or gemv_mt / gemv_mt_v with T, ldc)
  auto set_k = [&](cl_kernel kk, bool kmt, cl_mem raw) {
    if (kmt) set_args(kk, 0, raw, N, K, d_tabs, d_aq, d_asc, d_abs, M, T, d_c, ldc, c_off, acc);
    else set_args(kk, 0, raw, N, K, d_tabs, d_aq, d_asc, d_abs, M, d_c, c_off, acc);
  };
  auto launch = [&](cl_mem raw, cl_event* e) {
    set_k(k, mt, raw);
    CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k, 1, nullptr, &gws, &lws, 0, nullptr, e));
  };
  auto poison = [&]() {  // NaN in every output first: a missing store shows up in the comparisons
    const uint32_t nan = 0xFFFFFFFFu;
    CL_CHECK(clEnqueueFillBuffer(cl.queue, d_c, &nan, 4, 0, NC * 4, 0, nullptr, nullptr));
  };
  // correctness on the first matrix against the CPU dequantization
  std::vector<float> c(NC);
  poison();
  launch(rawk[0], nullptr);
  CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, NC * 4, c.data(), 0, nullptr, nullptr));
  if (const char* other = getenv("GB_CMP")) {  // GB_CMP=kernel[:defines]: bitwise comparison with another kernel
    std::string on = other, od = defs;
    if (const size_t colon = on.find(':'); colon != std::string::npos) { od = on.substr(colon + 1); on.resize(colon); }
    const char* cf = getenv("GB_CMP_FILE");  // (the other kernel from another source file, e.g. the unchanged one)
    const bool omt = on.rfind("gemv_mt", 0) == 0;
    if (omt != mt) { fprintf(stderr, "GB_CMP: %s has the other argument form\n", on.c_str()); return 1; }
    cl_program p2 = cl.build(cf ? read_file(cf) : src, "-cl-std=CL3.0 -cl-mad-enable -DQT=" + std::to_string(type) + " " + od);
    cl_kernel k2 = cl.kernel(p2, on.c_str());
    int g2 = 4;
    if (auto p = od.find("-DGROWS="); p != std::string::npos) g2 = atoi(od.c_str() + p + 8);
    const size_t gws2 = ((size_t)(N + g2 - 1) / g2 * lanes + lws - 1) / lws * lws;
    std::vector<float> c2(NC);
    size_t total = 0;
    for (size_t m = 0; m < raws.size(); ++m) {
      poison();
      launch(rawk[m], nullptr);
      CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, NC * 4, c.data(), 0, nullptr, nullptr));
      poison();
      set_k(k2, omt, raws[m]);
      CL_CHECK(clEnqueueNDRangeKernel(cl.queue, k2, 1, nullptr, &gws2, &lws, 0, nullptr, nullptr));
      CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, NC * 4, c2.data(), 0, nullptr, nullptr));
      if (memcmp(c.data(), c2.data(), NC * 4)) {
        for (int t = 0; t < T; ++t) {
          int nd = 0;
          for (int n = 0; n < N; ++n) nd += memcmp(&c[(size_t)t * N + n], &c2[(size_t)t * N + n], 4) != 0;
          if (nd) printf("matrix %zu token %d: %d of %d outputs differ from %s\n", m, t, nd, N, on.c_str());
          total += nd;
        }
      }
    }
    printf("bitwise comparison with %s (%s%s%s) over %zu matrices x %d tokens: %s\n", on.c_str(), od.c_str(),
           cf ? ", file " : "", cf ? cf : "", raws.size(), T, total ? "DIFFERENT" : "identical");
    poison();
    launch(rawk[0], nullptr);
    CL_CHECK(clEnqueueReadBuffer(cl.queue, d_c, CL_TRUE, 0, NC * 4, c.data(), 0, nullptr, nullptr));
  }
  std::vector<double> maxerr(T, 0.0), maxref(T, 0.0);
  std::vector<float> row(K);
  const size_t rb = (size_t)nsb * bb;
  for (int n = 0; n < N; n += std::max(1, N / 4096)) {
    dequant_row(type, src_of(0) + (size_t)n * rb, K, row.data());
    for (int t = 0; t < T; ++t) {
      double s = 0;
      for (int i = 0; i < K; ++i) s += (double)row[i] * xq[t][i];
      const double got = c[(size_t)t * N + n];
      maxerr[t] = std::isfinite(got) ? std::max(maxerr[t], std::fabs(s - got)) : INFINITY;
      maxref[t] = std::max(maxref[t], std::fabs(s));
    }
  }
  for (int t = 0; t < T; ++t)
    printf("%smax |err| %.3g (max |ref| %.3g, rel %.2e)\n", mt ? ("token " + std::to_string(t) + ": ").c_str() : "",
           maxerr[t], maxref[t], maxerr[t] / std::max(maxref[t], 1e-30));
  // timing: every matrix in turn, iters rounds
  std::vector<double> per;
  for (int it = 0; it < iters + 1; ++it) {
    double ms = 0;
    for (auto r : rawk) {
      cl_event e;
      launch(r, &e);
      ms += ClRt::event_ms(e);
      clReleaseEvent(e);
    }
    if (it > 0) per.push_back(ms / raws.size());
  }
  std::sort(per.begin(), per.end());
  const double med = per[per.size() / 2], bytes = (double)N * nsb * bb;
  printf("median %.3f ms per matrix (min %.3f) -> %.1f GB/s of weights%s\n", med, per[0], bytes / med / 1e6,
         mt ? (" (" + std::to_string(T) + " tokens per read)").c_str() : "");
  {  // back to back without events: the time per launch with the gaps between kernels, as in token generation
    CL_CHECK(clFinish(cl.queue));
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it)
      for (auto r : rawk) launch(r, nullptr);
    CL_CHECK(clFinish(cl.queue));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
                      ((double)iters * rawk.size());
    printf("back to back: %.3f ms per launch (%.1f GB/s)\n", ms, bytes / ms / 1e6);
  }
  return 0;
}
