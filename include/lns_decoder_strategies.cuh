#ifndef ACCESSOR_UNIVERSAL_TEST_LNS_DECODER_STRATEGIES_CUH_
#define ACCESSOR_UNIVERSAL_TEST_LNS_DECODER_STRATEGIES_CUH_

#include "bitwidth_benchmark_core.hpp"
#include "lns_benchmark_core.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace aut::lns_strategy {

using bitwidth::compute_kind;
using bitwidth::compute_t;

enum class decoder_kind {
  reference_exp2,
  ex2_approx,
  full_lut_global,
  full_lut_shared,
  fraction_lut_global,
  fraction_lut_shared,
  fraction_lut_warp,
  split_linear,
  split_quadratic,
  split_cubic,
  pair_lut_global,
  pair_lut_shared
};

enum class multiply_kind { ordinary, fused };

template <compute_kind Compute> struct table_bundle {
  const compute_t<Compute> *full{};
  const compute_t<Compute> *fraction{};
  const compute_t<Compute> *coarse{};
  const compute_t<Compute> *pair{};
};

template <typename Format, decoder_kind Decoder>
inline constexpr std::size_t table_entries_v = [] {
  if constexpr (Decoder == decoder_kind::full_lut_global ||
                Decoder == decoder_kind::full_lut_shared) {
    return std::size_t{1} << Format::total_bits;
  } else if constexpr (Decoder == decoder_kind::fraction_lut_global ||
                       Decoder == decoder_kind::fraction_lut_shared ||
                       Decoder == decoder_kind::fraction_lut_warp) {
    return std::size_t{1} << Format::log_fraction_bits;
  } else if constexpr (Decoder == decoder_kind::split_linear ||
                       Decoder == decoder_kind::split_quadratic ||
                       Decoder == decoder_kind::split_cubic) {
    constexpr int coarse_bits = Format::log_fraction_bits < 8
                                    ? Format::log_fraction_bits
                                    : 8;
    return std::size_t{1} << coarse_bits;
  } else if constexpr (Decoder == decoder_kind::pair_lut_global ||
                       Decoder == decoder_kind::pair_lut_shared) {
    return std::size_t{1} << (2 * Format::total_bits);
  } else {
    return std::size_t{0};
  }
}();

template <decoder_kind Decoder>
inline constexpr bool uses_shared_table_v =
    Decoder == decoder_kind::full_lut_shared ||
    Decoder == decoder_kind::fraction_lut_shared ||
    Decoder == decoder_kind::pair_lut_shared;

template <decoder_kind Decoder>
inline constexpr bool is_pair_decoder_v =
    Decoder == decoder_kind::pair_lut_global ||
    Decoder == decoder_kind::pair_lut_shared;

template <decoder_kind Decoder>
inline constexpr bool is_full_decoder_v =
    Decoder == decoder_kind::full_lut_global ||
    Decoder == decoder_kind::full_lut_shared;

template <typename Format, compute_kind Compute, decoder_kind Decoder>
struct decoder_context {
  const compute_t<Compute> *table{};
  compute_t<Compute> warp_entry{};
};

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__device__ __forceinline__ decoder_context<Format, Compute, Decoder>
prepare_context(table_bundle<Compute> tables, compute_t<Compute> *shared) {
  const compute_t<Compute> *global{};
  if constexpr (is_full_decoder_v<Decoder>) {
    global = tables.full;
  } else if constexpr (Decoder == decoder_kind::fraction_lut_global ||
                       Decoder == decoder_kind::fraction_lut_shared ||
                       Decoder == decoder_kind::fraction_lut_warp) {
    global = tables.fraction;
  } else if constexpr (Decoder == decoder_kind::split_linear ||
                       Decoder == decoder_kind::split_quadratic ||
                       Decoder == decoder_kind::split_cubic) {
    global = tables.coarse;
  } else if constexpr (is_pair_decoder_v<Decoder>) {
    global = tables.pair;
  }

  decoder_context<Format, Compute, Decoder> context{};
  if constexpr (uses_shared_table_v<Decoder>) {
    for (std::size_t index = threadIdx.x;
         index < table_entries_v<Format, Decoder>; index += blockDim.x) {
      shared[index] = global[index];
    }
    __syncthreads();
    context.table = shared;
  } else {
    context.table = global;
  }

  if constexpr (Decoder == decoder_kind::fraction_lut_warp) {
    static_assert(Format::log_fraction_bits <= 5,
                  "warp-register LUT supports at most 32 entries");
    const auto lane = threadIdx.x & 31;
    context.warp_entry =
        lane < static_cast<int>(table_entries_v<Format, Decoder>)
            ? global[lane]
            : compute_t<Compute>{};
  }
  return context;
}

template <typename Target>
__device__ __forceinline__ Target warp_lookup(Target entry,
                                               std::uint32_t index,
                                               const Target *fallback) {
  // Every lane named by `active` must execute the same shuffle.  `index` is
  // data dependent, so testing whether the holding lane is active *before*
  // shuffling deadlocks the warp: the lanes that take the early exit never
  // arrive and the remaining lanes wait for them until the job is killed.
  // Shuffle unconditionally instead and discard the result afterwards; a
  // shuffle from an inactive lane is undefined but never blocks.
  const auto active = __activemask();
  const auto source = static_cast<int>(index & 31u);
  Target shuffled{};
  if constexpr (std::is_same_v<Target, float>) {
    shuffled = __shfl_sync(active, entry, source);
  } else {
    const auto low = __double2loint(entry);
    const auto high = __double2hiint(entry);
    shuffled = __hiloint2double(__shfl_sync(active, high, source),
                                __shfl_sync(active, low, source));
  }
  const auto holder_active = (active & (std::uint32_t{1} << source)) != 0;
  return holder_active ? shuffled : fallback[index];
}

template <typename Target>
__device__ __forceinline__ Target apply_power_of_two(Target magnitude,
                                                     std::int64_t power) {
  if constexpr (std::is_same_v<Target, float>) {
    auto bits = __float_as_uint(magnitude);
    const auto exponent = static_cast<std::int64_t>((bits >> 23) & 0xffu);
    const auto adjusted = exponent + power;
    if (adjusted > 0 && adjusted < 0xff) {
      bits = (bits & 0x807fffffu) |
             (static_cast<std::uint32_t>(adjusted) << 23);
      return __uint_as_float(bits);
    }
    return ::ldexpf(magnitude, static_cast<int>(power));
  } else {
    auto high = static_cast<std::uint32_t>(__double2hiint(magnitude));
    const auto low = __double2loint(magnitude);
    const auto exponent = static_cast<std::int64_t>((high >> 20) & 0x7ffu);
    const auto adjusted = exponent + power;
    if (adjusted > 0 && adjusted < 0x7ff) {
      high = (high & 0x800fffffu) |
             (static_cast<std::uint32_t>(adjusted) << 20);
      return __hiloint2double(static_cast<int>(high), low);
    }
    return ::ldexp(magnitude, static_cast<int>(power));
  }
}

struct split_log_code {
  std::int64_t integer{};
  std::uint32_t fraction{};
};

template <typename Format>
__host__ __device__ __forceinline__ split_log_code
split_log(std::int64_t code) {
  constexpr auto scale = std::int64_t{1} << Format::log_fraction_bits;
  auto integer = code / scale;
  auto remainder = code % scale;
  if (remainder < 0) {
    remainder += scale;
    --integer;
  }
  return {integer, static_cast<std::uint32_t>(remainder)};
}

template <typename Target>
__device__ __forceinline__ Target approximate_ex2(float exponent) {
  float magnitude{};
#if defined(__CUDA_ARCH__)
  asm("ex2.approx.f32 %0, %1;" : "=f"(magnitude) : "f"(exponent));
#else
  magnitude = ::exp2f(exponent);
#endif
  return static_cast<Target>(magnitude);
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__device__ __forceinline__ compute_t<Compute>
decode_log(std::int64_t code, bool negative,
           decoder_context<Format, Compute, Decoder> context) {
  using target = compute_t<Compute>;
  target magnitude{};
  if constexpr (Decoder == decoder_kind::reference_exp2) {
    constexpr target scale =
        static_cast<target>(std::uint64_t{1} << Format::log_fraction_bits);
    if constexpr (Compute == compute_kind::fp32) {
      magnitude = ::exp2f(static_cast<float>(code) / scale);
    } else {
      magnitude = ::exp2(static_cast<double>(code) / scale);
    }
  } else if constexpr (Decoder == decoder_kind::ex2_approx) {
    constexpr float scale =
        static_cast<float>(std::uint64_t{1} << Format::log_fraction_bits);
    magnitude = approximate_ex2<target>(static_cast<float>(code) / scale);
  } else if constexpr (Decoder == decoder_kind::fraction_lut_global ||
                       Decoder == decoder_kind::fraction_lut_shared ||
                       Decoder == decoder_kind::fraction_lut_warp) {
    const auto split = split_log<Format>(code);
    if constexpr (Decoder == decoder_kind::fraction_lut_warp) {
      magnitude =
          warp_lookup(context.warp_entry, split.fraction, context.table);
    } else {
      magnitude = context.table[split.fraction];
    }
    magnitude = apply_power_of_two(magnitude, split.integer);
  } else if constexpr (Decoder == decoder_kind::split_linear ||
                       Decoder == decoder_kind::split_quadratic ||
                       Decoder == decoder_kind::split_cubic) {
    constexpr int coarse_bits = Format::log_fraction_bits < 8
                                    ? Format::log_fraction_bits
                                    : 8;
    constexpr int residual_bits = Format::log_fraction_bits - coarse_bits;
    const auto split = split_log<Format>(code);
    const auto coarse = residual_bits == 0
                            ? split.fraction
                            : split.fraction >> residual_bits;
    const auto residual_mask = residual_bits == 0
                                   ? 0u
                                   : (std::uint32_t{1} << residual_bits) - 1u;
    const auto residual_raw = split.fraction & residual_mask;
    constexpr target scale =
        static_cast<target>(std::uint64_t{1} << Format::log_fraction_bits);
    const auto residual = static_cast<target>(residual_raw) / scale;
    constexpr target ln2 = static_cast<target>(0.6931471805599453094172321215);
    const auto x = ln2 * residual;
    auto polynomial = target{1} + x;
    if constexpr (Decoder == decoder_kind::split_quadratic ||
                  Decoder == decoder_kind::split_cubic) {
      polynomial += target{0.5} * x * x;
    }
    if constexpr (Decoder == decoder_kind::split_cubic) {
      polynomial += static_cast<target>(1.0 / 6.0) * x * x * x;
    }
    magnitude = apply_power_of_two(context.table[coarse] * polynomial,
                                   split.integer);
  } else {
    static_assert(!is_full_decoder_v<Decoder> && !is_pair_decoder_v<Decoder>,
                  "full and pair LUTs do not decode widened product logs");
  }
  return negative ? -magnitude : magnitude;
}

template <typename Format, compute_kind Compute, decoder_kind Decoder>
__device__ __forceinline__ compute_t<Compute>
decode_raw(std::uint32_t raw,
           decoder_context<Format, Compute, Decoder> context) {
  using target = compute_t<Compute>;
  raw &= lns::raw_mask<Format>();
  if constexpr (is_full_decoder_v<Decoder>) {
    return context.table[raw];
  } else {
    if (lns::is_zero<Format>(raw)) {
      return target{0};
    }
    if (lns::is_nan<Format>(raw)) {
      if constexpr (Compute == compute_kind::fp32) {
        return __int_as_float(0x7fc00000);
      } else {
        return __hiloint2double(0x7ff80000, 0);
      }
    }
    return decode_log<Format, Compute, Decoder>(
        lns::log_code<Format>(raw), lns::sign<Format>(raw), context);
  }
}

template <typename Format, compute_kind Compute, decoder_kind Decoder,
          multiply_kind Multiply>
__device__ __forceinline__ compute_t<Compute>
multiply(std::uint32_t left, std::uint32_t right,
         decoder_context<Format, Compute, Decoder> context) {
  using target = compute_t<Compute>;
  if constexpr (Multiply == multiply_kind::ordinary) {
    static_assert(!is_pair_decoder_v<Decoder>);
    return decode_raw<Format, Compute, Decoder>(left, context) *
           decode_raw<Format, Compute, Decoder>(right, context);
  } else if constexpr (is_pair_decoder_v<Decoder>) {
    constexpr auto shift = Format::total_bits;
    const auto index = (left & lns::raw_mask<Format>()) |
                       ((right & lns::raw_mask<Format>()) << shift);
    return context.table[index];
  } else {
    static_assert(!is_full_decoder_v<Decoder>);
    const auto product = lns::multiply_codes<Format>(left, right);
    if (product.nan) {
      if constexpr (Compute == compute_kind::fp32) {
        return __int_as_float(0x7fc00000);
      } else {
        return __hiloint2double(0x7ff80000, 0);
      }
    }
    if (product.zero) {
      return product.negative ? -target{0} : target{0};
    }
    return decode_log<Format, Compute, Decoder>(product.log,
                                                product.negative, context);
  }
}

} // namespace aut::lns_strategy

#endif // ACCESSOR_UNIVERSAL_TEST_LNS_DECODER_STRATEGIES_CUH_
