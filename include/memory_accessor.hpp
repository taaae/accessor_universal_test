#ifndef ACCESSOR_UNIVERSAL_TEST_MEMORY_ACCESSOR_HPP_
#define ACCESSOR_UNIVERSAL_TEST_MEMORY_ACCESSOR_HPP_

#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
#define AUT_ACCESSOR_HD __host__ __device__
#define AUT_ACCESSOR_INLINE __forceinline__
#else
#define AUT_ACCESSOR_HD
#define AUT_ACCESSOR_INLINE inline
#endif

namespace aut {

/**
 * A non-owning read accessor for a contiguous one-dimensional array.
 *
 * The separation of arithmetic_type and storage_type follows Ginkgo's
 * reduced-storage accessor: values are stored as StorageType and converted to
 * ArithmeticType when read. This deliberately omits multidimensional layout,
 * extents, strides, slicing, and writable proxy references.
 */
template <typename ArithmeticType, typename StorageType> class memory_accessor {
public:
  using arithmetic_type = std::remove_cv_t<ArithmeticType>;
  using storage_type = std::remove_cv_t<StorageType>;
  using pointer = const storage_type *;

  static_assert(std::is_arithmetic_v<arithmetic_type>,
                "ArithmeticType must be an arithmetic type");
  static_assert(std::is_arithmetic_v<storage_type>,
                "StorageType must be an arithmetic type");

  constexpr AUT_ACCESSOR_HD explicit memory_accessor(pointer data) noexcept
      : data_{data} {}

  AUT_ACCESSOR_HD AUT_ACCESSOR_INLINE arithmetic_type
  operator[](std::size_t index) const noexcept {
    return static_cast<arithmetic_type>(data_[index]);
  }

  constexpr AUT_ACCESSOR_HD pointer data() const noexcept { return data_; }

private:
  pointer data_;
};

static_assert(sizeof(memory_accessor<float, float>) == sizeof(const float *));
static_assert(sizeof(memory_accessor<double, float>) == sizeof(const float *));
static_assert(std::is_trivially_copyable_v<memory_accessor<float, float>>);
static_assert(std::is_trivially_copyable_v<memory_accessor<double, float>>);
static_assert(std::is_standard_layout_v<memory_accessor<float, float>>);
static_assert(std::is_standard_layout_v<memory_accessor<double, float>>);

} // namespace aut

#undef AUT_ACCESSOR_HD
#undef AUT_ACCESSOR_INLINE

#endif
