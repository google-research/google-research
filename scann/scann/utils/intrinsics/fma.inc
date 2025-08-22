

template <typename T>
SCANN_SIMD_INLINE T FusedMultiplyAdd(T a, T b, T acc) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    return _mm256_fmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx2<double>>()) {
    return _mm256_fmadd_pd(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<float>>()) {
    return _mm512_fmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<double>>()) {
    return _mm512_fmadd_pd(*a, *b, *acc);
  }
#if HWY_HAVE_CONSTEXPR_LANES
  if constexpr (IsSame<T, Highway<float>>() || IsSame<T, Highway<double>>()) {
    return hwy::HWY_NAMESPACE::MulAdd(*a, *b, *acc);
  }
#endif

  return acc + (a * b);
}

template <typename T>
SCANN_SIMD_INLINE void FusedMultiplyAdd(T a, T b, T* acc) {
  *acc = FusedMultiplyAdd(a, b, *acc);
}

template <typename T>
SCANN_SIMD_INLINE T FusedMultiplySubtract(T a, T b, T acc) {
  if constexpr (IsSame<T, Avx2<float>>()) {
    return _mm256_fnmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx2<double>>()) {
    return _mm256_fnmadd_pd(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<float>>()) {
    return _mm512_fnmadd_ps(*a, *b, *acc);
  }
  if constexpr (IsSame<T, Avx512<double>>()) {
    return _mm512_fnmadd_pd(*a, *b, *acc);
  }
#if HWY_HAVE_CONSTEXPR_LANES
  if constexpr (IsSame<T, Highway<float>>() || IsSame<T, Highway<double>>()) {
    return hwy::HWY_NAMESPACE::NegMulAdd(*a, *b, *acc);
  }
#endif

  return acc - (a * b);
}

template <typename T>
SCANN_SIMD_INLINE void FusedMultiplySubtract(T a, T b, T* acc) {
  *acc = FusedMultiplySubtract(a, b, *acc);
}
