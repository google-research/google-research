class Int8QueryTile {
 public:
  static constexpr size_t kPoints = 1;
  static constexpr size_t kDims = 2;

  SCANN_SIMD_INLINE void Load(const int16_t* ptr) {
    data_ = Simd<int16_t>(
        Simd<int32_t>::Broadcast(*reinterpret_cast<const int32_t*>(ptr)));
  }

 private:
  friend class Int32AccumulatorTile;
  Simd<int16_t> data_;
};

class Int8QueryExpander {
 public:
  void Ensure(size_t size) {
    if (size_ < size) {
      buffer_ =
          std::make_unique<int16_t[]>(size + Simd<int8_t>::kRegisterBytes);
      size_ = size;
    }
  }

  using ExpandedQueryT = int16_t;
  SCANN_SIMD_INLINE const ExpandedQueryT* Expand(
      ConstSpan<int8_t> queries) const {
    DCHECK_LE(queries.size(), size_);
    for (size_t pos :
         SeqWithStride<Simd<int8_t>::kNumElements>(queries.size())) {
      Simd<int8_t>::Load(queries.data() + pos)
          .ExpandTo<int16_t>()
          .Store(buffer_.get() + pos);
    }
    return buffer_.get();
  }

 private:
  std::unique_ptr<int16_t[]> buffer_;
  size_t size_ = 0;
};

class Int8DatabaseTile {
 public:
  static constexpr size_t kPoints = Simd<int32_t>::kNumElements;
  static constexpr size_t kDims = 2;

  SCANN_SIMD_INLINE void Load(const int8_t* ptr) { LoadAndExpand(data_, ptr); }

 private:
  SCANN_SIMD_INLINE void LoadAndExpand(Sse4<int16_t>& result,
                                       const int8_t* ptr) {
    result = _mm_cvtepi8_epi16(_mm_loadu_si64(ptr));
  }
  SCANN_SIMD_INLINE void LoadAndExpand(Avx2<int16_t>& result,
                                       const int8_t* ptr) {
    result = _mm256_cvtepi8_epi16(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr)));
  }
  SCANN_SIMD_INLINE void LoadAndExpand(Avx512<int16_t>& result,
                                       const int8_t* ptr) {
    result = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr)));
  }

  friend class Int32AccumulatorTile;
  Simd<int16_t> data_;
};

class Int32AccumulatorTile {
 public:
  SCANN_SIMD_INLINE Int32AccumulatorTile() = default;

  SCANN_SIMD_INLINE void AccumulateDotProducts(
      const Int8QueryTile& query, const Int8DatabaseTile& database) {
    data_ += MultiplyAndAddPacked(query.data_, database.data_);
  }

  using ResultType = Simd<int32_t>;
  SCANN_SIMD_INLINE void GetResult(ResultType& result,
                                   const int32_t* query_sums) {
    result = data_;
  }

 private:
  SCANN_SIMD_INLINE Sse4<int32_t> MultiplyAndAddPacked(const Sse4<int16_t>& a,
                                                       const Sse4<int16_t>& b) {
    return _mm_madd_epi16(*a, *b);
  }
  SCANN_SIMD_INLINE Avx2<int32_t> MultiplyAndAddPacked(const Avx2<int16_t>& a,
                                                       const Avx2<int16_t>& b) {
    return _mm256_madd_epi16(*a, *b);
  }
  SCANN_SIMD_INLINE Avx512<int32_t> MultiplyAndAddPacked(
      const Avx512<int16_t>& a, const Avx512<int16_t>& b) {
    return _mm512_madd_epi16(*a, *b);
  }

  Simd<int32_t> data_{Zeros()};
};
