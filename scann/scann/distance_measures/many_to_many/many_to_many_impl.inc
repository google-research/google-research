

#ifdef __aarch64__
constexpr bool kArmAggressiveUnrolling = true;
#else
constexpr bool kArmAggressiveUnrolling = false;
#endif

template <typename T, size_t... kNumElements>
SCANN_SIMD_INLINE Simd<T, index_sequence_sum_v<kNumElements...>> SimdConcat(
    const Simd<T, kNumElements>&... inputs) {
  Simd<T, index_sequence_sum_v<kNumElements...>> ret;

  size_t idx = 0;
  auto assign_one_input = [&](auto input) SCANN_SIMD_INLINE_LAMBDA {
    for (size_t jj : Seq(decltype(input)::kNumElements)) {
      ret[idx++] = input[jj];
    }
  };
  (assign_one_input(inputs), ...);

  return ret;
}

template <typename FloatT>
SCANN_SIMD_INLINE void ExpandPretransposedFP8BlockImpl(
    ConstSpan<int8_t> block, size_t dimensionality, size_t n_to_transpose,
    const float* __restrict__ inverse_multipliers_or_null,
    FloatT* __restrict__ transposed_storage) {
  constexpr size_t kElementsPerRegister = Simd<FloatT>::kElementsPerRegister;
  DCHECK_EQ(n_to_transpose * dimensionality, block.size());
  if (n_to_transpose == kElementsPerRegister && !inverse_multipliers_or_null) {
#pragma clang loop vectorize_width(kElementsPerRegister)
    for (size_t i : Seq(kElementsPerRegister * dimensionality)) {
      transposed_storage[i] = static_cast<float>(block[i]);
    }
    return;
  }

  if (n_to_transpose == kElementsPerRegister) {
    const int8_t* __restrict__ src = block.data();

#ifdef __x86_64__

    if constexpr (IsSame<Simd<FloatT>, Avx2<float>>()) {
      static_assert(kElementsPerRegister == 8);
      for (size_t dim_idx : Seq(dimensionality)) {
        __m256 inv_multiplier_simd = _mm256_broadcast_ss((

            inverse_multipliers_or_null + dim_idx));
        __m128i int8s = _mm_loadl_pi(_mm_setzero_si128(),
                                     reinterpret_cast<const __m64*>(src));
        __m256i int32s = _mm256_cvtepi8_epi32(int8s);
        __m256 floats = _mm256_cvtepi32_ps(int32s) * inv_multiplier_simd;

        _mm256_store_ps(transposed_storage, floats);
        transposed_storage += kElementsPerRegister;
        src += kElementsPerRegister;
      }
      return;
    }
    if constexpr (IsSame<Simd<FloatT>, Avx512<float>>()) {
      static_assert(kElementsPerRegister == 16);
      for (size_t dim_idx : Seq(dimensionality)) {
        __m512 inv_multiplier_simd = _mm512_broadcast_f32x8(
            _mm256_broadcast_ss(inverse_multipliers_or_null + dim_idx));
        __m128i int8s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        __m512i int32s = _mm512_cvtepi8_epi32(int8s);
        __m512 floats = _mm512_cvtepi32_ps(int32s) * inv_multiplier_simd;
        _mm512_store_ps(transposed_storage, floats);
        transposed_storage += kElementsPerRegister;
        src += kElementsPerRegister;
      }
      return;
    }
#endif

    for (size_t dim_idx : Seq(dimensionality)) {
      const float inv_multiplier = inverse_multipliers_or_null[dim_idx];
      for (size_t dp_idx : Seq(kElementsPerRegister)) {
        transposed_storage[dp_idx] = src[dp_idx] * inv_multiplier;
      }
      transposed_storage += kElementsPerRegister;
      src += kElementsPerRegister;
    }
    return;
  }

  const int8_t* __restrict__ src = block.data();
  for (size_t dim_idx : Seq(dimensionality)) {
    const float inv_multiplier = inverse_multipliers_or_null
                                     ? inverse_multipliers_or_null[dim_idx]
                                     : 1.0f;
    for (size_t dp_idx : Seq(n_to_transpose)) {
      transposed_storage[dp_idx] = src[dp_idx] * inv_multiplier;
    }
    transposed_storage += kElementsPerRegister;
    src += n_to_transpose;
  }
}

template <bool kIsSquaredL2, typename FloatT>
class M2MTransposer {
 public:
  static constexpr size_t kElementsPerRegister =
      Simd<FloatT>::kElementsPerRegister;

  static unique_ptr<M2MTransposer> New(const size_t dimensionality,
                                       const size_t result_entries) {
    const size_t transposed_entries =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    const size_t total_entries = 2 * transposed_entries + result_entries;

    constexpr size_t kCacheLineBytes = 64;
    FloatT* storage = static_cast<FloatT*>(
        aligned_malloc(total_entries * sizeof(FloatT), kCacheLineBytes));
    std::fill(storage, storage + total_entries,
              numeric_limits<FloatT>::quiet_NaN());
    return absl::WrapUnique(reinterpret_cast<M2MTransposer*>(storage));
  }

  static void operator delete(void* ptr) { aligned_free(ptr); }

  FloatT* __restrict__ GetTransposedPtr0(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    FloatT* storage = &first_buffer_entry_;
    return storage + 0 * transposed_sz +
           (kIsSquaredL2 ? kElementsPerRegister : 0);
  }

  FloatT* __restrict__ GetTransposedPtr1(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    FloatT* storage = &first_buffer_entry_;
    return storage + 1 * transposed_sz +
           (kIsSquaredL2 ? kElementsPerRegister : 0);
  }

  FloatT* __restrict__ GetResultsPtr(const size_t dimensionality) {
    const size_t transposed_sz =
        (dimensionality + kIsSquaredL2) * kElementsPerRegister;
    FloatT* storage = &first_buffer_entry_;
    return storage + 2 * transposed_sz;
  }

  SCANN_SIMD_INLINE void TransposeDatabaseBlock(const size_t dimensionality,
                                                const FloatT* database,
                                                size_t first_dp_idx,
                                                size_t n_to_transpose) {
    FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
    FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);

    if (ABSL_PREDICT_FALSE(n_to_transpose <= kElementsPerRegister)) {
      TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                 n_to_transpose, transposed_ptr0);
    } else {
      TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                 kElementsPerRegister, transposed_ptr0);
      TransposeDatabaseBlockImpl(
          dimensionality, database, first_dp_idx + kElementsPerRegister,
          n_to_transpose - kElementsPerRegister, transposed_ptr1);
    }
    if constexpr (kIsSquaredL2) {
      AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
    }
  }

  SCANN_SIMD_INLINE static void TransposeDatabaseBlockImpl(
      const size_t dimensionality, const FloatT* database, size_t first_dp_idx,
      size_t n_to_transpose, FloatT* __restrict__ transposed_storage) {
    DCHECK_LE(n_to_transpose, kElementsPerRegister);
    size_t j = 0;
    for (; j + 4 <= n_to_transpose; j += 4) {
      const FloatT* database_ptr =
          database + (first_dp_idx + j) * dimensionality;
      FloatT* __restrict__ dest = transposed_storage + j;

      constexpr size_t kCacheLineElements = 64 / sizeof(FloatT);
      const FloatT* prefetch = database_ptr + 0 * dimensionality;
      const FloatT* prefetch_end = database_ptr + 4 * dimensionality;
      for (; prefetch < prefetch_end; prefetch += kCacheLineElements) {
        absl::PrefetchToLocalCache(prefetch);
      }

      const FloatT* ptr_begin = database_ptr + 1 * dimensionality;
      const FloatT* ptr_end = database_ptr + 2 * dimensionality;
      for (const FloatT* ptr = ptr_begin; ptr != ptr_end;
           ++ptr, dest += kElementsPerRegister) {
        dest[0] = *(ptr - dimensionality);
        dest[1] = ptr[0 * dimensionality];
        dest[2] = ptr[1 * dimensionality];
        dest[3] = ptr[2 * dimensionality];
      }
    }
    for (; j < n_to_transpose; ++j) {
      const FloatT* untransposed0 =
          database + (first_dp_idx + j) * dimensionality;
      FloatT* dest = transposed_storage + j;
      for (size_t dim_idx = 0; dim_idx < dimensionality;
           ++dim_idx, dest += kElementsPerRegister) {
        dest[0] = untransposed0[dim_idx];
      }
    }
    DCHECK_EQ(j, n_to_transpose);
  }

  SCANN_SIMD_INLINE void ExpandPretransposedFP8Block(
      const FP8SimdBlockTransposedDatabase& fp8_db, size_t first_dp_idx,
      size_t n_to_transpose) {
    DCHECK_EQ(kElementsPerRegister, fp8_db.simd_block_size());
    const size_t dimensionality = fp8_db.dimensionality();
    const size_t first_block_idx = first_dp_idx / kElementsPerRegister;
    FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
    FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);
    if (ABSL_PREDICT_FALSE(n_to_transpose <= kElementsPerRegister)) {
      ExpandPretransposedFP8BlockImpl(
          fp8_db.GetBlock(first_block_idx), dimensionality, n_to_transpose,
          fp8_db.inverse_fp8_multipliers().data(), transposed_ptr0);
    } else {
      ExpandPretransposedFP8BlockImpl(fp8_db.GetBlock(first_block_idx),
                                      dimensionality, kElementsPerRegister,
                                      fp8_db.inverse_fp8_multipliers().data(),
                                      transposed_ptr0);
      ExpandPretransposedFP8BlockImpl(
          fp8_db.GetBlock(first_block_idx + 1), dimensionality,
          n_to_transpose - kElementsPerRegister,
          fp8_db.inverse_fp8_multipliers().data(), transposed_ptr1);
    }
    if constexpr (kIsSquaredL2) {
      AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
    }
  }

 private:
  SCANN_SIMD_INLINE static void AugmentWithL2Norms(
      FloatT* __restrict__ transposed_ptr0,
      FloatT* __restrict__ transposed_ptr1, size_t dimensionality) {
    Simd<FloatT> norm0 = Zeros();
    Simd<FloatT> norm1 = Zeros();
    Simd<FloatT> two = 2.0;
    for (size_t dim : Seq(dimensionality)) {
      auto transposed_simd0 =
          Simd<FloatT>::Load(transposed_ptr0 + dim * kElementsPerRegister);
      auto transposed_simd1 =
          Simd<FloatT>::Load(transposed_ptr1 + dim * kElementsPerRegister);
      FusedMultiplySubtract(transposed_simd0, transposed_simd0, &norm0);
      FusedMultiplySubtract(transposed_simd1, transposed_simd1, &norm1);
      (transposed_simd0 * two)
          .Store(transposed_ptr0 + dim * kElementsPerRegister);
      (transposed_simd1 * two)
          .Store(transposed_ptr1 + dim * kElementsPerRegister);
    }
    auto neg1 = Simd<FloatT>::Broadcast(-1.0);
    (norm0 * neg1).Store(transposed_ptr0 - kElementsPerRegister);
    (norm1 * neg1).Store(transposed_ptr1 - kElementsPerRegister);
  }

  struct alignas(kElementsPerRegister * sizeof(FloatT)) {
    FloatT first_buffer_entry_;
  };
};

template <typename FloatT>
class DenseManyToManyTransposedBase : public VirtualDestructor {
 public:
  static constexpr size_t kElementsPerRegister =
      Simd<FloatT>::kElementsPerRegister;
  constexpr static size_t kSmallQueryStride = [] {
    if (kPlatformGeneration == kSapphireRapidsAmx) {
      return 32;
    } else if (kArmAggressiveUnrolling) {
      return 10;
    } else {
      return 6;
    }
  }();
  constexpr static size_t kResultsSize =
      2 * kSmallQueryStride * kElementsPerRegister;

  SCANN_INLINE DenseManyToManyTransposedBase(
      DefaultDenseDatasetView<FloatT> queries, size_t num_datapoints,
      ThreadPool* pool)
      : dimensionality_(queries.dimensionality()),
        queries_(queries.GetPtr(0)),
        num_queries_(queries.size()),
        num_datapoints_(num_datapoints),
        pool_(pool) {
    DCHECK_GE(num_queries_, 1);
    DCHECK_GE(num_datapoints_, 1);
    if (num_queries_ >= 2) {
      DCHECK_EQ(queries_ + dimensionality_, queries.GetPtr(1));
    }
  }

  SCANN_INLINE DenseManyToManyTransposedBase(
      const SFP8SimdBlockTransposedDatabase& queries, size_t num_datapoints,
      ThreadPool* pool)
      : dimensionality_(queries.dimensionality()),
        queries_(nullptr),
        num_queries_(queries.size()),
        num_datapoints_(num_datapoints),
        pool_(pool) {
    DCHECK_GE(num_queries_, 1);
    DCHECK_GE(num_datapoints_, 1);
  }

  SCANN_INLINE void TopLevelBatch() {
    const size_t dimensionality = dimensionality_;
    const size_t num_queries = num_queries_;
    const size_t num_datapoints = num_datapoints_;
    ThreadPool* pool = pool_;

    const size_t q_stride_est1 =
        std::max<size_t>(1, (1 << 19) / (dimensionality * sizeof(FloatT)));

    const size_t q_stride_est2 =
        num_queries / DivRoundUp(num_queries, q_stride_est1);

    const size_t q_stride = NextMultipleOf(q_stride_est2, kSmallQueryStride);

    for (size_t q_idx = 0; q_idx < num_queries; q_idx += q_stride) {
      const size_t q_batch_size = std::min(q_stride, num_queries - q_idx);

      constexpr size_t kDatabaseStride = 2 * kElementsPerRegister;
      const size_t num_db_blocks = DivRoundUp(num_datapoints, kDatabaseStride);
      ParallelFor<16>(Seq(num_db_blocks), pool, [&](size_t block_idx) {
        const size_t first_dp_idx = block_idx * kDatabaseStride;
        const size_t dp_batch_size =
            std::min(kDatabaseStride, num_datapoints - first_dp_idx);

        MidLevelBatch(q_idx, q_batch_size, first_dp_idx, dp_batch_size);
      });
    }
  }

  virtual void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                             size_t first_dp_idx, size_t num_datapoints) = 0;

 protected:
  template <typename F>
  static const F* DatabaseToPtr(const DenseDataset<F>& db) {
    return db[0].values();
  }

  static const FP8SimdBlockTransposedDatabase* DatabaseToPtr(
      const FP8SimdBlockTransposedDatabase& db) {
    return &db;
  }

  template <size_t kNumQueries, typename AccumulatorT, typename CallbackT>
  SCANN_SIMD_INLINE static void PassResultsToCallback(
      const AccumulatorT& accumulators, CallbackT& callback, size_t first_q_idx,
      size_t first_dp_idx, size_t num_datapoints) {
    if constexpr (IsOptimizedCallback<CallbackT>::value) {
      if (ABSL_PREDICT_TRUE(num_datapoints == 2 * kElementsPerRegister)) {
        for (size_t j : Seq(kNumQueries)) {
          const size_t query_idx = first_q_idx + j;
          callback.InvokeOptimized(accumulators[j], first_dp_idx, query_idx);
        }
        return;
      }
    }

    auto results = accumulators.Store();
    for (size_t j : Seq(kNumQueries)) {
      auto query_results = MakeMutableSpan(results[j].data(), num_datapoints);
      const size_t query_idx = first_q_idx + j;
      callback(query_results, first_dp_idx, query_idx);
    }
  }

  const size_t dimensionality_;

  const FloatT* queries_;
  const size_t num_queries_;
  const size_t num_datapoints_;

 private:
  ThreadPool* pool_;
};

template <bool kIsSquaredL2, bool kIsPretransposedFixed8, typename CallbackT,
          typename FloatT>
class DenseManyToManyTransposed final
    : public DenseManyToManyTransposedBase<FloatT> {
 public:
  using DatabaseT = typename std::conditional<kIsPretransposedFixed8,
                                              FP8SimdBlockTransposedDatabase,
                                              DenseDataset<FloatT>>::type;
  using DatabasePtrT =
      typename std::conditional<kIsPretransposedFixed8,
                                const FP8SimdBlockTransposedDatabase*,
                                const FloatT*>::type;

  using Transposer = M2MTransposer<kIsSquaredL2, FloatT>;

  using Base = DenseManyToManyTransposedBase<FloatT>;
  using Base::kElementsPerRegister;
  using Base::kResultsSize;
  using Base::kSmallQueryStride;

  SCANN_INLINE DenseManyToManyTransposed(
      DefaultDenseDatasetView<FloatT> queries_view, const DatabaseT& database,
      ThreadPool* pool, CallbackT callback)
      : Base(queries_view, database.size(), pool),
        database_(Base::DatabaseToPtr(database)),
        callback_(std::move(callback)) {
    CHECK_EQ(queries_view.dimensionality(), database.dimensionality());
    if constexpr (!kIsPretransposedFixed8) {
      if (this->num_datapoints_ >= 2) {
        DCHECK_EQ(this->database_ + this->dimensionality_,
                  database[1].values());
      }
    }

    if constexpr (kIsSquaredL2) {
      const size_t dimensionality = this->dimensionality_;
      const size_t num_queries = this->num_queries_;
      const FloatT* queries = this->queries_;
      query_norms_.reset(new FloatT[num_queries]);
      FloatT* query_norms = query_norms_.get();
      ParallelFor<16>(Seq(num_queries), pool, [&](size_t q_idx) {
        const FloatT* q_ptr = queries + q_idx * dimensionality;
        query_norms[q_idx] =
            SquaredL2Norm(MakeDatapointPtr(q_ptr, dimensionality));
      });
    }
  }

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) final {
    const size_t dimensionality = this->dimensionality_;
    const size_t q_idx_end = first_q_idx + num_queries;

    thread_local size_t allocated_dimensionality = 0;
    thread_local unique_ptr<Transposer> transposer_storage;
    if (allocated_dimensionality < dimensionality) {
      transposer_storage = Transposer::New(dimensionality, kResultsSize);
      allocated_dimensionality = dimensionality;
    }

    Transposer* transposer = transposer_storage.get();
    if constexpr (kIsPretransposedFixed8) {
      transposer->ExpandPretransposedFP8Block(*this->database_, first_dp_idx,
                                              num_datapoints);
    } else {
      transposer->TransposeDatabaseBlock(dimensionality, this->database_,
                                         first_dp_idx, num_datapoints);
    }

    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = this->queries_ + first_q_idx * dimensionality;
    if constexpr (kIsSquaredL2) {
      args.query_norms = this->query_norms_.get() + first_q_idx;
    } else {
      args.query_norms = nullptr;
    }
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.transposer = transposer;
    args.callback = &this->callback_;

    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
      BottomLevelBatch<kSmallQueryStride>(args);
      args.first_q_idx += kSmallQueryStride;
      if (kIsSquaredL2) {
        args.query_norms += kSmallQueryStride;
      }
      args.queries += kSmallQueryStride * dimensionality;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                         BottomLevelBatch, args);
  }

  struct BottomLevelBatchArgs {
    size_t dimensionality;
    const FloatT* queries;
    const FloatT* query_norms;
    size_t first_q_idx;
    size_t first_dp_idx;
    size_t num_datapoints;
    Transposer* transposer;
    CallbackT* callback;
  };

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void BottomLevelBatch(BottomLevelBatchArgs args) {
    const size_t dimensionality = args.dimensionality;
    const FloatT* queries = args.queries;
    const FloatT* query_norms = args.query_norms;
    const size_t first_q_idx = args.first_q_idx;
    const size_t first_dp_idx = args.first_dp_idx;
    const size_t num_datapoints = args.num_datapoints;
    const FloatT* transposed_ptr0 =
        args.transposer->GetTransposedPtr0(dimensionality);
    const FloatT* transposed_ptr1 =
        args.transposer->GetTransposedPtr1(dimensionality);
    CallbackT& callback = *args.callback;

    const FloatT* volatile query_ptrs_vol[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs_vol[j] = queries + j * dimensionality;
    }
    const FloatT* query_ptrs[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs[j] = query_ptrs_vol[j];
    }

    auto accumulators = DoAccumulationTransposedTemplate<kNumQueries>(
        transposed_ptr0, transposed_ptr1, query_ptrs, query_norms,
        dimensionality);
    Base::template PassResultsToCallback<kNumQueries>(
        accumulators, callback, first_q_idx, first_dp_idx, num_datapoints);
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static Simd<FloatT, kNumQueries, 2>
  DoAccumulationTransposedTemplate(const FloatT* transposed_block0,
                                   const FloatT* transposed_block1,
                                   const FloatT** query_ptrs,
                                   const FloatT* query_norms,
                                   size_t dimensionality) {
    Simd<FloatT, kNumQueries, 2> accumulators;
    for (size_t j : Seq(kNumQueries)) {
      if constexpr (kIsSquaredL2) {
        auto query_norm = Simd<FloatT>::Broadcast(query_norms[j]);
        auto db_norms0 =
            Simd<FloatT>::Load(transposed_block0 - kElementsPerRegister);
        auto db_norms1 =
            Simd<FloatT>::Load(transposed_block1 - kElementsPerRegister);
        accumulators[j][0] = db_norms0 + query_norm;
        accumulators[j][1] = db_norms1 + query_norm;
      } else {
        accumulators[j] = Zeros();
      }
    }

    for (size_t dim : Seq(dimensionality)) {
      auto transposed_simd0 =
          Simd<FloatT>::Load(transposed_block0 + dim * kElementsPerRegister);
      auto transposed_simd1 =
          Simd<FloatT>::Load(transposed_block1 + dim * kElementsPerRegister);

      for (size_t j : Seq(kNumQueries)) {
        auto query_simd = Simd<FloatT>::Broadcast(query_ptrs[j][dim]);
        FusedMultiplySubtract(query_simd, transposed_simd0,
                              &accumulators[j][0]);
        FusedMultiplySubtract(query_simd, transposed_simd1,
                              &accumulators[j][1]);
      }
    }

    return accumulators;
  }

 private:
  const DatabasePtrT database_;
  CallbackT callback_;

  unique_ptr<FloatT[]> query_norms_;
};

template <bool kIsSquaredL2, typename CallbackT, typename FloatT>
SCANN_INLINE void DenseManyToManyTransposedImpl(
    DefaultDenseDatasetView<FloatT> queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  DenseManyToManyTransposed<kIsSquaredL2, false, CallbackT, FloatT>(
      queries, database, pool, std::move(callback))
      .TopLevelBatch();
}

template <typename CallbackT>
SCANN_INLINE void DenseManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());
  if (is_squared_l2) {
    DenseManyToManyTransposed<true, true, CallbackT, float>(
        queries, database, pool, std::move(callback))
        .TopLevelBatch();
  } else {
    DenseManyToManyTransposed<false, true, CallbackT, float>(
        queries, database, pool, std::move(callback))
        .TopLevelBatch();
  }
}

template <bool kIsPretransposedFixed8, typename CallbackT>
class DenseManyToManyOrthogonalityAmplified final
    : public DenseManyToManyTransposedBase<float> {
 public:
  using FloatT = float;
  using DatabaseT = typename std::conditional<kIsPretransposedFixed8,
                                              FP8SimdBlockTransposedDatabase,
                                              DenseDataset<FloatT>>::type;
  using DatabasePtrT =
      typename std::conditional<kIsPretransposedFixed8,
                                const FP8SimdBlockTransposedDatabase*,
                                const FloatT*>::type;
  using Transposer = M2MTransposer<false, float>;
  using Base = DenseManyToManyTransposedBase<float>;
  using Base::kElementsPerRegister;
  using Base::kResultsSize;

  static constexpr size_t kSmallQueryStride = Base::kSmallQueryStride / 2;

  SCANN_INLINE DenseManyToManyOrthogonalityAmplified(
      const DenseDataset<FloatT>& queries,
      const DenseDataset<FloatT>& normalized_residuals, const FloatT lambda,
      const DatabaseT& database, ThreadPool* pool, CallbackT callback)
      : Base(queries, database.size(), pool),
        database_(Base::DatabaseToPtr(database)),
        normalized_residuals_(Base::DatabaseToPtr(normalized_residuals)),
        lambda_(lambda),
        callback_(std::move(callback)) {
    CHECK_EQ(queries.dimensionality(), database.dimensionality());
    CHECK_EQ(normalized_residuals.dimensionality(), queries.dimensionality());
    CHECK_EQ(queries.size(), normalized_residuals.size());
    if constexpr (!kIsPretransposedFixed8) {
      if (this->num_datapoints_ >= 2) {
        DCHECK_EQ(this->database_ + this->dimensionality_,
                  database[1].values());
      }
    }
  }

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) final {
    const size_t dimensionality = this->dimensionality_;
    const size_t q_idx_end = first_q_idx + num_queries;

    thread_local size_t allocated_dimensionality = 0;
    thread_local unique_ptr<Transposer> transposer_storage;
    if (allocated_dimensionality < dimensionality) {
      transposer_storage = Transposer::New(dimensionality, kResultsSize);
      allocated_dimensionality = dimensionality;
    }

    Transposer* transposer = transposer_storage.get();
    if constexpr (kIsPretransposedFixed8) {
      transposer->ExpandPretransposedFP8Block(*this->database_, first_dp_idx,
                                              num_datapoints);
    } else {
      transposer->TransposeDatabaseBlock(dimensionality, this->database_,
                                         first_dp_idx, num_datapoints);
    }

    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = this->queries_ + first_q_idx * dimensionality;
    args.normalized_residuals =
        this->normalized_residuals_ + first_q_idx * dimensionality;
    args.lambda = this->lambda_;
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.transposer = transposer;
    args.callback = &this->callback_;

    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
      BottomLevelBatch<kSmallQueryStride>(args);
      args.first_q_idx += kSmallQueryStride;
      args.queries += kSmallQueryStride * dimensionality;
      args.normalized_residuals += kSmallQueryStride * dimensionality;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                         BottomLevelBatch, args);
  }

  struct BottomLevelBatchArgs {
    size_t dimensionality;
    const FloatT* queries;
    const FloatT* normalized_residuals;
    float lambda;
    size_t first_q_idx;
    size_t first_dp_idx;
    size_t num_datapoints;
    Transposer* transposer;
    CallbackT* callback;
  };

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void BottomLevelBatch(BottomLevelBatchArgs args) {
    const size_t dimensionality = args.dimensionality;
    const FloatT* queries = args.queries;
    const FloatT* normalized_residuals = args.normalized_residuals;
    const float lambda = args.lambda;
    const size_t first_q_idx = args.first_q_idx;
    const size_t first_dp_idx = args.first_dp_idx;
    const size_t num_datapoints = args.num_datapoints;
    const FloatT* transposed_ptr0 =
        args.transposer->GetTransposedPtr0(dimensionality);
    const FloatT* transposed_ptr1 =
        args.transposer->GetTransposedPtr1(dimensionality);
    CallbackT& callback = *args.callback;

    const FloatT* volatile query_ptrs_vol[kNumQueries];
    const FloatT* volatile residual_ptrs_vol[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs_vol[j] = queries + j * dimensionality;
      residual_ptrs_vol[j] = normalized_residuals + j * dimensionality;
    }
    const FloatT* query_ptrs[kNumQueries];
    const FloatT* residual_ptrs[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs[j] = query_ptrs_vol[j];
      residual_ptrs[j] = residual_ptrs_vol[j];
    }

    auto accumulators = DoAccumulationTransposedTemplate<kNumQueries>(
        transposed_ptr0, transposed_ptr1, query_ptrs, residual_ptrs, lambda,
        dimensionality);
    Base::template PassResultsToCallback<kNumQueries>(
        accumulators, callback, first_q_idx, first_dp_idx, num_datapoints);
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static Simd<FloatT, kNumQueries, 2>
  DoAccumulationTransposedTemplate(const FloatT* transposed_block0,
                                   const FloatT* transposed_block1,
                                   const FloatT** query_ptrs,
                                   const FloatT** residual_ptrs,
                                   const FloatT lambda, size_t dimensionality) {
    Simd<FloatT, kNumQueries, 2> term1_accumulators;
    Simd<FloatT, kNumQueries, 2> term2_accumulators;
    for (size_t j : Seq(kNumQueries)) {
      term1_accumulators[j] = Zeros();
      term2_accumulators[j] = Zeros();
    }

    Simd<FloatT, kNumQueries, 2> final_accumulators;
    for (size_t dim : Seq(dimensionality)) {
      auto transposed_simd0 =
          Simd<FloatT>::Load(transposed_block0 + dim * kElementsPerRegister);
      auto transposed_simd1 =
          Simd<FloatT>::Load(transposed_block1 + dim * kElementsPerRegister);

      for (size_t j : Seq(kNumQueries)) {
        Simd<FloatT> query_simd = query_ptrs[j][dim];
        Simd<FloatT> residual_simd = residual_ptrs[j][dim];

        Simd<FloatT> diff0 = query_simd - transposed_simd0;
        FusedMultiplyAdd(diff0, diff0, &term1_accumulators[j][0]);
        FusedMultiplyAdd(diff0, residual_simd, &term2_accumulators[j][0]);

        Simd<FloatT> diff1 = query_simd - transposed_simd1;
        FusedMultiplyAdd(diff1, diff1, &term1_accumulators[j][1]);
        FusedMultiplyAdd(diff1, residual_simd, &term2_accumulators[j][1]);
      }

      Simd<FloatT> lambda_vec(lambda);
      for (size_t j : Seq(kNumQueries)) {
        for (size_t k : Seq(2)) {
          final_accumulators[j][k] =
              term1_accumulators[j][k] +
              lambda_vec * term2_accumulators[j][k] * term2_accumulators[j][k];
        }
      }
    }
    return final_accumulators;
  }

 private:
  const DatabasePtrT database_;
  const FloatT* normalized_residuals_;
  const FloatT lambda_;
  CallbackT callback_;
};

template <typename DatabaseT, typename CallbackT>
void DenseManyToManyOrthogonalityAmplifiedImpl(
    const DenseDataset<float>& queries,
    const DenseDataset<float>& normalized_residuals, float lambda,
    const DatabaseT& database, ThreadPool* pool, CallbackT callback) {
  if (queries.empty() || database.empty()) return;
  constexpr bool kIsPretransposedFixed8 =
      std::is_same_v<DatabaseT, FP8SimdBlockTransposedDatabase>;
  DenseManyToManyOrthogonalityAmplified<kIsPretransposedFixed8, CallbackT>(
      queries, normalized_residuals, lambda, database, pool, callback)
      .TopLevelBatch();
}

template <bool kIsSquaredL2, typename FloatT>
class DenseManyToManyUntransposedBase : public VirtualDestructor {
 public:
  constexpr static size_t kElementsPerRegister =
      Simd<FloatT>::kElementsPerRegister;

  constexpr static size_t kFloatDivisor = sizeof(FloatT) / sizeof(float);

  constexpr static size_t kSmallQueryStride = kArmAggressiveUnrolling ? 10 : 5;

  constexpr static size_t kMaxDbChunk = 256;

  SCANN_INLINE DenseManyToManyUntransposedBase(
      DefaultDenseDatasetView<FloatT> queries,
      const DenseDataset<FloatT>& database, ThreadPool* pool)
      : dimensionality_(queries.dimensionality()),
        queries_(queries.GetPtr(0)),
        num_queries_(queries.size()),
        database_(database[0].values()),
        num_datapoints_(database.size()),
        pool_(pool) {
    CHECK_EQ(queries.dimensionality(), database.dimensionality());
    DCHECK_GE(num_queries_, 1);
    if (num_queries_ >= 2) {
      DCHECK_EQ(queries_ + dimensionality_, queries.GetPtr(1));
    }
    DCHECK_GE(num_datapoints_, 1);
    if (num_datapoints_ >= 2) {
      DCHECK_EQ(database_ + dimensionality_, database[1].values());
    }
  }

  SCANN_INLINE void TopLevelBatch() {
    const size_t dimensionality = dimensionality_;
    if (dimensionality > 256) return TopLevelBatchImpl<32 / kFloatDivisor>();
    if (dimensionality > 128) return TopLevelBatchImpl<64 / kFloatDivisor>();
    if (dimensionality > 64) return TopLevelBatchImpl<128 / kFloatDivisor>();
    return TopLevelBatchImpl<256 / kFloatDivisor>();
  }

  template <size_t kDatabaseStride>
  SCANN_INLINE void TopLevelBatchImpl() {
    const size_t num_queries = num_queries_;
    const size_t num_datapoints = num_datapoints_;
    ThreadPool* pool = pool_;

    constexpr size_t kBigQueryStride = 256 / kFloatDivisor;
    const size_t num_query_blocks = DivRoundUp(num_queries, kBigQueryStride);

    const size_t num_db_blocks = DivRoundUp(num_datapoints, kDatabaseStride);

    ParallelFor<1>(
        Seq(num_query_blocks * num_db_blocks), pool, [&](size_t block_idx) {
          const size_t query_block_idx = block_idx % num_query_blocks;
          const size_t first_q_idx = query_block_idx * kBigQueryStride;
          const size_t q_batch_size =
              std::min(kBigQueryStride, num_queries - first_q_idx);

          const size_t db_block_idx = block_idx / num_query_blocks;
          const size_t first_dp_idx = db_block_idx * kDatabaseStride;
          const size_t dp_batch_size =
              std::min(kDatabaseStride, num_datapoints - first_dp_idx);

          MidLevelBatch(first_q_idx, q_batch_size, first_dp_idx, dp_batch_size);
        });
  }

  virtual void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                             size_t first_dp_idx, size_t num_datapoints) = 0;

 private:
  const size_t dimensionality_;

  const FloatT* queries_;
  const size_t num_queries_;

  const FloatT* database_;
  const size_t num_datapoints_;

  ThreadPool* pool_;

  template <bool, typename, typename>
  friend class DenseManyToManyUntransposed;
};

template <bool kIsSquaredL2, typename CallbackT, typename FloatT>
class DenseManyToManyUntransposed final
    : public DenseManyToManyUntransposedBase<kIsSquaredL2, FloatT> {
 public:
  using Base = DenseManyToManyUntransposedBase<kIsSquaredL2, FloatT>;
  using Base::kElementsPerRegister;
  using Base::kMaxDbChunk;
  using Base::kSmallQueryStride;

  SCANN_INLINE DenseManyToManyUntransposed(
      DefaultDenseDatasetView<FloatT> queries,
      const DenseDataset<FloatT>& database, ThreadPool* pool,
      CallbackT callback)
      : Base(queries, database, pool), callback_(std::move(callback)) {}

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) final {
    const size_t dimensionality = this->dimensionality_;

    thread_local unique_ptr<FloatT[]> results;
    if (!results) {
      results.reset(new FloatT[kSmallQueryStride * kMaxDbChunk]);
    }

    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = this->queries_ + first_q_idx * dimensionality;
    args.database = this->database_ + first_dp_idx * dimensionality;
    args.results = results.get();
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.callback = &callback_;

    const size_t q_idx_end = first_q_idx + num_queries;
    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
      BottomLevelBatch<kSmallQueryStride>(args);
      args.queries += kSmallQueryStride * dimensionality;
      args.first_q_idx += kSmallQueryStride;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                         BottomLevelBatch, args);
  }

  struct BottomLevelBatchArgs {
    BottomLevelBatchArgs() {}
    const FloatT* queries;
    const FloatT* database;
    FloatT* __restrict__ results;
    uint32_t dimensionality;
    DatapointIndex first_q_idx;
    DatapointIndex first_dp_idx;
    DatapointIndex num_datapoints;
    CallbackT* callback;
  };

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void BottomLevelBatch(BottomLevelBatchArgs args) {
    const size_t dimensionality = args.dimensionality;
    const FloatT* queries = args.queries;
    const FloatT* database = args.database;
    FloatT* __restrict__ results = args.results;
    const DatapointIndex first_q_idx = args.first_q_idx;
    const DatapointIndex first_dp_idx = args.first_dp_idx;
    const DatapointIndex num_datapoints = args.num_datapoints;
    CallbackT& callback = *args.callback;

    array<const FloatT* volatile, kNumQueries> query_ptrs_vol;
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs_vol[j] = queries + j * dimensionality;
    }
    const FloatT* query_ptrs[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs[j] = query_ptrs_vol[j];
    }

    FloatT* __restrict__ result_ptrs[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      result_ptrs[j] = results + j * kMaxDbChunk;
    }

    const FloatT* dp0_ptr = database + 0 * dimensionality;
    const FloatT* dp0_end = database + num_datapoints * dimensionality;

    while (dp0_ptr < dp0_end) {
      const FloatT* dp1_ptr = dp0_ptr + dimensionality;

      if (dp1_ptr >= dp0_end) dp1_ptr = dp0_ptr;

      DoAccumulationUntransposedTemplate<kNumQueries>(
          dimensionality, query_ptrs, dp0_ptr, dp1_ptr, result_ptrs);

      dp0_ptr += 2 * dimensionality;
      for (size_t j : Seq(kNumQueries)) {
        result_ptrs[j] += 2;
      }
    }

    for (size_t j : Seq(kNumQueries)) {
      auto query_results =
          MakeMutableSpan(results + j * kMaxDbChunk, num_datapoints);
      callback(query_results, first_dp_idx, first_q_idx + j);
    }
  }

  template <typename T>
  SCANN_SIMD_INLINE static void Accumulate(T a, T b, T* acc) {
    if constexpr (kIsSquaredL2) {
      const T diff = a - b;
      FusedMultiplyAdd(diff, diff, acc);
    } else {
      FusedMultiplySubtract(a, b, acc);
    }
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void DoAccumulationUntransposedTemplate(
      size_t dimensionality, const FloatT** query_ptrs, const FloatT* dp0,
      const FloatT* dp1, FloatT* __restrict__* __restrict__ result_ptrs) {
    size_t n_blocks = dimensionality / kElementsPerRegister;
    size_t dim_idx = 0;
    if (n_blocks) {
      Simd<FloatT, kNumQueries> accumulators0 = Zeros();
      Simd<FloatT, kNumQueries> accumulators1 = Zeros();

      for (; n_blocks > 0; --n_blocks, dim_idx += kElementsPerRegister) {
        auto dp0_simd = Simd<FloatT>::Load(dp0 + dim_idx);
        auto dp1_simd = Simd<FloatT>::Load(dp1 + dim_idx);
        for (size_t j : Seq(kNumQueries)) {
          auto query_simd = Simd<FloatT>::Load(query_ptrs[j] + dim_idx);
          Accumulate(query_simd, dp0_simd, &accumulators0[j]);
          Accumulate(query_simd, dp1_simd, &accumulators1[j]);
        }
      }

      for (size_t j = 0; j < kNumQueries;) {
        if (j + 1 < kNumQueries) {
          HorizontalSum4X(accumulators0[j + 0], accumulators1[j + 0],
                          accumulators0[j + 1], accumulators1[j + 1],
                          &result_ptrs[j + 0][0], &result_ptrs[j + 0][1],
                          &result_ptrs[j + 1][0], &result_ptrs[j + 1][1]);
          j += 2;
        } else {
          HorizontalSum2X(accumulators0[j], accumulators1[j],
                          &result_ptrs[j][0], &result_ptrs[j][1]);
          break;
        }
      }
    } else {
      for (size_t j : Seq(kNumQueries)) {
        result_ptrs[j][0] = 0;
        result_ptrs[j][1] = 0;
      }
    }

    for (; dim_idx < dimensionality; ++dim_idx) {
      const FloatT dp0_val = dp0[dim_idx];
      const FloatT dp1_val = dp1[dim_idx];
      for (size_t j : Seq(kNumQueries)) {
        const FloatT query_val = query_ptrs[j][dim_idx];
        Accumulate(dp0_val, query_val, &result_ptrs[j][0]);
        Accumulate(dp1_val, query_val, &result_ptrs[j][1]);
      }
    }
  }

 private:
  CallbackT callback_;
};

template <bool kIsSquaredL2, typename CallbackT, typename FloatT>
SCANN_INLINE void DenseManyToManyUntransposedImpl(
    DefaultDenseDatasetView<FloatT> queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(), "");
  DenseManyToManyUntransposed<kIsSquaredL2, CallbackT, FloatT>(
      queries, database, pool, std::move(callback))
      .TopLevelBatch();
}

template <typename FloatT>
constexpr bool ShouldTranspose(size_t n_queries, size_t dimensionality) {
  const size_t n_dims = dimensionality;

  constexpr size_t kDoubleDivisor = sizeof(FloatT) / sizeof(float);

  constexpr size_t kMaxMidLevelQueryBatchSize = 256 / kDoubleDivisor;

  const size_t n_mid_level_batches =
      DivRoundUp(n_queries, kMaxMidLevelQueryBatchSize);

  constexpr float kUntransposedCostPerQuery = 2.0;

  constexpr float kUntransposedCostPerDimPerQuery = 0.025;

  constexpr float kRemainderCostPerDimPerQuery = 0.025;

  constexpr float kRemainderCostPerQuery = 2.0;

  const float transposed_cost_per_dp = dimensionality * n_mid_level_batches;

  const bool has_remainder = (n_dims % Simd<FloatT>::kElementsPerRegister) != 0;
  const float untransposed_postprocess_cost_per_dp =
      n_queries * kUntransposedCostPerQuery +
      n_queries * n_dims * kUntransposedCostPerDimPerQuery +
      n_queries * n_dims * kRemainderCostPerDimPerQuery * has_remainder +
      n_queries * kRemainderCostPerQuery * has_remainder;

  return transposed_cost_per_dp < untransposed_postprocess_cost_per_dp;
}

static_assert(kPlatformGeneration != kSkylakeAvx512 ||
              ShouldTranspose<float>(3, 128) == false);
static_assert(kPlatformGeneration != kSkylakeAvx512 ||
              ShouldTranspose<float>(24, 128) == false);
static_assert(kPlatformGeneration != kSkylakeAvx512 ||
              ShouldTranspose<float>(25, 128) == true);
static_assert(kPlatformGeneration != kSkylakeAvx512 ||
              ShouldTranspose<float>(300, 128) == true);

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl(
    const DistanceMeasure& dist, DefaultDenseDatasetView<FloatT> queries,
    const DenseDataset<FloatT>& database, ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float and double.");
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());

  if constexpr (Simd<FloatT>::kElementsPerRegister > 1) {
    bool should_transpose;
    if (absl::GetFlag(FLAGS_enable_scann_brute_force_determinism)) {
      should_transpose = true;
    } else {
      should_transpose =
          ShouldTranspose<FloatT>(queries.size(), queries.dimensionality());
    }
    if (should_transpose) {
      if (is_squared_l2) {
        return DenseManyToManyTransposedImpl<true>(queries, database, pool,
                                                   std::move(callback));
      } else {
        return DenseManyToManyTransposedImpl<false>(queries, database, pool,
                                                    std::move(callback));
      }
    }
  }

  if (is_squared_l2) {
    return DenseManyToManyUntransposedImpl<true>(queries, database, pool,
                                                 std::move(callback));
  } else {
    return DenseManyToManyUntransposedImpl<false>(queries, database, pool,
                                                  std::move(callback));
  }
}
