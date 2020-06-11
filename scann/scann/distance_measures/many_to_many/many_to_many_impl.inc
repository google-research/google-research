

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
template <typename SimdFuncs, size_t kBlockSize, bool kIsSquaredL2,
          typename CallbackT, typename FloatT>
class DenseManyToManyTransposed {
  class Transposer {
   public:
    static unique_ptr<Transposer> New(const size_t dimensionality,
                                      const size_t result_entries) {
      const size_t transposed_entries =
          (dimensionality + kIsSquaredL2) * kBlockSize;
      const size_t total_entries = 2 * transposed_entries + result_entries;

      constexpr size_t kCacheLineBytes = 64;
      FloatT* storage = static_cast<FloatT*>(
          aligned_malloc(total_entries * sizeof(FloatT), kCacheLineBytes));
      std::fill(storage, storage + total_entries,
                numeric_limits<FloatT>::quiet_NaN());
      return absl::WrapUnique(reinterpret_cast<Transposer*>(storage));
    }

    static void operator delete(void* ptr) { aligned_free(ptr); }

    FloatT* __restrict__ GetTransposedPtr0(const size_t dimensionality) {
      const size_t transposed_sz = (dimensionality + kIsSquaredL2) * kBlockSize;
      FloatT* storage = &first_buffer_entry_;
      return storage + 0 * transposed_sz + (kIsSquaredL2 ? kBlockSize : 0);
    }

    FloatT* __restrict__ GetTransposedPtr1(const size_t dimensionality) {
      const size_t transposed_sz = (dimensionality + kIsSquaredL2) * kBlockSize;
      FloatT* storage = &first_buffer_entry_;
      return storage + 1 * transposed_sz + (kIsSquaredL2 ? kBlockSize : 0);
    }

    FloatT* __restrict__ GetResultsPtr(const size_t dimensionality) {
      const size_t transposed_sz = (dimensionality + kIsSquaredL2) * kBlockSize;
      FloatT* storage = &first_buffer_entry_;
      return storage + 2 * transposed_sz;
    }

    SCANN_SIMD_INLINE void TransposeDatabaseBlock(const size_t dimensionality,
                                                  const FloatT* database,
                                                  size_t first_dp_idx,
                                                  size_t n_to_transpose) {
      FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
      FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);

      if (ABSL_PREDICT_FALSE(n_to_transpose <= kBlockSize)) {
        TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                   n_to_transpose, transposed_ptr0);
      } else {
        TransposeDatabaseBlockImpl(dimensionality, database, first_dp_idx,
                                   kBlockSize, transposed_ptr0);
        TransposeDatabaseBlockImpl(
            dimensionality, database, first_dp_idx + kBlockSize,
            n_to_transpose - kBlockSize, transposed_ptr1);
      }
      if constexpr (kIsSquaredL2) {
        AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
      }
    }

    SCANN_SIMD_INLINE static void TransposeDatabaseBlockImpl(
        const size_t dimensionality, const FloatT* database,
        size_t first_dp_idx, size_t n_to_transpose,
        FloatT* __restrict__ transposed_storage) {
      DCHECK_LE(n_to_transpose, kBlockSize);
      size_t j = 0;
      for (; j + 4 <= n_to_transpose; j += 4) {
        const FloatT* database_ptr =
            database + (first_dp_idx + j) * dimensionality;
        FloatT* __restrict__ dest = transposed_storage + j;

        constexpr size_t kCacheLineElements = 64 / sizeof(FloatT);
        const FloatT* prefetch = database_ptr + 0 * dimensionality;
        const FloatT* prefetch_end = database_ptr + 4 * dimensionality;
        for (; prefetch < prefetch_end; prefetch += kCacheLineElements) {
          ::tensorflow::port::prefetch<::tensorflow::port::PREFETCH_HINT_T0>(
              prefetch);
        }

        const FloatT* ptr_begin = database_ptr + 1 * dimensionality;
        const FloatT* ptr_end = database_ptr + 2 * dimensionality;
        for (const FloatT* ptr = ptr_begin; ptr != ptr_end;
             ++ptr, dest += kBlockSize) {
          dest[0] = ptr[-1 * dimensionality];
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
             ++dim_idx, dest += kBlockSize) {
          dest[0] = untransposed0[dim_idx];
        }
      }
      DCHECK_EQ(j, n_to_transpose);
    }

    SCANN_SIMD_INLINE void ExpandPretransposedFP8Block(
        const FP8SimdBlockTransposedDatabase& fp8_db, size_t first_dp_idx,
        size_t n_to_transpose) {
      DCHECK_EQ(kBlockSize, fp8_db.simd_block_size());
      const size_t dimensionality = fp8_db.dimensionality();
      const size_t first_block_idx = first_dp_idx / kBlockSize;
      FloatT* __restrict__ transposed_ptr0 = GetTransposedPtr0(dimensionality);
      FloatT* __restrict__ transposed_ptr1 = GetTransposedPtr1(dimensionality);
      if (ABSL_PREDICT_FALSE(n_to_transpose <= kBlockSize)) {
        ExpandPretransposedFP8BlockImpl(
            fp8_db.GetBlock(first_block_idx), dimensionality, n_to_transpose,
            fp8_db.inverse_fp8_multipliers().data(), transposed_ptr0);
      } else {
        ExpandPretransposedFP8BlockImpl(
            fp8_db.GetBlock(first_block_idx), dimensionality, kBlockSize,
            fp8_db.inverse_fp8_multipliers().data(), transposed_ptr0);
        ExpandPretransposedFP8BlockImpl(
            fp8_db.GetBlock(first_block_idx + 1), dimensionality,
            n_to_transpose - kBlockSize,
            fp8_db.inverse_fp8_multipliers().data(), transposed_ptr1);
      }
      if constexpr (kIsSquaredL2) {
        AugmentWithL2Norms(transposed_ptr0, transposed_ptr1, dimensionality);
      }
    }

    SCANN_SIMD_INLINE void ExpandPretransposedFP8BlockImpl(
        ConstSpan<int8_t> block, size_t dimensionality, size_t n_to_transpose,
        const float* __restrict__ inverse_multipliers_or_null,
        FloatT* __restrict__ transposed_storage) {
      DCHECK_EQ(n_to_transpose * dimensionality, block.size());
      if (n_to_transpose == kBlockSize && !inverse_multipliers_or_null) {
#ifdef __clang__
#pragma clang loop vectorize_width(SCANN_FP8_EXPAND_WIDTH)
#endif
        for (size_t i : Seq(kBlockSize * dimensionality)) {
          transposed_storage[i] = static_cast<float>(block[i]);
        }
        return;
      }

      if (n_to_transpose == kBlockSize) {
        const int8_t* __restrict__ src = block.data();

#if defined(__x86_64__) && SCANN_FP8_EXPAND_WIDTH == 8
        if constexpr (std::is_same<FloatT, float>::value) {
          static_assert(kBlockSize == 8);
          for (size_t dim_idx : Seq(dimensionality)) {
            __m256 inv_multiplier_simd = _mm256_broadcast_ss((

                inverse_multipliers_or_null + dim_idx));
            __m128i int8s = _mm_loadl_pi(_mm_setzero_si128(),
                                         reinterpret_cast<const __m64*>(src));
            __m256i int32s = _mm256_cvtepi8_epi32(int8s);
            __m256 floats = _mm256_cvtepi32_ps(int32s) * inv_multiplier_simd;

            _mm256_store_ps(transposed_storage, floats);
            transposed_storage += kBlockSize;
            src += kBlockSize;
          }
          return;
        }
#elif defined(__x86_64__) && SCANN_FP8_EXPAND_WIDTH == 16
        if constexpr (std::is_same<FloatT, float>::value) {
          static_assert(kBlockSize == 16);
          for (size_t dim_idx : Seq(dimensionality)) {
            __m512 inv_multiplier_simd = _mm512_broadcast_f32x8(
                _mm256_broadcast_ss(inverse_multipliers_or_null + dim_idx));
            __m128i int8s =
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
            __m512i int32s = _mm512_cvtepi8_epi32(int8s);
            __m512 floats = _mm512_cvtepi32_ps(int32s) * inv_multiplier_simd;
            _mm512_store_ps(transposed_storage, floats);
            transposed_storage += kBlockSize;
            src += kBlockSize;
          }
          return;
        }
#endif
        for (size_t dim_idx : Seq(dimensionality)) {
          const float inv_multiplier = inverse_multipliers_or_null[dim_idx];
          for (size_t dp_idx : Seq(kBlockSize)) {
            transposed_storage[dp_idx] = src[dp_idx] * inv_multiplier;
          }
          transposed_storage += kBlockSize;
          src += kBlockSize;
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
        transposed_storage += kBlockSize;
        src += n_to_transpose;
      }
    }

   private:
    SCANN_SIMD_INLINE static void AugmentWithL2Norms(
        FloatT* __restrict__ transposed_ptr0,
        FloatT* __restrict__ transposed_ptr1, size_t dimensionality) {
      auto norm0 = SimdFuncs::Zeros();
      auto norm1 = SimdFuncs::Zeros();
      FloatT scalar_two = 2.0;
      auto two = SimdFuncs::Broadcast(&scalar_two);
      for (size_t dim : Seq(dimensionality)) {
        auto transposed_simd0 =
            SimdFuncs::Loadu(transposed_ptr0 + dim * kBlockSize);
        auto transposed_simd1 =
            SimdFuncs::Loadu(transposed_ptr1 + dim * kBlockSize);
        SimdFuncs::Accumulate(transposed_simd0, transposed_simd0, &norm0);
        SimdFuncs::Accumulate(transposed_simd1, transposed_simd1, &norm1);
        SimdFuncs::Storeu(transposed_ptr0 + dim * kBlockSize,
                          SimdFuncs::Multiply(two, transposed_simd0));
        SimdFuncs::Storeu(transposed_ptr1 + dim * kBlockSize,
                          SimdFuncs::Multiply(two, transposed_simd1));
      }
      FloatT scalar_neg1 = -1.0;
      auto neg1 = SimdFuncs::Broadcast(&scalar_neg1);
      SimdFuncs::Storeu(transposed_ptr0 - kBlockSize,
                        SimdFuncs::Multiply(norm0, neg1));
      SimdFuncs::Storeu(transposed_ptr1 - kBlockSize,
                        SimdFuncs::Multiply(norm1, neg1));
    }

    struct alignas(kBlockSize * sizeof(FloatT)) {
      FloatT first_buffer_entry_;
    };
  };

 public:
  constexpr static size_t kSmallQueryStride = 6;
  constexpr static size_t kResultsSize = 2 * kSmallQueryStride * kBlockSize;

  SCANN_INLINE DenseManyToManyTransposed(const DenseDataset<FloatT>& queries,
                                         const DenseDataset<FloatT>& database,
                                         thread::ThreadPool* pool,
                                         CallbackT callback)
      : dimensionality_(queries.dimensionality()),
        queries_(queries[0].values()),
        num_queries_(queries.size()),
        float_database_(database[0].values()),
        num_datapoints_(database.size()),
        pool_(pool),
        callback_(std::move(callback)) {
    DCHECK_EQ(queries.dimensionality(), database.dimensionality());
    DCHECK_GE(num_queries_, 1);
    if (num_queries_ >= 2) {
      DCHECK_EQ(queries_ + dimensionality_, queries[1].values());
    }
    DCHECK_GE(num_datapoints_, 1);
    if (num_datapoints_ >= 2) {
      DCHECK_EQ(float_database_ + dimensionality_, database[1].values());
    }
  }

  SCANN_INLINE DenseManyToManyTransposed(
      const DenseDataset<FloatT>& queries,
      const FP8SimdBlockTransposedDatabase* database, thread::ThreadPool* pool,
      CallbackT callback)
      : dimensionality_(queries.dimensionality()),
        queries_(queries[0].values()),
        num_queries_(queries.size()),
        fp8_database_(database),
        num_datapoints_(database->size()),
        pool_(pool),
        callback_(std::move(callback)) {
    DCHECK_EQ(queries.dimensionality(), database->dimensionality());
    DCHECK_GE(num_queries_, 1);
    if (num_queries_ >= 2) {
      DCHECK_EQ(queries_ + dimensionality_, queries[1].values());
    }
    DCHECK_GE(num_datapoints_, 1);
  }

  SCANN_INLINE void TopLevelBatch() {
    const size_t dimensionality = dimensionality_;
    const FloatT* queries = queries_;
    const size_t num_queries = num_queries_;
    const size_t num_datapoints = num_datapoints_;
    thread::ThreadPool* pool = pool_;

    if (kIsSquaredL2) {
      query_norms_.reset(new FloatT[num_queries]);
      FloatT* query_norms = query_norms_.get();
      ParallelFor<16>(Seq(num_queries), pool, [&](size_t q_idx) {
        const FloatT* q_ptr = queries + q_idx * dimensionality;
        query_norms[q_idx] =
            SquaredL2Norm(MakeDatapointPtr(q_ptr, dimensionality));
      });
    }

    const size_t q_stride_est1 = (1 << 19) / (dimensionality * sizeof(FloatT));

    const size_t q_stride_est2 =
        num_queries / DivRoundUp(num_queries, q_stride_est1);

    const size_t q_stride = NextMultipleOf(q_stride_est2, kSmallQueryStride);

    for (size_t q_idx = 0; q_idx < num_queries; q_idx += q_stride) {
      const size_t q_batch_size = std::min(q_stride, num_queries - q_idx);

      constexpr size_t kDatabaseStride = 2 * kBlockSize;
      const size_t num_db_blocks = DivRoundUp(num_datapoints, kDatabaseStride);
      ParallelFor<16>(Seq(num_db_blocks), pool, [&](size_t block_idx) {
        const size_t first_dp_idx = block_idx * kDatabaseStride;
        const size_t dp_batch_size =
            std::min(kDatabaseStride, num_datapoints - first_dp_idx);

        MidLevelBatch(q_idx, q_batch_size, first_dp_idx, dp_batch_size);
      });
    }
  }

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) {
    const size_t dimensionality = dimensionality_;
    const size_t q_idx_end = first_q_idx + num_queries;

    thread_local size_t allocated_dimensionality = 0;
    thread_local unique_ptr<Transposer> transposer_storage;
    if (allocated_dimensionality < dimensionality) {
      transposer_storage = Transposer::New(dimensionality, kResultsSize);
      allocated_dimensionality = dimensionality;
    }
    Transposer* transposer = transposer_storage.get();

    if (float_database_) {
      DCHECK(!fp8_database_);
      transposer->TransposeDatabaseBlock(dimensionality, float_database_,
                                         first_dp_idx, num_datapoints);
    } else if constexpr (std::is_same_v<FloatT, float>) {
      DCHECK(fp8_database_);
      transposer->ExpandPretransposedFP8Block(*fp8_database_, first_dp_idx,
                                              num_datapoints);
    }

    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = queries_ + first_q_idx * dimensionality;
    if (kIsSquaredL2) {
      args.query_norms = query_norms_.get() + first_q_idx;
    }
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.transposer = transposer;
    args.callback = &callback_;

    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
      BottomLevelBatch<kSmallQueryStride>(args);
      args.first_q_idx += kSmallQueryStride;
      if (kIsSquaredL2) {
        args.query_norms += kSmallQueryStride;
      }
      args.queries += kSmallQueryStride * dimensionality;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_6(final_batch_size, BottomLevelBatch,
                                           args);
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
    FloatT* results = args.transposer->GetResultsPtr(dimensionality);
    CallbackT& callback = *args.callback;

    const FloatT* volatile query_ptrs_vol[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs_vol[j] = queries + j * dimensionality;
    }
    const FloatT* query_ptrs[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      query_ptrs[j] = query_ptrs_vol[j];
    }

    DoAccumulationTransposedTemplate<kNumQueries>(
        transposed_ptr0, transposed_ptr1, query_ptrs, query_norms,
        dimensionality, results);

    for (size_t j : Seq(kNumQueries)) {
      auto query_results =
          MakeMutableSpan(results + j * 2 * kBlockSize, num_datapoints);
      const size_t query_idx = first_q_idx + j;
      callback(query_results, first_dp_idx, query_idx);
    }
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void DoAccumulationTransposedTemplate(
      const FloatT* transposed_block0, const FloatT* transposed_block1,
      const FloatT** query_ptrs, const FloatT* query_norms,
      size_t dimensionality, FloatT* __restrict__ real_acc) {
    typename SimdFuncs::Accumulator accumulators0[kNumQueries];
    typename SimdFuncs::Accumulator accumulators1[kNumQueries];
    for (size_t j : Seq(kNumQueries)) {
      if (kIsSquaredL2) {
        auto query_norm = SimdFuncs::Broadcast(&query_norms[j]);
        auto db_norms0 = SimdFuncs::Loadu(transposed_block0 - kBlockSize);
        auto db_norms1 = SimdFuncs::Loadu(transposed_block1 - kBlockSize);
        accumulators0[j] = SimdFuncs::Add(db_norms0, query_norm);
        accumulators1[j] = SimdFuncs::Add(db_norms1, query_norm);
      } else {
        accumulators0[j] = SimdFuncs::Zeros();
        accumulators1[j] = SimdFuncs::Zeros();
      }
    }

    for (size_t dim : Seq(dimensionality)) {
      auto transposed_simd0 =
          SimdFuncs::Loadu(transposed_block0 + dim * kBlockSize);
      auto transposed_simd1 =
          SimdFuncs::Loadu(transposed_block1 + dim * kBlockSize);

      for (size_t j : Seq(kNumQueries)) {
        auto query_simd = SimdFuncs::Broadcast(query_ptrs[j] + dim);
        SimdFuncs::Accumulate(query_simd, transposed_simd0, &accumulators0[j]);
        SimdFuncs::Accumulate(query_simd, transposed_simd1, &accumulators1[j]);
      }
    }

    for (size_t j : Seq(kNumQueries)) {
      FloatT* acc = real_acc + j * 2 * kBlockSize;
      SimdFuncs::Storeu(acc + 0 * kBlockSize, accumulators0[j]);
      SimdFuncs::Storeu(acc + 1 * kBlockSize, accumulators1[j]);
    }
  }

 private:
  const size_t dimensionality_;

  const FloatT* queries_;
  const size_t num_queries_;

  const FloatT* float_database_ = nullptr;
  const FP8SimdBlockTransposedDatabase* fp8_database_ = nullptr;
  const size_t num_datapoints_;

  unique_ptr<FloatT[]> query_norms_;

  thread::ThreadPool* pool_;
  CallbackT callback_;
};

template <typename SimdFuncs, bool kIsSquaredL2, typename CallbackT,
          typename FloatT>
SCANN_INLINE void DenseManyToManyTransposedImpl(
    const DenseDataset<FloatT>& queries, const DenseDataset<FloatT>& database,
    thread::ThreadPool* pool, CallbackT callback) {
  DenseManyToManyTransposed<SimdFuncs, SimdFuncs::kBlockSize, kIsSquaredL2,
                            CallbackT, FloatT>(queries, database, pool,
                                               std::move(callback))
      .TopLevelBatch();
}

template <typename CallbackT>
SCANN_INLINE void DenseManyToManyFP8PretransposedImpl(
    const DistanceMeasure& dist, const DenseDataset<float>& queries,
    const FP8SimdBlockTransposedDatabase& database, thread::ThreadPool* pool,
    CallbackT callback) {
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());
  using DotProductFns = DotProductDistanceFunctions<float>;
  using SquaredL2Fns = SquaredL2DistanceFunctions<float>;
  if (is_squared_l2) {
    DenseManyToManyTransposed<DotProductFns, SquaredL2Fns::kBlockSize, true,
                              CallbackT, float>(queries, &database, pool,
                                                std::move(callback))
        .TopLevelBatch();
  } else {
    DenseManyToManyTransposed<DotProductFns, DotProductFns::kBlockSize, false,
                              CallbackT, float>(queries, &database, pool,
                                                std::move(callback))
        .TopLevelBatch();
  }
}

template <typename SimdFuncs, typename CallbackT, typename FloatT>
class DenseManyToManyUntransposed {
 public:
  constexpr static size_t kBlockSize = SimdFuncs::kBlockSize;

  constexpr static size_t kFloatDivisor = sizeof(FloatT) / sizeof(float);

  constexpr static size_t kSmallQueryStride = 5;

  constexpr static size_t kMaxDbChunk = 256;

  SCANN_INLINE DenseManyToManyUntransposed(const DenseDataset<FloatT>& queries,
                                           const DenseDataset<FloatT>& database,
                                           thread::ThreadPool* pool,
                                           CallbackT callback)
      : dimensionality_(queries.dimensionality()),
        queries_(queries[0].values()),
        num_queries_(queries.size()),
        database_(database[0].values()),
        num_datapoints_(database.size()),
        pool_(pool),
        callback_(std::move(callback)) {
    DCHECK_EQ(queries.dimensionality(), database.dimensionality());
    DCHECK_GE(num_queries_, 1);
    if (num_queries_ >= 2) {
      DCHECK_EQ(queries_ + dimensionality_, queries[1].values());
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
    thread::ThreadPool* pool = pool_;

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

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) {
    const size_t dimensionality = dimensionality_;

    thread_local unique_ptr<FloatT[]> results;
    if (!results) {
      results.reset(new FloatT[kSmallQueryStride * kMaxDbChunk]);
    }

    BottomLevelBatchArgs args;
    args.dimensionality = dimensionality;
    args.queries = queries_ + first_q_idx * dimensionality;
    args.database = database_ + first_dp_idx * dimensionality;
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
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE_5(final_batch_size, BottomLevelBatch,
                                           args);
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

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void DoAccumulationUntransposedTemplate(
      size_t dimensionality, const FloatT** query_ptrs, const FloatT* dp0,
      const FloatT* dp1, FloatT* __restrict__* __restrict__ result_ptrs) {
    size_t n_blocks = dimensionality / kBlockSize;
    size_t dim_idx = 0;
    if (n_blocks) {
      typename SimdFuncs::Accumulator accumulators0[kNumQueries];
      typename SimdFuncs::Accumulator accumulators1[kNumQueries];
      for (size_t j : Seq(kNumQueries)) {
        accumulators0[j] = SimdFuncs::Zeros();
        accumulators1[j] = SimdFuncs::Zeros();
      }

      for (; n_blocks > 0; --n_blocks, dim_idx += kBlockSize) {
        auto dp0_simd = SimdFuncs::Loadu(dp0 + dim_idx);
        auto dp1_simd = SimdFuncs::Loadu(dp1 + dim_idx);
        for (size_t j : Seq(kNumQueries)) {
          auto query_simd = SimdFuncs::Loadu(query_ptrs[j] + dim_idx);
          SimdFuncs::Accumulate(query_simd, dp0_simd, &accumulators0[j]);
          SimdFuncs::Accumulate(query_simd, dp1_simd, &accumulators1[j]);
        }
      }

      for (size_t j = 0; j < kNumQueries;) {
        if (j + 1 < kNumQueries) {
          SimdFuncs::Postprocess4SimdRegisters(
              accumulators0[j], accumulators1[j], accumulators0[j + 1],
              accumulators1[j + 1], &result_ptrs[j][0], &result_ptrs[j][1],
              &result_ptrs[j + 1][0], &result_ptrs[j + 1][1]);
          j += 2;
        } else {
          SimdFuncs::Postprocess2SimdRegisters(
              accumulators0[j], accumulators1[j], &result_ptrs[j][0],
              &result_ptrs[j][1]);
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
        SimdFuncs::Accumulate(dp0_val, query_val, &result_ptrs[j][0]);
        SimdFuncs::Accumulate(dp1_val, query_val, &result_ptrs[j][1]);
      }
    }
  }

 private:
  const size_t dimensionality_;

  const FloatT* queries_;
  const size_t num_queries_;

  const FloatT* database_;
  const size_t num_datapoints_;

  thread::ThreadPool* pool_;
  CallbackT callback_;
};

template <typename SimdFuncs, typename CallbackT, typename FloatT>
SCANN_INLINE void DenseManyToManyUntransposedImpl(
    const DenseDataset<FloatT>& queries, const DenseDataset<FloatT>& database,
    thread::ThreadPool* pool, CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(), "");
  DenseManyToManyUntransposed<SimdFuncs, CallbackT, FloatT>(
      queries, database, pool, std::move(callback))
      .TopLevelBatch();
}

template <typename FloatT>
bool ShouldTranspose(const DenseDataset<FloatT>& queries,
                     const DenseDataset<FloatT>& database,
                     size_t simd_block_size) {
  const size_t n_dims = queries.dimensionality();

  constexpr size_t kDoubleDivisor = IsSame<FloatT, double>() ? 2 : 1;

  constexpr size_t kMaxMidLevelQueryBatchSize = 256 / kDoubleDivisor;

  const size_t n_queries = queries.size();
  const size_t n_mid_level_batches =
      DivRoundUp(n_queries, kMaxMidLevelQueryBatchSize);

  constexpr float kUntransposedCostPerQuery = 2.0;

  constexpr float kUntransposedCostPerDimPerQuery = 0.025;

  constexpr float kRemainderCostPerDimPerQuery = 0.025;

  constexpr float kRemainderCostPerQuery = 2.0;

  const float transposed_cost_per_dp = n_dims * n_mid_level_batches;

  const bool has_remainder = (n_dims % simd_block_size) != 0;
  const float untransposed_postprocess_cost_per_dp =
      n_queries * kUntransposedCostPerQuery +
      n_queries * n_dims * kUntransposedCostPerDimPerQuery +
      n_queries * n_dims * kRemainderCostPerDimPerQuery * has_remainder +
      n_queries * kRemainderCostPerQuery * has_remainder;

  return transposed_cost_per_dp < untransposed_postprocess_cost_per_dp;
}

template <typename FloatT, typename CallbackT>
SCANN_INLINE void DenseDistanceManyToManyImpl(
    const DistanceMeasure& dist, const DenseDataset<FloatT>& queries,
    const DenseDataset<FloatT>& database, thread::ThreadPool* pool,
    CallbackT callback) {
  static_assert(IsSameAny<FloatT, float, double>(),
                "DenseDistanceManyToMany only works with float and double.");
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());
  using DotProductFns = DotProductDistanceFunctions<FloatT>;
  using SquaredL2Fns = SquaredL2DistanceFunctions<FloatT>;
  constexpr size_t kBlockSize = DotProductFns::kBlockSize;
  const bool should_transpose = ShouldTranspose(queries, database, kBlockSize);
  if (kBlockSize > 1 && should_transpose) {
    if (is_squared_l2) {
      return DenseManyToManyTransposedImpl<DotProductFns, true>(
          queries, database, pool, std::move(callback));
    } else {
      return DenseManyToManyTransposedImpl<DotProductFns, false>(
          queries, database, pool, std::move(callback));
    }
  } else {
    if (is_squared_l2) {
      return DenseManyToManyUntransposedImpl<SquaredL2Fns>(
          queries, database, pool, std::move(callback));
    } else {
      return DenseManyToManyUntransposedImpl<DotProductFns>(
          queries, database, pool, std::move(callback));
    }
  }
}
