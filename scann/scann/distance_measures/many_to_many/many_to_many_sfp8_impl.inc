

template <bool kIsSquaredL2, typename CallbackT>
class DenseManyToManySFP8Pretransposed final
    : public DenseManyToManyTransposedBase<float> {
 public:
  using Base = DenseManyToManyTransposedBase<float>;
  using Base::kElementsPerRegister;
  using Base::kResultsSize;
  using Base::kSmallQueryStride;

  SCANN_INLINE DenseManyToManySFP8Pretransposed(
      const SFP8SimdBlockTransposedDatabase& queries,
      const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
      CallbackT callback)
      : Base(queries, database.size(), pool),
        queries_(queries),
        database_(database),
        callback_(std::move(callback)) {
    static_assert(Int8QueryTile::kDims == Int8DatabaseTile::kDims);
    static_assert(kSmallQueryStride % Int8QueryTile::kPoints == 0);
    static_assert(kElementsPerRegister == Int8DatabaseTile::kPoints);
    CHECK_EQ(queries.dimensionality(), database.dimensionality());
    CHECK_EQ(queries.datapoint_bytes(), database.datapoint_bytes());
    CHECK_EQ(queries.platform_generation(), kPlatformGeneration);
    CHECK_EQ(database.platform_generation(), kPlatformGeneration);
  }

  SCANN_SIMD_OUTLINE void MidLevelBatch(size_t first_q_idx, size_t num_queries,
                                        size_t first_dp_idx,
                                        size_t num_datapoints) final {
    const size_t datapoint_bytes = this->queries_.datapoint_bytes();
    const size_t q_idx_end = first_q_idx + num_queries;
    BottomLevelBatchArgs args;
    args.query_block_bytes = this->queries_.block_bytes();
    args.int8_queries =
        this->queries_.payload().data() + first_q_idx * datapoint_bytes;
    args.database_block0 =
        this->database_.payload().data() + first_dp_idx * datapoint_bytes;
    args.database_block1 = args.database_block0 + this->database_.block_bytes();
    args.query_scales = this->queries_.scales().data() + first_q_idx;
    args.query_sums = this->queries_.sums().data() + first_q_idx;
    args.db_scales = this->database_.scales().data() + first_dp_idx;
    args.first_q_idx = first_q_idx;
    args.first_dp_idx = first_dp_idx;
    args.num_datapoints = num_datapoints;
    args.callback = &this->callback_;
    if constexpr (kIsSquaredL2) {
      args.query_squared_l2_norms =
          this->queries_.squared_l2_norms().data() + first_q_idx;
      args.database_squared_l2_norms =
          this->database_.squared_l2_norms().data() + first_dp_idx;
    } else {
      args.query_squared_l2_norms = nullptr;
      args.database_squared_l2_norms = nullptr;
    }

    constexpr size_t kMaxNumQueryBlocks =
        DivRoundUp(kSmallQueryStride, Int8QueryTile::kPoints);
    thread_local Int8QueryExpander expander;
    expander.Ensure(kMaxNumQueryBlocks * args.query_block_bytes);
    args.int8_query_expander = &expander;

    while (args.first_q_idx + kSmallQueryStride <= q_idx_end) {
      BottomLevelBatch<kSmallQueryStride>(args);
      args.first_q_idx += kSmallQueryStride;
      args.query_scales += kSmallQueryStride;
      args.query_sums += kSmallQueryStride;
      if constexpr (kIsSquaredL2) {
        args.query_squared_l2_norms += kSmallQueryStride;
      }
      args.int8_queries += kSmallQueryStride * datapoint_bytes;
    }

    const size_t final_batch_size = q_idx_end - args.first_q_idx;
    SCANN_CALL_FUNCTION_BY_MM_BATCH_SIZE(kSmallQueryStride, final_batch_size,
                                         BottomLevelBatch, args);
  }

  struct BottomLevelBatchArgs {
    size_t query_block_bytes;
    const int8_t* int8_queries;
    Int8QueryExpander* int8_query_expander;
    const int8_t* database_block0;
    const int8_t* database_block1;
    const float* query_scales;
    const int32_t* query_sums;
    const float* db_scales;
    size_t first_q_idx;
    size_t first_dp_idx;
    size_t num_datapoints;
    CallbackT* callback;
    const float* query_squared_l2_norms;
    const float* database_squared_l2_norms;
  };

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static void BottomLevelBatch(BottomLevelBatchArgs args) {
    CallbackT& callback = *args.callback;

    auto accumulators = DoAccumulationTransposedTemplate<kNumQueries>(
        args.query_scales, args.query_sums, args.db_scales,
        args.query_squared_l2_norms, args.database_squared_l2_norms,
        args.database_block0, args.database_block1, args.int8_queries,
        args.query_block_bytes, args.int8_query_expander);
    Base::template PassResultsToCallback<kNumQueries>(
        accumulators, callback, args.first_q_idx, args.first_dp_idx,
        args.num_datapoints);
  }

  template <size_t kNumQueries>
  SCANN_SIMD_INLINE static Simd<float, kNumQueries, 2>
  DoAccumulationTransposedTemplate(
      const float* query_scales, const int32_t* query_sums,
      const float* database_scales, const float* query_squared_l2_norms,
      const float* database_squared_l2_norms, const int8_t* database_block0,
      const int8_t* database_block1, const int8_t* int8_queries,
      size_t query_block_bytes, Int8QueryExpander* int8_query_expander) {
    constexpr size_t kNumQueryBlocks =
        DivRoundUp(kNumQueries, Int8QueryTile::kPoints);

    using QueryT = typename Int8QueryExpander::ExpandedQueryT;
    const QueryT* queries = int8_query_expander->Expand(
        MakeConstSpan(int8_queries, kNumQueryBlocks * query_block_bytes));

    const QueryT* volatile query_blocks_vol[kNumQueryBlocks];
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      query_blocks_vol[q_b] = queries + q_b * query_block_bytes;
    }

    const QueryT* query_blocks[kNumQueryBlocks];
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      query_blocks[q_b] = query_blocks_vol[q_b];
    }

    Int32AccumulatorTile accumulators[2][kNumQueryBlocks];

    constexpr size_t kQueryTileBytes = Int8TileBytes<Int8QueryTile>();
    constexpr size_t kDatapointTileBytes = Int8TileBytes<Int8DatabaseTile>();
    constexpr size_t kTileBytesRatio = kDatapointTileBytes / kQueryTileBytes;
    static_assert(kDatapointTileBytes % kQueryTileBytes == 0);

    for (size_t query_pos :
         SeqWithStride<kQueryTileBytes>(0, query_block_bytes)) {
      Int8DatabaseTile database_tile[2];
      database_tile[0].Load(database_block0 + query_pos * kTileBytesRatio);
      database_tile[1].Load(database_block1 + query_pos * kTileBytesRatio);

      for (size_t q_b : Seq(kNumQueryBlocks)) {
        Int8QueryTile query_tile;
        query_tile.Load(query_blocks[q_b] + query_pos);

        accumulators[0][q_b].AccumulateDotProducts(query_tile,
                                                   database_tile[0]);
        accumulators[1][q_b].AccumulateDotProducts(query_tile,
                                                   database_tile[1]);
      }
    }

    Simd<int32_t, 2, kNumQueryBlocks, Int8QueryTile::kPoints> int32_results;
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      accumulators[0][q_b].GetResult(int32_results[0][q_b], query_sums);
      accumulators[1][q_b].GetResult(int32_results[1][q_b], query_sums);
      query_sums += Int8QueryTile::kPoints;
    }
    Simd<float, kNumQueries, 2> float_results;

    auto db_scales =
        Simd<float>::Zeros() - Simd<float, 2>::Load(database_scales);
    Simd<float, 2> db_squared_l2_norms;
    if constexpr (kIsSquaredL2) {
      db_squared_l2_norms = Simd<float, 2>::Load(database_squared_l2_norms);
      db_scales += db_scales;
    }
    size_t query_idx = 0;
    for (size_t q_b : Seq(kNumQueryBlocks)) {
      for (size_t p : Seq(Int8QueryTile::kPoints)) {
        Simd<int32_t, 2> int32_result;
        int32_result[0] = int32_results[0][q_b][p];
        int32_result[1] = int32_results[1][q_b][p];
        const auto query_scale =
            Simd<float>::Broadcast(query_scales[query_idx]);
        auto float_result =
            int32_result.ConvertTo<float>() * db_scales * query_scale;
        if constexpr (kIsSquaredL2) {
          float_result +=
              db_squared_l2_norms +
              Simd<float>::Broadcast(query_squared_l2_norms[query_idx]);
        }
        float_results[query_idx++] = float_result;
        if (query_idx == kNumQueries) break;
      }
    }
    return float_results;
  }

 private:
  const SFP8SimdBlockTransposedDatabase& queries_;
  const SFP8SimdBlockTransposedDatabase& database_;
  CallbackT callback_;
};

template <typename CallbackT>
SCANN_INLINE void DenseManyToManySFP8PretransposedImpl(
    const DistanceMeasure& dist, const SFP8SimdBlockTransposedDatabase& queries,
    const SFP8SimdBlockTransposedDatabase& database, ThreadPool* pool,
    CallbackT callback) {
  const bool is_squared_l2 =
      (DistanceMeasure::SQUARED_L2 == dist.specially_optimized_distance_tag());
  if (is_squared_l2) {
    DenseManyToManySFP8Pretransposed<true, CallbackT>(queries, database, pool,
                                                      std::move(callback))
        .TopLevelBatch();
  } else {
    DCHECK_EQ(dist.specially_optimized_distance_tag(),
              DistanceMeasure::DOT_PRODUCT);
    DenseManyToManySFP8Pretransposed<false, CallbackT>(queries, database, pool,
                                                       std::move(callback))
        .TopLevelBatch();
  }
}
