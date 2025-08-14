template <Int8TileSide kSide>
class Int8TileCodecImpl : public Int8TileCodec {
  using Int8Tile = std::conditional_t<kSide == Int8TileSide::kQuery,
                                      Int8QueryTile, Int8DatabaseTile>;
  static constexpr bool kUnsigned =
      IsSame<Int8Tile, avx512_vnni::Int8DatabaseTile>();
  static constexpr int kUnsignedOffset = kUnsigned ? 128 : 0;

 public:
  explicit Int8TileCodecImpl(uint32_t dimensionality)
      : Int8TileCodec(dimensionality,
                      NextMultipleOf(dimensionality, Int8Tile::kDims),
                      Int8Tile::kPoints, Simd<int8_t>::kRegisterBytes,
                      kPlatformGeneration, kSide) {}

  void EncodeDatapoint(ConstSpan<int8_t> datapoint, size_t dp_idx,
                       MutableSpan<int8_t> payload) const override {
    DCHECK_EQ(datapoint.size(), dimensionality());
    size_t block_start = BlockStart(dp_idx);
    auto block = payload.subspan(block_start, block_bytes_);
    for (size_t dim_idx : IndicesOf(datapoint)) {
      block[PayloadIndex(dp_idx, dim_idx)] =
          datapoint[dim_idx] + kUnsignedOffset;
    }
  }

  Datapoint<int8_t> ReconstructDatapoint(
      size_t dp_idx, ConstSpan<int8_t> payload) const override {
    auto block = payload.subspan(BlockStart(dp_idx), block_bytes_);
    std::vector<int8_t> data(dimensionality_);

    for (DimensionIndex dim_idx : Seq(dimensionality_)) {
      data[dim_idx] = block[PayloadIndex(dp_idx, dim_idx)] - kUnsignedOffset;
    }
    return Datapoint<int8_t>({}, std::move(data), dimensionality_);
  }

 private:
  size_t BlockStart(DatapointIndex dp_idx) const {
    return block_bytes_ * (dp_idx / Int8Tile::kPoints);
  }

  size_t PayloadIndex(DatapointIndex dp_idx, DimensionIndex dim_idx) const {
    const size_t tile_idx = dim_idx / Int8Tile::kDims;
    const size_t tile_dp_idx = dp_idx % Int8Tile::kPoints;
    const size_t tile_dim_idx = dim_idx % Int8Tile::kDims;

    size_t tile_row;
    size_t tile_col;
    if constexpr (IsSame<Int8Tile, amx::Int8DatabaseTile>()) {
      tile_row = tile_dim_idx / 4;
      tile_col = tile_dp_idx * 4 + tile_dim_idx % 4;
    } else {
      tile_row = tile_dp_idx;
      tile_col = tile_dim_idx;
    }
    return tile_idx * Int8TileBytes<Int8Tile>() + tile_row * Int8Tile::kDims +
           tile_col;
  }
};
