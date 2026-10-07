#if defined(__aarch64__)
#include <arm_neon.h>

namespace research_scann {
namespace asymmetric_hashing_internal {
namespace neon {

SCANN_INLINE uint32_t ComputePushMask(const int16_t* dist16,
                                      int16_t threshold) {
  const uint8x16_t bit_weights =
      vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL));

  const int16x8_t th = vdupq_n_s16(threshold);
  const int16x8_t d0 = vld1q_s16(dist16 + 0);
  const int16x8_t d1 = vld1q_s16(dist16 + 8);
  const int16x8_t d2 = vld1q_s16(dist16 + 16);
  const int16x8_t d3 = vld1q_s16(dist16 + 24);

  uint16x8_t cmp0 = vcgtq_s16(th, d0);
  uint16x8_t cmp1 = vcgtq_s16(th, d1);
  uint16x8_t cmp2 = vcgtq_s16(th, d2);
  uint16x8_t cmp3 = vcgtq_s16(th, d3);

  const uint8x16_t cmp01 =
      vuzp1q_u8(vreinterpretq_u8_u16(cmp0), vreinterpretq_u8_u16(cmp1));
  const uint8x16_t cmp23 =
      vuzp1q_u8(vreinterpretq_u8_u16(cmp2), vreinterpretq_u8_u16(cmp3));

  const uint8x16_t d01_weights = vandq_u8(cmp01, bit_weights);
  const uint8x16_t d23_weights = vandq_u8(cmp23, bit_weights);

  const uint8x16_t d0123_weights = vpaddq_u8(d01_weights, d23_weights);

  uint8x8_t res = vget_low_u8(vpaddq_u8(d0123_weights, d0123_weights));
  res = vpadd_u8(res, res);

  return vget_lane_u32(vreinterpret_u32_u8(res), 0);
}

}  // namespace neon
}  // namespace asymmetric_hashing_internal
}  // namespace research_scann

#endif  // defined(__aarch64__)
