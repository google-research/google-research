#ifdef USE_EXPOSURE
vec2 clippedLogit(vec2 y) {
  const float eps = 1e-7;
  y = clamp(y, eps, 1.0 - eps);
  return log(y) - log(1.0 - y);
}
vec3 clippedLogit(vec3 y) {
  const float eps = 1e-7;
  y = clamp(y, eps, 1.0 - eps);
  return log(y) - log(1.0 - y);
}
vec4 clippedLogit(vec4 y) {
  const float eps = 1e-7;
  y = clamp(y, eps, 1.0 - eps);
  return log(y) - log(1.0 - y);
}
#define EXPOSURE_ADJUST(x) (clippedLogit(x) + log(exposure))
#else
#define EXPOSURE_ADJUST(x) (x)
#endif

/* This macro implements the following function:
 * float posEnc(float dir, int exponent) {
 *   float offset = 0.0;
 *   if (exponent >= 4) {
 *     const float kHalfPi = 1.57079632679489661923;
 *     offset = kHalfPi;
 *     exponent -= 4;
 *   }
 *  dir *= pow(2.0, float(exponent));
 *   return sin(dir + offset);  // We compute cos(x) as sin(x+pi/2).
 * }
 */
#define posEnc(dir, exponent) sin((dir) *\
  pow(2.0, (exponent) >= 4 ? float(exponent) - 4.0 : float(exponent)) +\
  ((exponent) >= 4 ? 1.57079632679489661923 : 0.0))

vec4 elu(vec4 x) {
  return vec4(
    x.r < 0.0 ? exp(x.r) - 1.0 : x.r,
    x.g < 0.0 ? exp(x.g) - 1.0 : x.g,
    x.b < 0.0 ? exp(x.b) - 1.0 : x.b,
    x.a < 0.0 ? exp(x.a) - 1.0 : x.a
  );
}

vec4 relu(vec4 x) {
  return max(vec4(0.0), x);
}

vec3 evaluateNetwork(
    vec3 color, vec4 features,
    #ifdef USE_FEATURE_CONCAT
    vec3 coarseColor, vec4 coarseFeatures,
    #endif
    vec3 viewdir) {
  vec4 intermediate_one[NUM_CHANNELS_ONE/4];
  INITIALIZE_OUTPUT_ACTIVATIONS_0
#define FIRST_LAYER(INDEX, INPUT) \
  for (int i = 0; i < NUM_CHANNELS_ONE; i += 4) {\
    intermediate_one[i/4] += INPUT * mat4(\
      texelFetch(weightsZero, ivec2(0, INDEX * NUM_CHANNELS_ONE + i), 0),\
      texelFetch(weightsZero, ivec2(0, INDEX * NUM_CHANNELS_ONE + (i+1)), 0),\
      texelFetch(weightsZero, ivec2(0, INDEX * NUM_CHANNELS_ONE + (i+2)), 0),\
      texelFetch(weightsZero, ivec2(0, INDEX * NUM_CHANNELS_ONE + (i+3)), 0)\
    );\
  }
#ifdef USE_FEATURE_CONCAT
  FIRST_LAYER(0, EXPOSURE_ADJUST(vec4(color, features.r)))
  FIRST_LAYER(1, EXPOSURE_ADJUST(vec4(features.gba, coarseColor.r)))
  FIRST_LAYER(2, EXPOSURE_ADJUST(vec4(coarseColor.gb, coarseFeatures.rg)))
  FIRST_LAYER(3, vec4(EXPOSURE_ADJUST(coarseFeatures.ba), viewdir.rg))
  FIRST_LAYER(4, vec4(viewdir.b, posEnc(viewdir.rgb, 0)))
  FIRST_LAYER(5, vec4(posEnc(viewdir.rgb, 1), posEnc(viewdir.r, 2)))
  FIRST_LAYER(6, vec4(posEnc(viewdir.gb, 2), posEnc(viewdir.rg, 3)))
  FIRST_LAYER(7, vec4(posEnc(viewdir.b, 3), posEnc(viewdir.rgb, 4)))
  FIRST_LAYER(8, vec4(posEnc(viewdir.rgb, 5), posEnc(viewdir.r, 6)))
  FIRST_LAYER(9, vec4(posEnc(viewdir.gb, 6), posEnc(viewdir.rg, 7)))
  FIRST_LAYER(10, vec4(posEnc(viewdir.b, 7), posEnc(viewdir.rgb, 8)))
#else
  FIRST_LAYER(0, EXPOSURE_ADJUST(vec4(color, features.r)))
  FIRST_LAYER(1, vec4(EXPOSURE_ADJUST(features.gba), viewdir.r))
  FIRST_LAYER(2, vec4(viewdir.gb, posEnc(viewdir.rg, 0)))
  FIRST_LAYER(3, vec4(posEnc(viewdir.b, 0), posEnc(viewdir.rgb, 1)))
  FIRST_LAYER(4, vec4(posEnc(viewdir.rgb, 2), posEnc(viewdir.r, 3)))
  FIRST_LAYER(5, vec4(posEnc(viewdir.gb, 3), posEnc(viewdir.rg, 4)))
  FIRST_LAYER(6, vec4(posEnc(viewdir.b, 4), posEnc(viewdir.rgb, 5)))
  FIRST_LAYER(7, vec4(posEnc(viewdir.rgb, 6), posEnc(viewdir.r, 7)))
  FIRST_LAYER(8, vec4(posEnc(viewdir.gb, 7), posEnc(viewdir.rg, 8)))
  FIRST_LAYER(9, vec4(posEnc(viewdir.b, 8), 0.0, 0.0, 0.0))
#endif
  vec4 intermediate_two[NUM_CHANNELS_TWO/4];
  INITIALIZE_OUTPUT_ACTIVATIONS_1
  for (int j = 0; j < NUM_CHANNELS_ONE/4; ++j) {
    vec4 inp = ACTIVATION_FN(intermediate_one[j]);
    for (int i = 0; i < NUM_CHANNELS_TWO; i += 4) {
      intermediate_two[i/4] += inp * mat4(
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + i), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+1)), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+2)), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+3)), 0)
      );
    }
  }

  vec4 result = bias_2[0];

  for (int j = 0; j < NUM_CHANNELS_TWO/4; ++j) {
    vec4 inp = ACTIVATION_FN(intermediate_two[j]);
    result += inp * mat4(
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 1), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 2), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 3), 0)
    );
  }
  return 1.0 / (1.0 + exp(-result.xyz)); // Sigmoid
}
