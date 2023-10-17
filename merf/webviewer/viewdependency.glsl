float indexToPosEnc(vec3 dir, int index) {
  float coordinate =
    (index % 3 == 0) ? dir.x : (
    (index % 3 == 1) ? dir.y : dir.z);
  if (index < 3) {
    return coordinate;
  }
  int scaleExponent = ((index - 3) % (3 * 4)) / 3;
  coordinate *= pow(2.0, float(scaleExponent));
  if ((index - 3) >= 3 * 4) {
    const float kHalfPi = 1.57079632679489661923;
    coordinate += kHalfPi;
  }
  return sin(coordinate);
}

float indexToInputValue(vec3 color, vec4 features, vec3 viewdir, int j) {
  float input_value = 0.0;
  if (j < 3) {
    input_value =
      (j == 0) ? color.r : (
      (j == 1) ? color.g : color.b);
  } else if (j < 7) {
    input_value =
      (j == 3) ? features.r : (
      (j == 4) ? features.g : (
      (j == 5) ? features.b : features.a));
  } else {
    input_value = indexToPosEnc(viewdir, j - 7);
  }
  if (abs(input_value) < 0.1 / 255.0) {
    input_value = 0.0;
  }
  return input_value;
}

vec4 relu(vec4 x) {
  return max(x, 0.0);
}

vec3 evaluateNetwork(
    vec3 color, vec4 features, vec3 viewdir) {

  vec4 intermediate_one[NUM_CHANNELS_ONE/4] = vec4[](
    BIAS_LIST_0
  );

  vec4 inp;
  mat4 w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 0),
      indexToInputValue(color, features, viewdir, 1),
      indexToInputValue(color, features, viewdir, 2),
      indexToInputValue(color, features, viewdir, 3));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 0), 0),
        texelFetch(weightsZero, ivec2(0, 1), 0),
        texelFetch(weightsZero, ivec2(0, 2), 0),
        texelFetch(weightsZero, ivec2(0, 3), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 4), 0),
        texelFetch(weightsZero, ivec2(0, 5), 0),
        texelFetch(weightsZero, ivec2(0, 6), 0),
        texelFetch(weightsZero, ivec2(0, 7), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 8), 0),
        texelFetch(weightsZero, ivec2(0, 9), 0),
        texelFetch(weightsZero, ivec2(0, 10), 0),
        texelFetch(weightsZero, ivec2(0, 11), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 12), 0),
        texelFetch(weightsZero, ivec2(0, 13), 0),
        texelFetch(weightsZero, ivec2(0, 14), 0),
        texelFetch(weightsZero, ivec2(0, 15), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 4),
      indexToInputValue(color, features, viewdir, 5),
      indexToInputValue(color, features, viewdir, 6),
      indexToInputValue(color, features, viewdir, 7));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 16), 0),
        texelFetch(weightsZero, ivec2(0, 17), 0),
        texelFetch(weightsZero, ivec2(0, 18), 0),
        texelFetch(weightsZero, ivec2(0, 19), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 20), 0),
        texelFetch(weightsZero, ivec2(0, 21), 0),
        texelFetch(weightsZero, ivec2(0, 22), 0),
        texelFetch(weightsZero, ivec2(0, 23), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 24), 0),
        texelFetch(weightsZero, ivec2(0, 25), 0),
        texelFetch(weightsZero, ivec2(0, 26), 0),
        texelFetch(weightsZero, ivec2(0, 27), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 28), 0),
        texelFetch(weightsZero, ivec2(0, 29), 0),
        texelFetch(weightsZero, ivec2(0, 30), 0),
        texelFetch(weightsZero, ivec2(0, 31), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 8),
      indexToInputValue(color, features, viewdir, 9),
      indexToInputValue(color, features, viewdir, 10),
      indexToInputValue(color, features, viewdir, 11));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 32), 0),
        texelFetch(weightsZero, ivec2(0, 33), 0),
        texelFetch(weightsZero, ivec2(0, 34), 0),
        texelFetch(weightsZero, ivec2(0, 35), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 36), 0),
        texelFetch(weightsZero, ivec2(0, 37), 0),
        texelFetch(weightsZero, ivec2(0, 38), 0),
        texelFetch(weightsZero, ivec2(0, 39), 0)
      );
  intermediate_one[1] += inp * w;
  

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 40), 0),
        texelFetch(weightsZero, ivec2(0, 41), 0),
        texelFetch(weightsZero, ivec2(0, 42), 0),
        texelFetch(weightsZero, ivec2(0, 43), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 44), 0),
        texelFetch(weightsZero, ivec2(0, 45), 0),
        texelFetch(weightsZero, ivec2(0, 46), 0),
        texelFetch(weightsZero, ivec2(0, 47), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 12),
      indexToInputValue(color, features, viewdir, 13),
      indexToInputValue(color, features, viewdir, 14),
      indexToInputValue(color, features, viewdir, 15));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 48), 0),
        texelFetch(weightsZero, ivec2(0, 49), 0),
        texelFetch(weightsZero, ivec2(0, 50), 0),
        texelFetch(weightsZero, ivec2(0, 51), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 52), 0),
        texelFetch(weightsZero, ivec2(0, 53), 0),
        texelFetch(weightsZero, ivec2(0, 54), 0),
        texelFetch(weightsZero, ivec2(0, 55), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 56), 0),
        texelFetch(weightsZero, ivec2(0, 57), 0),
        texelFetch(weightsZero, ivec2(0, 58), 0),
        texelFetch(weightsZero, ivec2(0, 59), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 60), 0),
        texelFetch(weightsZero, ivec2(0, 61), 0),
        texelFetch(weightsZero, ivec2(0, 62), 0),
        texelFetch(weightsZero, ivec2(0, 63), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 16),
      indexToInputValue(color, features, viewdir, 17),
      indexToInputValue(color, features, viewdir, 18),
      indexToInputValue(color, features, viewdir, 19));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 64), 0),
        texelFetch(weightsZero, ivec2(0, 65), 0),
        texelFetch(weightsZero, ivec2(0, 66), 0),
        texelFetch(weightsZero, ivec2(0, 67), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 68), 0),
        texelFetch(weightsZero, ivec2(0, 69), 0),
        texelFetch(weightsZero, ivec2(0, 70), 0),
        texelFetch(weightsZero, ivec2(0, 71), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 72), 0),
        texelFetch(weightsZero, ivec2(0, 73), 0),
        texelFetch(weightsZero, ivec2(0, 74), 0),
        texelFetch(weightsZero, ivec2(0, 75), 0)
      );
      intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 76), 0),
        texelFetch(weightsZero, ivec2(0, 77), 0),
        texelFetch(weightsZero, ivec2(0, 78), 0),
        texelFetch(weightsZero, ivec2(0, 79), 0)
      );
      intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 20),
      indexToInputValue(color, features, viewdir, 21),
      indexToInputValue(color, features, viewdir, 22),
      indexToInputValue(color, features, viewdir, 23));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 80), 0),
        texelFetch(weightsZero, ivec2(0, 81), 0),
        texelFetch(weightsZero, ivec2(0, 82), 0),
        texelFetch(weightsZero, ivec2(0, 83), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 84), 0),
        texelFetch(weightsZero, ivec2(0, 85), 0),
        texelFetch(weightsZero, ivec2(0, 86), 0),
        texelFetch(weightsZero, ivec2(0, 87), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 88), 0),
        texelFetch(weightsZero, ivec2(0, 89), 0),
        texelFetch(weightsZero, ivec2(0, 90), 0),
        texelFetch(weightsZero, ivec2(0, 91), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 92), 0),
        texelFetch(weightsZero, ivec2(0, 93), 0),
        texelFetch(weightsZero, ivec2(0, 94), 0),
        texelFetch(weightsZero, ivec2(0, 95), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 24),
      indexToInputValue(color, features, viewdir, 25),
      indexToInputValue(color, features, viewdir, 26),
      indexToInputValue(color, features, viewdir, 27));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 96), 0),
        texelFetch(weightsZero, ivec2(0, 97), 0),
        texelFetch(weightsZero, ivec2(0, 98), 0),
        texelFetch(weightsZero, ivec2(0, 99), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 100), 0),
        texelFetch(weightsZero, ivec2(0, 101), 0),
        texelFetch(weightsZero, ivec2(0, 102), 0),
        texelFetch(weightsZero, ivec2(0, 103), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 104), 0),
        texelFetch(weightsZero, ivec2(0, 105), 0),
        texelFetch(weightsZero, ivec2(0, 106), 0),
        texelFetch(weightsZero, ivec2(0, 107), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 108), 0),
        texelFetch(weightsZero, ivec2(0, 109), 0),
        texelFetch(weightsZero, ivec2(0, 110), 0),
        texelFetch(weightsZero, ivec2(0, 111), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 28),
      indexToInputValue(color, features, viewdir, 29),
      indexToInputValue(color, features, viewdir, 30),
      indexToInputValue(color, features, viewdir, 31));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 112), 0),
        texelFetch(weightsZero, ivec2(0, 113), 0),
        texelFetch(weightsZero, ivec2(0, 114), 0),
        texelFetch(weightsZero, ivec2(0, 115), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 116), 0),
        texelFetch(weightsZero, ivec2(0, 117), 0),
        texelFetch(weightsZero, ivec2(0, 118), 0),
        texelFetch(weightsZero, ivec2(0, 119), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 120), 0),
        texelFetch(weightsZero, ivec2(0, 121), 0),
        texelFetch(weightsZero, ivec2(0, 122), 0),
        texelFetch(weightsZero, ivec2(0, 123), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 124), 0),
        texelFetch(weightsZero, ivec2(0, 125), 0),
        texelFetch(weightsZero, ivec2(0, 126), 0),
        texelFetch(weightsZero, ivec2(0, 127), 0)
      );
  intermediate_one[3] += inp * w;

  inp = vec4(
      indexToInputValue(color, features, viewdir, 32),
      indexToInputValue(color, features, viewdir, 33),
      indexToInputValue(color, features, viewdir, 34),
      indexToInputValue(color, features, viewdir, 35));

  w = mat4(
        texelFetch(weightsZero, ivec2(0, 128), 0),
        texelFetch(weightsZero, ivec2(0, 129), 0),
        texelFetch(weightsZero, ivec2(0, 130), 0),
        texelFetch(weightsZero, ivec2(0, 131), 0)
      );
  intermediate_one[0] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 132), 0),
        texelFetch(weightsZero, ivec2(0, 133), 0),
        texelFetch(weightsZero, ivec2(0, 134), 0),
        texelFetch(weightsZero, ivec2(0, 135), 0)
      );
  intermediate_one[1] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 136), 0),
        texelFetch(weightsZero, ivec2(0, 137), 0),
        texelFetch(weightsZero, ivec2(0, 138), 0),
        texelFetch(weightsZero, ivec2(0, 139), 0)
      );
  intermediate_one[2] += inp * w;


  w = mat4(
        texelFetch(weightsZero, ivec2(0, 140), 0),
        texelFetch(weightsZero, ivec2(0, 141), 0),
        texelFetch(weightsZero, ivec2(0, 142), 0),
        texelFetch(weightsZero, ivec2(0, 143), 0)
      );
  intermediate_one[3] += inp * w;


  vec4 intermediate_two[NUM_CHANNELS_TWO/4] = vec4[](
    BIAS_LIST_1
  );
  for (int j = 0; j < NUM_CHANNELS_ONE/4; ++j) {
    inp = relu(intermediate_one[j]);
    for (int i = 0; i < NUM_CHANNELS_TWO; i += 4) {
      w = mat4(
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + i), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+1)), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+2)), 0),
        texelFetch(weightsOne, ivec2(0, j * NUM_CHANNELS_TWO + (i+3)), 0)
      );
      intermediate_two[i/4] += inp * w;
    }
  }

  vec4 result = BIAS_LIST_2;
  for (int j = 0; j < NUM_CHANNELS_TWO/4; ++j) {
    inp = relu(intermediate_two[j]);
    w = mat4(
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 1), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 2), 0),
      texelFetch(weightsTwo, ivec2(0, j * NUM_CHANNELS_THREE + 3), 0)
    );
    result.xyz += (inp * w).xyz;
  }
  return 1.0 / (1.0 + exp(-result.xyz)); // Sigmoid
}
