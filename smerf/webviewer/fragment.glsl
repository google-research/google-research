vec2 rayAabbIntersection(vec3 aabbMin, vec3 aabbMax, vec3 origin,
                         vec3 invDirection) {
  vec3 t1 = (aabbMin - origin) * invDirection;
  vec3 t2 = (aabbMax - origin) * invDirection;
  vec3 tMin = min(t1, t2);
  vec3 tMax = max(t1, t2);
  return vec2(max(tMin.x, max(tMin.y, tMin.z)),
              min(tMax.x, min(tMax.y, tMax.z)));
}

// Declare these short functions as macros to force inlining.
#define sigmoid(x) (1.0 / (1.0 + exp(-(x))))
#define denormalize(x, min, max) ((min) + (x) * ((max) - (min)))
#define densityActivation(x) exp((x) - 1.0f)
#define densityToAlpha(x, stepSize) (1.0 - exp(-(x) * (stepSize)))
#define max3(v) max(max((v).x, (v).y), (v).z)

// Projective contraction
vec3 contract(vec3 x) {
  vec3 xAbs = abs(x);
  float xMax = max3(xAbs);
  if (xMax <= 1.0) {
    return x;
  }
  float scale = 1.0 / xMax;
  vec3 z = scale * x;
  // note that z.a = sign(z.a) where a is the the argmax component
  if (xAbs.x >= xAbs.y && xAbs.x >= xAbs.z) {
    z.x *= (2.0 - scale); // argmax = 0
  } else if (xAbs.y >= xAbs.x && xAbs.y >= xAbs.z) {
    z.y *= (2.0 - scale); // argmax = 1
  } else {
    z.z *= (2.0 - scale); // argmax = 2
  }
  return z;
}

// Inverse projective contraction
vec3 inverseContract(vec3 z) {
  vec3 zAbs = abs(z);
  float zMax = max3(zAbs);
  if (zMax <= 1.0) {
    return z;
  }
  const float eps = 1e-6;
  float invZMax = max(eps, 2.0 - zMax);
  float scale = 1.0 / invZMax;
  vec3 x = scale * z;
  if (zAbs.x >= zAbs.y && zAbs.x >= zAbs.z) {
    x.x = sign(x.x) * scale; // argmax = 0
  } else if (zAbs.y >= zAbs.x && zAbs.y >= zAbs.z) {
    x.y = sign(x.y) * scale; // argmax = 1
  } else {
    x.z = sign(x.z) * scale; // argmax = 2
  }
  return x;
}

// Sorts an array of length 5 in-place. This is hardcoded to 5 since a ray
// traverses up to 5 quadrants.
void sort5(inout float[5] array, int arrayLength) {
  float t;
  for (int i = 0; i < arrayLength; ++i) {
    for (int j = i+1; j < arrayLength; ++j) {
      if (array[j] < array[i]) {
        t = array[i];
        array[i] = array[j];
        array[j] = t;
      }
    }
  }
}

// A solution is invalid if it does not lie on the plane or is outside of 
// the bounding box
#define INF 1e25
#define SOLUTION_CHECK(T, P, AXIS)\
q = contract(o + T.AXIS * d);\
if (abs(q.AXIS - P.AXIS) > eps || any(lessThan(q, aabbMin - eps)) ||\
    any(greaterThan(q, aabbMax + eps))) {\
  T.AXIS = -INF;\
}

// First checks wether the computed cancidate solutions are actually lieing on
// the bounding box. Then of all the valid intersections we return the one with
// the highest t-value (tMax).
// o: origin
// d: direction
// t0: candiate solutions for intersections with minimum YZ, XZ, XY planes
// t1: candiate solutions for intersections with maximum YZ, XZ, XY planes
// aabbMin: minimum of axis-aligned bounding bound
// aabbMin: maximum of axis-aligned bounding bound
float getTMax(vec3 o, vec3 d, vec3 t0, vec3 t1,
  vec3 aabbMin, vec3 aabbMax) {
  float eps = 1e-3;
  vec3 q;

  // Invalid solutions are set to -INF and therefore ignored.
  SOLUTION_CHECK(t0, aabbMin, x)
  SOLUTION_CHECK(t0, aabbMin, y)
  SOLUTION_CHECK(t0, aabbMin, z)
  SOLUTION_CHECK(t1, aabbMax, x)
  SOLUTION_CHECK(t1, aabbMax, y)
  SOLUTION_CHECK(t1, aabbMax, z)
  return max(max3(t0), max3(t1));
}

// The following functions compute intersections between rays and axis-aligned
// planes in contracted space.
// The seven functions correspond to seven cases assiociated with the seven
// quadrants present in projective contraction. The functions are derived 
// by solving contract(o+t*d) for t.
// o: origin
// d: direction
// p: x, y and z components define axis-aligned planes that the ray (o, d) is
//    intersected against
//    (x -> YZ-plane, y -> XZ-plane, z -> XY-plane)
vec3 h(vec3 o, vec3 d, vec3 p) {
  return (p - o) / d;
}

vec3 h0(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = (1.0 / (2.0 - p.x) - o.x) / d.x;
  t.y = (o.y - p.y * o.x) / (p.y * d.x - d.y);
  t.z = (o.z - p.z * o.x) / (p.z * d.x - d.z);
  return t;
}

vec3 h1(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = (o.x - p.x * o.y) / (p.x * d.y - d.x);
  t.y = (1.0 / (2.0 - p.y) - o.y) / d.y;
  t.z = (o.z - p.z * o.y) / (p.z * d.y - d.z);
  return t;
}

vec3 h2(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = (o.x - p.x * o.z) / (p.x * d.z - d.x);
  t.y = (o.y - p.y * o.z) / (p.y * d.z - d.y);
  t.z = (1.0 / (2.0 - p.z) - o.z) / d.z;
  return t;
}

vec3 h3(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = (1.0 / (-p.x - 2.0) - o.x) / d.x;
  t.y = -(o.x*p.y + o.y) / (d.x*p.y + d.y);
  t.z = -(o.x*p.z + o.z) / (d.x*p.z + d.z);
  return t;
}

vec3 h4(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = -(o.y*p.x + o.x) / (d.y*p.x + d.x);
  t.y = (1.0 / (-p.y - 2.0) - o.y) / d.y;
  t.z = -(o.y*p.z + o.z) / (d.y*p.z + d.z);
  return t;
}

vec3 h5(vec3 o, vec3 d, vec3 p) {
  vec3 t;
  t.x = -(o.z*p.x + o.x) / (d.z*p.x + d.x);
  t.y = -(o.z*p.y + o.y) / (d.z*p.y + d.y);
  t.z = (1.0 / (-p.z - 2.0) - o.z) / d.z;
  return t;
}

// Intersects ray with all seven quadrants to obtain t-values at which the ray
// exits a quadrant. We need to know these t-values since whenever we 
// enter a new quadrant the origin and direction of the ray in contracted space
// needs to be recomputed.
float[5] findTraversedQuadrants(vec3 o, vec3 d, float near) {
   // Rays traverse up to 5 quadrants
  float[5] listQuadrantTMax = float[](INF, INF, INF, INF, INF);
  int numQuadrantsTraversed = 0;
  float c1 = 1.0 - 1e-5;
  float c2 = 2.0 - 1e-4;
  vec3 aabbMin;
  vec3 aabbMax;
  vec3 t0;
  vec3 t1;
  float tMax;

  // core region
  aabbMin = vec3(-1.0, -1.0, -1.0);
  aabbMax = vec3(1.0, 1.0, 1.0);
  t0 = h(o, d, aabbMin);
  t1 = h(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);

  // We discard intersections with quadrants that lie behind the camera
  // (tMax < near). When a quadrant is not traversed, getTMax returns -INF
  // and therefore this check also discards these values.
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 0, o[0]+t*d[0] >= 0
  aabbMin = vec3( c1, -c1, -c1);
  aabbMax = vec3( c2,  c1,  c1);
  t0 = h0(o, d, aabbMin);
  t1 = h0(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 1, o[1]+t*d[1] >= 0
  aabbMin = vec3(-c1, c1, -c1);
  aabbMax = vec3(c1, c2, c1);
  t0 = h1(o, d, aabbMin);
  t1 = h1(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 2, o[2]+t*d[2] >= 0
  aabbMin = vec3(-c1, -c1, c1);
  aabbMax = vec3(c1, c1, c2);
  t0 = h2(o, d, aabbMin);
  t1 = h2(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 0, o[0]+t*d[0] < 0
  aabbMin = vec3(-c2, -c1, -c1);
  aabbMax = vec3(-c1, c1, c1);
  t0 = h3(o, d, aabbMin);
  t1 = h3(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 1, o[1]+t*d[1] < 0
  aabbMin = vec3(-c1, -c2, -c1);
  aabbMax = vec3(c1, -c1, c1);
  t0 = h4(o, d, aabbMin);
  t1 = h4(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  // argmax(|o+t*d|) = 2, o[2]+t*d[2] < 0
  aabbMin = vec3(-c1, -c1, -c2);
  aabbMax = vec3(c1, c1, -c1);
  t0 = h5(o, d, aabbMin);
  t1 = h5(o, d, aabbMax);
  tMax = getTMax(o, d, t0, t1, aabbMin, aabbMax);
  if (tMax >= near) {
    listQuadrantTMax[numQuadrantsTraversed] = tMax;
    numQuadrantsTraversed++;
  }

  sort5(listQuadrantTMax, numQuadrantsTraversed);
  return listQuadrantTMax;
}

struct QuadrantSetupResults {
  vec3 oContracted; // Ray origin in contracted space.
  vec3 dContracted; // Ray direction in contracted space.
  vec2 quadrantTMinMaxContracted; // Contraction-space t-values at which the
  // ray enters or exits the current quadrant.
};

// This function is called whenever we enter a new quadrant. We compute
// origin and direction of the ray in contracted space and compute for which
// t-value (in contracted space) the ray enters/exits the quadrant
// tP and tQ are two world-space t-values that must lie within th entered
// quadrant, i.e. contract(o+tP*d) and  contract(o+tQ*d) must lie within the
// entered quadrant.
QuadrantSetupResults quadrantSetup(vec3 o, vec3 d, float tP, float tQ) {
  QuadrantSetupResults r;

  // Which quadrant did we enter?
  vec3 xP = o + tP * d;
  vec3 xAbs = abs(xP);
  float xMax = max3(xAbs);

  // Get the AABB of the quadrant the point x is in
  // Non-squash case, central quadrant:
  vec3 aabbMin = vec3(-1.0, -1.0, -1.0);
  vec3 aabbMax = vec3(1.0, 1.0, 1.0);
  if (xMax > 1.0) {
    // The point is inside in one of the outer quadrants ("squash zone")
    if (xAbs.x >= xAbs.y && xAbs.x >= xAbs.z) {
      aabbMin.x = xP.x > 0.0 ? 1.0 : -2.0; // argmax = 0
      aabbMax.x = xP.x > 0.0 ? 2.0 : -1.0;
    } else if (xAbs.y >= xAbs.x && xAbs.y >= xAbs.z) {
      aabbMin.y = xP.y > 0.0 ? 1.0 : -2.0; // argmax = 1
      aabbMax.y = xP.y > 0.0 ? 2.0 : -1.0;
    } else {
      aabbMin.z = xP.z > 0.0 ? 1.0 : -2.0; // argmax = 2
      aabbMax.z = xP.z > 0.0 ? 2.0 : -1.0;
    }
  }

  // Estimate the direction of the ray in contracted space by computing the
  // vector difference with two different t-values that are guanteed to
  // correspond to points within the current quadrant
  r.oContracted = contract(xP);
  vec3 zQ = contract(o + tQ * d);
  r.dContracted = normalize(zQ - r.oContracted);

  // When is the ray exiting the current quadrant? We need this value in order
  // to know when we enter a new quadrant or when to terminate ray marching.
  // Note that im findTraversedQuadrants word-space t-values are computed,
  // while we compute here contraction-space t-values. The world-space t-values
  // are needed to robustly obtain two points (tP and tQ) that are guranteed to
  // lie within a quadrant. With help of these values we can generate two
  // points in contracted space from which we can estimate the ray origin and
  // direction in contracted space. However, once we raymarch in contracted
  // space we need the contraction-space t-value to conveniently check whether 
  // we are still in the same quadrant. Alternatively, one could convert the
  // contraction-space point to a world-space point and estimate a world space
  // t-value, but this has been found to be numerically unstable.
  r.quadrantTMinMaxContracted = rayAabbIntersection(
    aabbMin, aabbMax, r.oContracted, 1.0 / r.dContracted);
  return r;
}


struct OccupancyQueryResults {
  bool inEmptySpace;
  float tBlockMax;
};

OccupancyQueryResults queryOccupancyGrid(
    vec3 z, vec3 minPosition, vec3 oContracted,
    vec3 invDContracted, highp sampler3D occupancyGrid,
    float voxelSizeOccupancy, vec3 gridSizeOccupancy) {
  vec3 posOccupancy = (z - minPosition) / voxelSizeOccupancy;
  vec3 blockMin = floor(posOccupancy);
  vec3 blockMax = floor(posOccupancy) + 1.0;
  float occupancy = 255.0 * texture(
    occupancyGrid, (blockMin + 0.5) / gridSizeOccupancy).r;

  OccupancyQueryResults r;
  r.inEmptySpace = occupancy == 0.0;
  #ifdef USE_BITS
  if (!r.inEmptySpace) {
    vec3 relativePosOccupancy = posOccupancy - blockMin;
    float bit = 1.0;
    if (relativePosOccupancy.z > 0.5) {
      bit *= 2.0;
      blockMin.z += 0.5;
    }
    if (relativePosOccupancy.y > 0.5) {
      bit *= 4.0;
      blockMin.y += 0.5;
    }
    if (relativePosOccupancy.x > 0.5) {
      bit *= 16.0;
      blockMin.x += 0.5;
    }
    blockMax = blockMin + 0.5;
    r.inEmptySpace = mod(occupancy / bit, 2.0) < 1.0;
  }
  #endif
  blockMin = blockMin * voxelSizeOccupancy + minPosition;
  blockMax = blockMax * voxelSizeOccupancy + minPosition;
  r.tBlockMax = rayAabbIntersection(
    blockMin, blockMax, oContracted, invDContracted).y;
  return r;
}

#define QUERY_DISTANCE_GRID(distanceGrid, voxelSizeDistance, gridSizeDistance)\
{\
  vec3 blockMin = floor((z - kMinPosition) / voxelSizeDistance);\
  float distance = 255.0 * voxelSizeDistance * texture(\
    distanceGrid, (blockMin + 0.5) / gridSizeDistance).r;\
  if (distance > 0.0) {\
    tContracted += distance;\
    continue;\
  }\
}

#define QUERY_OCCUPANCY_GRID(tBlockMax_L, occupancyGrid, voxelSizeOccupancy, gridSizeOccupancy)\
if (tContracted > tBlockMax_L) {\
  occupancyQueryResults =\
    queryOccupancyGrid(z, kMinPosition, r.oContracted, invDContracted,\
                        occupancyGrid, voxelSizeOccupancy, gridSizeOccupancy);\
  tBlockMax_L = occupancyQueryResults.tBlockMax;\
  if (occupancyQueryResults.inEmptySpace) {\
    tContracted = max(tContracted, tBlockMax_L) + 0.5 * stepSizeContracted;\
    continue;\
  }\
}

void main() {
  // See the DisplayMode enum at the top of this file.
  // Runs the full model with view dependence.
  const int DISPLAY_NORMAL = 0;
  // Disables the view-dependence network.
  const int DISPLAY_DIFFUSE = 1;
  // Only shows the latent features.
  const int DISPLAY_FEATURES = 2;
  // Only shows the view dependent component.
  const int DISPLAY_VIEW_DEPENDENT = 3;
  // Only shows the coarse block grid.
  const int DISPLAY_COARSE_GRID = 4;

  // Set up the ray parameters in world space.
  float nearWorld = 1.0;
  vec3 originWorld = vOrigin;
  vec3 directionWorld = vDirection;

  vec3 blockGridSize;
  {
    ivec3 iGridSize = ivec3(round(kSparseGridGridSize));
    int iBlockSize = int(round(kDataBlockSize));
    ivec3 iBlockGridBlocks = (iGridSize + iBlockSize - 1) / iBlockSize;
    ivec3 iBlockGridSize = iBlockGridBlocks * iBlockSize;
    blockGridSize = vec3(iBlockGridSize);
  }

  float[5] listQuadrantTMax = findTraversedQuadrants(
    originWorld, directionWorld, nearWorld);

  QuadrantSetupResults r = quadrantSetup(
    originWorld,
    directionWorld,
    nearWorld,
    mix(nearWorld, listQuadrantTMax[0], 0.5)
  );
  float tContracted = 0.0;
  int quadrantIndex = 1;

#ifdef USE_DISTANCE_GRID
  float tBlockMax_L0 = -INF;
#else
  float tBlockMax_L0 = -INF;
  float tBlockMax_L1 = -INF;
  float tBlockMax_L2 = -INF;
#ifndef USE_BITS
  float tBlockMax_L3 = -INF;
  float tBlockMax_L4 = -INF;
#endif
#endif

  float visibility = 1.0;
  vec3 accumulatedColor = vec3(0.0);
  vec4 accumulatedFeatures = vec4(0.0);
  int step = 0;

#ifdef USE_FEATURE_CONCAT
  vec3 accumulatedCoarseColor = vec3(0.0);
  vec4 accumulatedCoarseFeatures = vec4(0.0);
#endif

  #define GRID_SIZE kTriplaneGridSize
  #define VOXEL_SIZE kTriplaneVoxelSize

  int maxStep = kStepMult * int(ceil(length(GRID_SIZE)));
  float origStepSizeContracted = VOXEL_SIZE / float(kStepMult);

  while (step < maxStep && visibility > 1.0 / 255.0) {
    step++;
#ifdef LARGER_STEPS_WHEN_OCCLUDED
    float stepSizeContracted = origStepSizeContracted *
        mix(8.0, 1.0, min(1.0, visibility / 0.66));
#else
    float stepSizeContracted = origStepSizeContracted;
#endif

    // Check if the ray is exiting the current quadrant.
    if (tContracted > r.quadrantTMinMaxContracted.y) {
      {
        // Check if we hit an outer wall. If so, we can terminate the ray as
        // it won't enter another quadrant.
        vec3 z = r.oContracted + r.quadrantTMinMaxContracted.y * r.dContracted;
        if (max3(abs(z)) >= 2.0 - 1e-3) break;
      }

      // Setup ray in the new quadrant. First, check if our ray will
      // immediately skip past the next quadrant (i.e. near the line
      // that is at the intersection of three or more quadrants).
      if (listQuadrantTMax[quadrantIndex - 1] >=
        listQuadrantTMax[quadrantIndex] - stepSizeContracted) {
        quadrantIndex++;
      }

      // Next, using the precomputed t-values we can find two points that are
      // guranteed to lie within the new quadrant.
      r = quadrantSetup(
        originWorld,
        directionWorld,
        mix(listQuadrantTMax[quadrantIndex - 1],
            listQuadrantTMax[quadrantIndex], 0.1),
        mix(listQuadrantTMax[quadrantIndex - 1],
            listQuadrantTMax[quadrantIndex], 0.9)
      );
      tContracted = r.quadrantTMinMaxContracted.x + 0.5 * stepSizeContracted;
      quadrantIndex++;

      // Reset all tMax values to force occupancy queries
#ifdef USE_DISTANCE_GRID
      tBlockMax_L0 = -INF;
#else
      tBlockMax_L0 = -INF;
      tBlockMax_L1 = -INF;
      tBlockMax_L2 = -INF;
#ifndef USE_BITS
      tBlockMax_L3 = -INF;
      tBlockMax_L4 = -INF;
#endif
#endif
    }

    // Position of current sample in contracted space
    vec3 z = r.oContracted + tContracted * r.dContracted;

    // Hierarchical empty space skipping
    {
      vec3 invDContracted = 1.0 / r.dContracted;
      OccupancyQueryResults occupancyQueryResults;
#ifdef USE_DISTANCE_GRID
      QUERY_DISTANCE_GRID(
        distanceGrid, kVoxelSizeDistance, kGridSizeDistance)
      QUERY_OCCUPANCY_GRID(tBlockMax_L0, occupancyGrid_L0,
        kVoxelSizeOccupancy_L0, kGridSizeOccupancy_L0)
#else
      QUERY_OCCUPANCY_GRID(tBlockMax_L0, occupancyGrid_L0,
        kVoxelSizeOccupancy_L0, kGridSizeOccupancy_L0)
      QUERY_OCCUPANCY_GRID(tBlockMax_L1, occupancyGrid_L1,
        kVoxelSizeOccupancy_L1, kGridSizeOccupancy_L1)
      QUERY_OCCUPANCY_GRID(tBlockMax_L2, occupancyGrid_L2,
        kVoxelSizeOccupancy_L2, kGridSizeOccupancy_L2)
#ifndef USE_BITS
      QUERY_OCCUPANCY_GRID(tBlockMax_L3, occupancyGrid_L3,
        kVoxelSizeOccupancy_L3, kGridSizeOccupancy_L3)
      QUERY_OCCUPANCY_GRID(tBlockMax_L4, occupancyGrid_L4,
        kVoxelSizeOccupancy_L4, kGridSizeOccupancy_L4)
#endif
#endif
    }

    // We are in occupied space, so there is no need to check if the atlas
    // fetch is out of bounds. Instead, directly compute grid positions for
    // the sparse 3D grid and on the triplane planes.
    vec3 atlasUvw;
    {
      vec3 posSparseGrid = (z - kMinPosition) / kSparseGridVoxelSize - 0.5;
      vec3 atlasBlockMin =
        floor(posSparseGrid / kDataBlockSize) * kDataBlockSize;
      vec3 atlasBlockIndex = 255.0 * texture(
        sparseGridBlockIndices,
        (atlasBlockMin + 0.5 * kDataBlockSize) / blockGridSize
      ).xyz;

      if (atlasBlockIndex.x > 254.0) {
        tContracted += stepSizeContracted;
        continue;
      }

      atlasUvw = (1.0 / atlasSize) * (
        // Half voxel offset.
        0.5 +
        // Position within the block.
        clamp(posSparseGrid - atlasBlockMin, 0.0, kDataBlockSize) +
        // The block location.
        atlasBlockIndex * (kDataBlockSize + 1.0)
      );

      if (displayMode == DISPLAY_COARSE_GRID) {
        accumulatedColor = atlasBlockIndex * (kDataBlockSize + 1.0) / atlasSize;
        visibility = 0.0;
        continue;
      }
    }

    // First fetch all densities and weight.
    vec2 sparse_density_weight = texture(sparseGridDensity, atlasUvw).xw;
    sparse_density_weight.x = denormalize(
      sparse_density_weight.x, kRangeDensityMin, kRangeDensityMax);
    sparse_density_weight.y = denormalize(
      sparse_density_weight.y, kRangeFeaturesMin, kRangeFeaturesMax);

    // Fetch triplane density.
    vec3 posTriplaneGrid =
      (z - kMinPosition) / (kTriplaneVoxelSize * kTriplaneGridSize.x);

    float triplane_density =
      texture(planeDensity, vec3(posTriplaneGrid.yz, 0.0)).x +
      texture(planeDensity, vec3(posTriplaneGrid.xz, 1.0)).x +
      texture(planeDensity, vec3(posTriplaneGrid.xy, 2.0)).x;

    // Since we do denormalize(sum) instead of sum(denormalize), we need to
    // add the missing bias manually.
    triplane_density = 2.0 * kRangeDensityMin +
      denormalize(triplane_density, kRangeDensityMin, kRangeDensityMax);

    // Now, compute alpha from the segment length and the fetch density value.
    float alpha = 0.0;
    {
      // Calculate where the next sample would land in order to compute the
      // step size in world space (required for density-to-alpha conversion)
      // make sure not to shoot ouf the current quadrant
      float tContractedNext = min(
        tContracted + stepSizeContracted, r.quadrantTMinMaxContracted.y);
      // Position of the next sample in contracted space
      vec3 zNext = r.oContracted + tContractedNext * r.dContracted;
      float stepSizeWorld = (1.0 / kSubmodelScale) * distance(
        inverseContract(zNext), inverseContract(z));

      // Activate density and convert density to alpha.
      #ifdef USE_FEATURE_GATING
      float density = densityActivation(
        sparse_density_weight.x + sparse_density_weight.y * triplane_density
      );
      #else
      float density = densityActivation(
        sparse_density_weight.x + triplane_density
      );
      #endif
      alpha = densityToAlpha(density, stepSizeWorld);
    }

    // Save bandwidth: Only fetch RGBFFFF (7 bytes) if alpha is non-negligible.
#ifdef ONLY_DENSITY_FETCHES
    if (false) {
#else
    if (alpha > 0.25 / 255.0) {
#endif
      vec3 sparse_rgb = denormalize(
        texture(sparseGridRgb, atlasUvw).rgb,
        kRangeFeaturesMin,
        kRangeFeaturesMax
      );
      vec4 sparse_features = vec4(
        denormalize(
          texture(sparseGridFeatures, atlasUvw).rgb,
          kRangeFeaturesMin,
          kRangeFeaturesMax),
        sparse_density_weight.y
      );

      vec3 triplane_rgb =
        texture(planeRgb, vec3(posTriplaneGrid.yz, 0.0)).rgb +
        texture(planeRgb, vec3(posTriplaneGrid.xz, 1.0)).rgb +
        texture(planeRgb, vec3(posTriplaneGrid.xy, 2.0)).rgb;

      vec4 triplane_features =
        texture(planeFeatures, vec3(posTriplaneGrid.yz, 0.0)) +
        texture(planeFeatures, vec3(posTriplaneGrid.xz, 1.0)) +
        texture(planeFeatures, vec3(posTriplaneGrid.xy, 2.0));

      // Since we do denormalize(sum) instead of sum(denormalize), we need to
      // add the missing bias manually.
      triplane_features = 2.0 * kRangeFeaturesMin +
        denormalize(triplane_features, kRangeFeaturesMin, kRangeFeaturesMax);
      triplane_rgb = 2.0 * kRangeFeaturesMin +
        denormalize(triplane_rgb, kRangeFeaturesMin, kRangeFeaturesMax);

      { // Apply the activation functions and accumulate.
        float weight = visibility * alpha;
#ifdef USE_FEATURE_GATING
        accumulatedColor += weight * sigmoid(
          sparse_rgb + sparse_features.a * triplane_rgb);
        accumulatedFeatures += weight * sigmoid(
          sparse_features + sparse_features.a * triplane_features);
#else
        accumulatedColor += weight * sigmoid(
          sparse_rgb + triplane_rgb);
        accumulatedFeatures += weight * sigmoid(
          sparse_features + triplane_features);
#endif

#ifdef USE_FEATURE_CONCAT
        accumulatedCoarseColor += weight * sigmoid(sparse_rgb);
        accumulatedCoarseFeatures += weight * sigmoid(sparse_features);
#endif
      }
    }

    visibility *= 1.0 - alpha;
    tContracted += stepSizeContracted;
  }

  if (displayMode == DISPLAY_FEATURES) {
    accumulatedColor = accumulatedFeatures.rgb;
  }

  // Run view-dependency network and composite onto a white background.
  #ifdef USE_VFR
  if ((displayMode == DISPLAY_NORMAL ||
       displayMode == DISPLAY_VIEW_DEPENDENT)) {
    accumulatedColor = evaluateNetwork(
      accumulatedColor,
      accumulatedFeatures,
      #ifdef USE_FEATURE_CONCAT
      accumulatedCoarseColor,
      accumulatedCoarseFeatures,
      #endif
      worldspaceROpengl * normalize(vDirection)
    );
  }
  accumulatedColor = visibility * kBackgroundColor + accumulatedColor;
  #else // SNeRG view-dependence, which first composits onto the background.
  if ((displayMode == DISPLAY_NORMAL ||
       displayMode == DISPLAY_VIEW_DEPENDENT)) {
    vec3 specular = evaluateNetwork(
      visibility * kBackgroundColor + accumulatedColor,
      accumulatedFeatures,
      #ifdef USE_FEATURE_CONCAT
      accumulatedCoarseColor,
      accumulatedCoarseFeatures,
      #endif
      worldspaceROpengl * normalize(vDirection)
    );
    if (displayMode == DISPLAY_VIEW_DEPENDENT) {
      accumulatedColor = vec3(0.0);
    }
    accumulatedColor += specular;
  }
  #endif
  gl_FragColor = vec4(accumulatedColor, 1.0);
}
