const TILE_WIDTH: u32 = 16u;
// Nb: TILE_SIZE should be <= 512 for max compatibilty.
const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

const MAIN_WG: u32 = 256u;

struct RenderUniforms {
    // View matrix transform world to view position.
    // Offset 0.
    viewmat: mat4x4f,
    // Focal of camera (fx, fy)
    // offset 64
    focal: vec2f,
    // Img resolution (w, h)
    // offset 72
    img_size: vec2u,
    // offset 80
    tile_bounds: vec2u,
    // Camera center (cx, cy).
    // offset 88
    pixel_center: vec2f,
    // Background color of the scene.
    // Offset 96.
    background: vec4f,
    // Degree of sh coeffecients used.
    // Offset 112
    sh_degree: u32,
#ifdef UNIFORM_WRITE
    // Number of visible gaussians, written by project_forward.
    // This needs to be non-atomic for other kernels as you can't have
    // read-only atomic data.
    num_visible: atomic<u32>,
#else
    // Number of visible gaussians.
    // Offset 116
    num_visible: u32,
#endif
    // Offset 120
    total_splats: u32,
    // Offset 124
    padding: u32,
}

// nb: this struct has a bunch of padding but that's probably fine.
struct ProjectedSplat {
    x: f32,
    y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

struct PackedVec3 {
    x: f32,
    y: f32,
    z: f32,
}

fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    let min = vec2u(clamp(vec2i(center - dims), vec2i(0), vec2i(bounds)));
    let max = vec2u(clamp(vec2i(center + dims + vec2f(1.0)), vec2i(0), vec2i(bounds)));
    return vec4u(min, max);
}

fn get_tile_bbox(pix_center: vec2f, pix_radius: u32, tile_bounds: vec2u) -> vec4u {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    let tile_center = pix_center / f32(TILE_WIDTH);
    let tile_radius = f32(pix_radius) / f32(TILE_WIDTH);

    return get_bbox(tile_center, vec2f(tile_radius, tile_radius), tile_bounds);
}

// device helper to get 3D covariance from scale and quat parameters
fn quat_to_rotmat(quat: vec4f) -> mat3x3f {
    // quat to rotation matrix
    let w = quat.x;
    let x = quat.y;
    let y = quat.z;
    let z = quat.w;

    // See https://www.songho.ca/opengl/gl_quaternion.html
    return mat3x3f(
        vec3f(
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        vec3f(
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ),
        vec3f(
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y)
        ),
    );
}

fn scale_to_mat(scale: vec3f) -> mat3x3f {
    return mat3x3(
        vec3f(scale.x, 0.0, 0.0),
        vec3f(0.0, scale.y, 0.0),
        vec3f(0.0, 0.0, scale.z)
    );
}

fn project_pix(fxfy: vec2f, p_view: vec3f, pp: vec2f) -> vec2f {
    let p_proj = p_view.xy / (p_view.z + 1e-6f);
    return p_proj * fxfy + pp;
}

fn calc_cov2d(focal: vec2f, img_size: vec2u, pixel_center: vec2f, viewmat: mat4x4f, p_view: vec3f, scale: vec3f, quat: vec4f) -> vec3f {
    let tan_fov = 0.5 * vec2f(img_size.xy) / focal;

    let lims_neg = pixel_center / focal + 0.3f * tan_fov;
    let lims_pos = (vec2f(img_size.xy) - pixel_center) / focal + 0.3f * tan_fov;

    let lims = 1.3 * tan_fov;

    // Get ndc coords +- clipped to the frustum.
    let t = p_view.z * clamp(p_view.xy / p_view.z, -lims_neg, lims_pos);

    var M = quat_to_rotmat(quat);
    M[0] *= scale.x;
    M[1] *= scale.y;
    M[2] *= scale.z;
    var V = M * transpose(M);

    let J = mat3x3f(
        vec3f(focal.x, 0.0, 0.0),
        vec3f(0.0, focal.y, 0.0),
        vec3f(-focal * t / p_view.z, 0.0)
    ) * (1.0 / p_view.z);

    let W = mat3x3f(viewmat[0].xyz, viewmat[1].xyz, viewmat[2].xyz);
    let T = J * W;
    let cov = T * V * transpose(T);

    let c00 = cov[0][0] + COV_BLUR;
    let c11 = cov[1][1] + COV_BLUR;
    let c01 = cov[0][1];

    // add a little blur along axes and save upper triangular elements
    let cov2d = vec3f(c00, c01, c11);
    return cov2d;
}

fn cov_to_conic(cov2d: vec3f) -> vec3f {
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    return vec3f(cov2d.z, -cov2d.y, cov2d.x) / det;
}

// TODO: Is this 0.3 good? Make this configurable?
const COV_BLUR: f32 = 0.3;

fn cov_compensation(cov2d: vec3f) -> f32 {
    let cov_orig = cov2d - vec3f(COV_BLUR, 0.0, COV_BLUR);
    let det_orig = cov_orig.x * cov_orig.z - cov_orig.y * cov_orig.y;
    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    return sqrt(max(0.0, det_orig / det));
}

fn calc_sigma(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    let delta = pixel_coord - xy;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
}

fn calc_vis(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    return exp(-calc_sigma(pixel_coord, conic, xy));
}

fn inverse(m: mat2x2f) -> mat2x2f {
    let det = determinant(m);
    return mat2x2f(
        m[1][1] / det, -m[1][0] / det,
        -m[0][1] / det, m[0][0] / det
    );
}

fn radius_from_conic(conic: vec3f, opac: f32) -> u32 {
    // Calculate tbe pixel radius.
    // Original implementation:
    let det = 1.0 / (conic.x * conic.z - conic.y * conic.y);
    let cov2d = vec3f(conic.z, -conic.y, conic.x) * det;
    let b = 0.5 * (cov2d.x + cov2d.z);
    let v1 = b + sqrt(max(0.1f, b * b - det));
    let v2 = b - sqrt(max(0.1f, b * b - det));
    let radius = 3.0 * sqrt(max(0.0, max(v1, v2)));
    return u32(ceil(radius));

    // I think we can do better and derive an exact bound when we hit some eps threshold.
    // Also, we should take into account the opoacity of the gaussian.
    // So, opac * exp(-0.5 * x^T Sigma^-1 x) = eps  (with eps being e.g. 1.0 / 255.0).
    // x^T Sigma^-1 x = -2 * log(eps / opac)
    // Find maximal |x| using quadratic form
    // |x|^2 = c / lambd_min.
    // // Now solve for maximal |r| such that min alpha = 1.0 / 255.0.
    // //
    // // we actually go for 2.0 / 255.0 or so to match the cutoff from gsplat better.
    // // maybe can be more precise here if we don't need 1:1 compat with gsplat anymore.
    // let trace = conic.x + conic.z;
    // let determinant = conic.x * conic.z - conic.y * conic.y;
    // let l_min = 0.5 * (trace - sqrt(trace * trace - 4 * determinant));
    // let eps_const = -2.0 * log(1.0 / (opac * 255.0));
    // return sqrt(eps_const / l_min);
}

fn check_edge(p1: vec2f, p2: vec2f, ellipse_center: vec2f, ellipse_conic: mat2x2f) -> bool {
    let edge = p2 - p1;
    let f = p1 - ellipse_center;
    let a = dot(edge * ellipse_conic, edge);
    let b = 2.0 * dot(f * ellipse_conic, edge);
    let c = dot(f * ellipse_conic, f) - 1.0;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return false;
    }

    let sqrt_discriminant = sqrt(discriminant);
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);
    return (t1 >= 0.0 && t1 <= 1.0) || (t2 >= 0.0 && t2 <= 1.0);
}

fn ellipse_intersects_aabb(box_pos: vec2f, box_extent: vec2f, ellipse_center: vec2f, ellipse_conic: mat2x2f) -> bool {
    let d = ellipse_center - box_pos;

    // Check if ellipse center is inside AABB.
    if all(abs(d) <= box_extent) {
        return true;
    }

    // Determine the nearest corner
    let corner_sign = sign(d);
    let nearest_corner = box_pos + corner_sign * box_extent;

    // Check if the nearest corner is inside the ellipse
    let cp = nearest_corner - ellipse_center;
    if dot(cp * ellipse_conic, cp) <= 1.0 {
        return true;
    }

    // Check the two edges adjacent to the nearest corner
    let edge1_end = nearest_corner - vec2f(corner_sign.x * 2.0 * box_extent.x, 0.0);
    let edge2_end = nearest_corner - vec2f(0.0, corner_sign.y * 2.0 * box_extent.y);

    return check_edge(nearest_corner, edge1_end, ellipse_center, ellipse_conic) ||
           check_edge(nearest_corner, edge2_end, ellipse_center, ellipse_conic);
}

fn can_be_visible(tile: vec2u, xy: vec2f, conic: vec3f, opac: f32) -> bool {
    // return true;

    // opac * exp(-sigma) == 1.0 / 255.0
    // exp(-sigma) == 1.0 / (opac * 255.0)
    // -sigma == log(1.0 / (opac * 255.0))
    // sigma == log(opac * 255.0);
    let sigma = log(opac * 255.0);
    if sigma <= 0.0 {
        return false;
    }
    let conic_scaled = conic / (2.0 * sigma);
    let tile_extent = vec2f(f32(TILE_WIDTH)) / 2.0;
    let tile_center = vec2f(tile * TILE_WIDTH) + tile_extent;
    return ellipse_intersects_aabb(tile_center, tile_extent, xy, mat2x2f(conic_scaled.x, conic_scaled.y, conic_scaled.y, conic_scaled.z));
}

fn ceil_div(a: u32, b: u32) -> u32 {
    return (a + b - 1) / b;
}

fn as_vec(packed: PackedVec3) -> vec3f {
    return vec3f(packed.x, packed.y, packed.z);
}

fn as_packed(vec: vec3f) -> PackedVec3 {
    return PackedVec3(vec.x, vec.y, vec.z);
}
