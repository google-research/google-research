#import grads;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read_write> global_from_compact_gid: array<u32>;

@group(0) @binding(2) var<storage, read_write> raw_opacities: array<f32>;
@group(0) @binding(3) var<storage, read_write> means: array<helpers::PackedVec3>;
@group(0) @binding(4) var<storage, read_write> v_colors: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> v_xy_local: array<vec2f>;

@group(0) @binding(6) var<storage, read_write> v_coeffs: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_opacs: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_xy_global: array<vec2f>;

fn sh_coeffs_to_color_fast_vjp(
    degree: u32,
    viewdir: vec3f,
    v_colors: vec3f,
) -> ShCoeffs {
    var v_coeffs = ShCoeffs();

    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    v_coeffs.b0_c0 = 0.2820947917738781f * v_colors;

    if (degree < 1) {
        return v_coeffs;
    }
    let norm = normalize(viewdir);
    let x = viewdir.x;
    let y = viewdir.y;
    let z = viewdir.z;

    let fTmp0A = 0.48860251190292f;
    v_coeffs.b1_c0 = -fTmp0A * y * v_colors;
    v_coeffs.b1_c1 = fTmp0A * z * v_colors;
    v_coeffs.b1_c2 = -fTmp0A * x * v_colors;

    if (degree < 2) {
        return v_coeffs;
    }

    let z2 = z * z;
    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.f * x * y;
    let pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;
    v_coeffs.b2_c0 = pSH4 * v_colors;
    v_coeffs.b2_c1 = pSH5 * v_colors;
    v_coeffs.b2_c2 = pSH6 * v_colors;
    v_coeffs.b2_c3 = pSH7 * v_colors;
    v_coeffs.b2_c4 = pSH8 * v_colors;

    if (degree < 3) {
        return v_coeffs;
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9  = fTmp2A * fS2;
    v_coeffs.b3_c0 = pSH9 * v_colors;
    v_coeffs.b3_c1 = pSH10 * v_colors;
    v_coeffs.b3_c2 = pSH11 * v_colors;
    v_coeffs.b3_c3 = pSH12 * v_colors;
    v_coeffs.b3_c4 = pSH13 * v_colors;
    v_coeffs.b3_c5 = pSH14 * v_colors;
    v_coeffs.b3_c6 = pSH15 * v_colors;
    return v_coeffs;

    // if (degree < 4) {
    //     return v_coeffs;
    // }
    // let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    // let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    // let fTmp2B = -1.770130769779931f * z;
    // let fTmp3A = 0.6258357354491763f;
    // let fC3 = x * fC2 - y * fS2;
    // let fS3 = x * fS2 + y * fC2;
    // let pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    // let pSH21 = fTmp0D * x;
    // let pSH19 = fTmp0D * y;
    // let pSH22 = fTmp1C * fC1;
    // let pSH18 = fTmp1C * fS1;
    // let pSH23 = fTmp2B * fC2;
    // let pSH17 = fTmp2B * fS2;
    // let pSH24 = fTmp3A * fC3;
    // let pSH16 = fTmp3A * fS3;
    // v_coeffs.b4_c0 = pSH16 * v_colors;
    // v_coeffs.b4_c1 = pSH17 * v_colors;
    // v_coeffs.b4_c2 = pSH18 * v_colors;
    // v_coeffs.b4_c3 = pSH19 * v_colors;
    // v_coeffs.b4_c4 = pSH20 * v_colors;
    // v_coeffs.b4_c5 = pSH21 * v_colors;
    // v_coeffs.b4_c6 = pSH22 * v_colors;
    // v_coeffs.b4_c7 = pSH23 * v_colors;
    // v_coeffs.b4_c8 = pSH24 * v_colors;
    // return v_coeffs;
}

struct ShCoeffs {
    b0_c0: vec3f,

    b1_c0: vec3f,
    b1_c1: vec3f,
    b1_c2: vec3f,

    b2_c0: vec3f,
    b2_c1: vec3f,
    b2_c2: vec3f,
    b2_c3: vec3f,
    b2_c4: vec3f,

    b3_c0: vec3f,
    b3_c1: vec3f,
    b3_c2: vec3f,
    b3_c3: vec3f,
    b3_c4: vec3f,
    b3_c5: vec3f,
    b3_c6: vec3f,

    // b4_c0: vec3f,
    // b4_c1: vec3f,
    // b4_c2: vec3f,
    // b4_c3: vec3f,
    // b4_c4: vec3f,
    // b4_c5: vec3f,
    // b4_c6: vec3f,
    // b4_c7: vec3f,
    // b4_c8: vec3f,
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1) * (degree + 1);
}

fn write_coeffs(base_id: ptr<function, u32>, val: vec3f) {
    v_coeffs[*base_id + 0] = val.x;
    v_coeffs[*base_id + 1] = val.y;
    v_coeffs[*base_id + 2] = val.z;
    *base_id += 3u;
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn v_sigmoid(x: f32) -> f32 {
    return sigmoid(x) * (1.0 - sigmoid(x));
}

@compute
@workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let compact_gid = gid.x;

    if compact_gid >= uniforms.num_visible {
        return;
    }

    // Load colors gradients.
    var v_color = v_colors[compact_gid];

    // Convert RGB to global SH gradients.
    let global_gid = global_from_compact_gid[compact_gid];

    let mean = helpers::as_vec(means[global_gid]);
    let viewdir = normalize(-uniforms.viewmat[3].xyz - mean);

    let sh_degree = uniforms.sh_degree;
    let v_coeff = sh_coeffs_to_color_fast_vjp(sh_degree, viewdir, v_color.xyz);
    let num_coeffs = num_sh_coeffs(sh_degree);
    var base_id = global_gid * num_coeffs * 3;

    write_coeffs(&base_id, v_coeff.b0_c0);
    if sh_degree > 0 {
        write_coeffs(&base_id, v_coeff.b1_c0);
        write_coeffs(&base_id, v_coeff.b1_c1);
        write_coeffs(&base_id, v_coeff.b1_c2);
        if sh_degree > 1 {
            write_coeffs(&base_id, v_coeff.b2_c0);
            write_coeffs(&base_id, v_coeff.b2_c1);
            write_coeffs(&base_id, v_coeff.b2_c2);
            write_coeffs(&base_id, v_coeff.b2_c3);
            write_coeffs(&base_id, v_coeff.b2_c4);
            if sh_degree > 2 {
                write_coeffs(&base_id, v_coeff.b3_c0);
                write_coeffs(&base_id, v_coeff.b3_c1);
                write_coeffs(&base_id, v_coeff.b3_c2);
                write_coeffs(&base_id, v_coeff.b3_c3);
                write_coeffs(&base_id, v_coeff.b3_c4);
                write_coeffs(&base_id, v_coeff.b3_c5);
                write_coeffs(&base_id, v_coeff.b3_c6);
                // if sh_degree > 3 {
                //     write_coeffs(&base_id, v_coeff.b4_c0);
                //     write_coeffs(&base_id, v_coeff.b4_c1);
                //     write_coeffs(&base_id, v_coeff.b4_c2);
                //     write_coeffs(&base_id, v_coeff.b4_c3);
                //     write_coeffs(&base_id, v_coeff.b4_c4);
                //     write_coeffs(&base_id, v_coeff.b4_c5);
                //     write_coeffs(&base_id, v_coeff.b4_c6);
                //     write_coeffs(&base_id, v_coeff.b4_c7);
                //     write_coeffs(&base_id, v_coeff.b4_c8);
                // }
            }
        }
    }

    // Transform alpha gradient to opacity gradient.
    let raw_opac = raw_opacities[global_gid];
    let v_opac = v_color.w * v_sigmoid(raw_opac);
    v_opacs[global_gid] = v_opac;

    // Scatter the xy gradients, as later operations need them to be global.
    v_xy_global[global_gid] = v_xy_local[compact_gid];
}
