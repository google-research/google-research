#import helpers;

@group(0) @binding(0) var<storage, read_write> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read_write> projected_splats: array<helpers::ProjectedSplat>;
@group(0) @binding(2) var<storage, read_write> cum_tiles_hit: array<u32>;
@group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<u32>;

@compute
@workgroup_size(helpers::MAIN_WG, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let compact_gid = global_id.x;

    if compact_gid >= uniforms.num_visible {
        return;
    }

    let projected = projected_splats[compact_gid];
    // get the tile bbox for gaussian
    let xy = vec2f(projected.x, projected.y);
    let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
    let opac = projected.a;

    let radius = helpers::radius_from_conic(conic, opac);
    let tile_bounds = uniforms.tile_bounds;

    let tile_minmax = helpers::get_tile_bbox(xy, radius, tile_bounds);
    let tile_min = tile_minmax.xy;
    let tile_max = tile_minmax.zw;

    // Get exclusive prefix sum of tiles hit.
    var isect_id = 0u;
    if compact_gid > 0u {
        isect_id = cum_tiles_hit[compact_gid - 1u];
    }

    for (var ty = tile_min.y; ty < tile_max.y; ty++) {
        for (var tx = tile_min.x; tx < tile_max.x; tx++) {
            if helpers::can_be_visible(vec2u(tx, ty), xy, conic, opac) && isect_id < arrayLength(&tile_id_from_isect) {
                let tile_id = tx + ty * tile_bounds.x; // tile within image
                tile_id_from_isect[isect_id] = tile_id;
                compact_gid_from_isect[isect_id] = compact_gid;
                isect_id++; // handles gaussians that hit more than one tile
            }
        }
    }
}
