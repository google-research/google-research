#import sorting

@group(0) @binding(0) var<storage, read_write> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> reduced: array<u32>;

var<workgroup> sums: array<u32, sorting::WG>;
var<workgroup> num_keys_wg: u32;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) gid: vec3<u32>,
) {
    if local_id.x == 0u {
        num_keys_wg = num_keys_arr[0];
    }
    let num_keys = workgroupUniformLoad(&num_keys_wg);
    // let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    let group_id = gid.x;

    if group_id >= num_reduce_wgs {
        return;
    }

    let num_reduce_wg_per_bin = num_reduce_wgs / sorting::BIN_COUNT;
    let bin_id = group_id / num_reduce_wg_per_bin;

    let bin_offset = bin_id * num_wgs;
    let base_index = (group_id % num_reduce_wg_per_bin) * sorting::BLOCK_SIZE;
    var sum = 0u;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = base_index + i * sorting::WG + local_id.x;
        if data_index < num_wgs {
            sum += counts[bin_offset + data_index];
        }
    }
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x < ((sorting::WG / 2u) >> i) {
            sum += sums[local_id.x + ((sorting::WG / 2u) >> i)];
            sums[local_id.x] = sum;
        }
    }
    if local_id.x == 0u {
        reduced[group_id] = sum;
    }
}
