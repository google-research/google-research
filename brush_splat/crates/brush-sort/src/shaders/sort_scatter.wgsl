#import sorting

struct Uniforms {
    shift: u32,
}

@group(0) @binding(0) var<storage, read_write> config: Uniforms;
@group(0) @binding(1) var<storage, read_write> num_keys_arr: array<u32>;
@group(0) @binding(2) var<storage, read_write> src: array<u32>;
@group(0) @binding(3) var<storage, read_write> values: array<u32>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_values: array<u32>;

var<workgroup> lds_sums: array<u32, sorting::WG>;
var<workgroup> lds_scratch: array<u32, sorting::WG>;
var<workgroup> bin_offset_cache: array<u32, sorting::WG>;
var<workgroup> local_histogram: array<atomic<u32>, sorting::BIN_COUNT>;

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

    let group_id = gid.x;

    if group_id >= num_wgs {
        return;
    }

    if local_id.x < sorting::BIN_COUNT {
        bin_offset_cache[local_id.x] = counts[local_id.x * num_wgs + group_id];
    }
    workgroupBarrier();
    let wg_block_start = sorting::BLOCK_SIZE * group_id;
    let block_index = wg_block_start + local_id.x;
    var data_index = block_index;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        if local_id.x < sorting::BIN_COUNT {
            local_histogram[local_id.x] = 0u;
        }
        var local_key = ~0u;
        var local_value = 0u;

        if data_index < num_keys {
            local_key = src[data_index];
            local_value = values[data_index];
        }

        for (var bit_shift = 0u; bit_shift < sorting::BITS_PER_PASS; bit_shift += 2u) {
            let key_index = (local_key >> config.shift) & 0xfu;
            let bit_key = (key_index >> bit_shift) & 3u;
            var packed_histogram = 1u << (bit_key * 8u);
            // workgroup prefix sum
            var sum = packed_histogram;
            lds_scratch[local_id.x] = sum;
            for (var i = 0u; i < 8u; i++) {
                workgroupBarrier();
                if local_id.x >= (1u << i) {
                    sum += lds_scratch[local_id.x - (1u << i)];
                }
                workgroupBarrier();
                lds_scratch[local_id.x] = sum;
            }
            workgroupBarrier();
            packed_histogram = lds_scratch[sorting::WG - 1u];
            packed_histogram = (packed_histogram << 8u) + (packed_histogram << 16u) + (packed_histogram << 24u);
            var local_sum = packed_histogram;
            if local_id.x > 0u {
                local_sum += lds_scratch[local_id.x - 1u];
            }
            let key_offset = (local_sum >> (bit_key * 8u)) & 0xffu;
            
            lds_sums[key_offset] = local_key;
            workgroupBarrier();
            local_key = lds_sums[local_id.x];
            workgroupBarrier();
        
            lds_sums[key_offset] = local_value;
            workgroupBarrier();
            local_value = lds_sums[local_id.x];
            workgroupBarrier();
        }
        let key_index = (local_key >> config.shift) & 0xfu;
        atomicAdd(&local_histogram[key_index], 1u);
        workgroupBarrier();
        var histogram_local_sum = 0u;
        if local_id.x < sorting::BIN_COUNT {
            histogram_local_sum = local_histogram[local_id.x];
        }
        // workgroup prefix sum of histogram
        var histogram_prefix_sum = histogram_local_sum;
        if local_id.x < sorting::BIN_COUNT {
            lds_scratch[local_id.x] = histogram_prefix_sum;
        }
        for (var i = 0u; i < 4u; i++) {
            workgroupBarrier();
            if local_id.x >= (1u << i) && local_id.x < sorting::BIN_COUNT {
                histogram_prefix_sum += lds_scratch[local_id.x - (1u << i)];
            }
            workgroupBarrier();
            if local_id.x < sorting::BIN_COUNT {
                lds_scratch[local_id.x] = histogram_prefix_sum;
            }
        }
        let global_offset = bin_offset_cache[key_index];
        workgroupBarrier();
        var local_offset = local_id.x;
        if key_index > 0u {
            local_offset -= lds_scratch[key_index - 1u];
        }
        let total_offset = global_offset + local_offset;
        if total_offset < num_keys {
            out[total_offset] = local_key;
            out_values[total_offset] = local_value;
        }
        if local_id.x < sorting::BIN_COUNT {
            bin_offset_cache[local_id.x] += local_histogram[local_id.x];
        }
        workgroupBarrier();
        data_index += sorting::WG;
    }
}
