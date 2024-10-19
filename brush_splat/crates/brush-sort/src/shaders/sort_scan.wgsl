#import sorting

@group(0) @binding(0) var<storage, read_write> num_keys_arr: array<u32>;
@group(0) @binding(1) var<storage, read_write> reduced: array<u32>;

var<workgroup> sums: array<u32, sorting::WG>;
var<workgroup> lds: array<array<u32, sorting::WG>, sorting::ELEMENTS_PER_THREAD>;

var<workgroup> num_keys_wg: u32;

@compute
@workgroup_size(sorting::WG, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    if local_id.x == 0u {
        num_keys_wg = num_keys_arr[0];
    }
    let num_keys = workgroupUniformLoad(&num_keys_wg);
    // let num_keys = num_keys_arr[0];
    let num_wgs = sorting::div_ceil(num_keys, sorting::BLOCK_SIZE);
    let num_reduce_wgs = sorting::BIN_COUNT * sorting::div_ceil(num_wgs, sorting::BLOCK_SIZE);

    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = i * sorting::WG + local_id.x;
        let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
        let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
        lds[row][col] = reduced[data_index];
    }
    workgroupBarrier();
    var sum = 0u;
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let tmp = lds[i][local_id.x];
        lds[i][local_id.x] = sum;
        sum += tmp;
    }
    // workgroup prefix sum
    sums[local_id.x] = sum;
    for (var i = 0u; i < 8u; i++) {
        workgroupBarrier();
        if local_id.x >= (1u << i) {
            sum += sums[local_id.x - (1u << i)];
        }
        workgroupBarrier();
        sums[local_id.x] = sum;
    }
    workgroupBarrier();
    sum = 0u;
    if local_id.x > 0u {
        sum = sums[local_id.x - 1u];
    }
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        lds[i][local_id.x] += sum;
    }
    // lds now contains exclusive prefix sum
    workgroupBarrier();
    for (var i = 0u; i < sorting::ELEMENTS_PER_THREAD; i++) {
        let data_index = i * sorting::WG + local_id.x;
        let col = (i * sorting::WG + local_id.x) / sorting::ELEMENTS_PER_THREAD;
        let row = (i * sorting::WG + local_id.x) % sorting::ELEMENTS_PER_THREAD;
        if data_index < num_reduce_wgs {
            reduced[data_index] = lds[row][col];
        }
    }
}
