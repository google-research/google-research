@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

const THREADS_PER_GROUP: u32 = 512u;

var<workgroup> bucket: array<u32, THREADS_PER_GROUP>;

fn groupScan(id: u32, gi: u32, x: u32) {
    bucket[gi] = x;
    for (var t = 1u; t < THREADS_PER_GROUP; t = t * 2u) {
        workgroupBarrier();
        var temp = bucket[gi];
        if (gi >= t) {
            temp += bucket[gi - t];
        }
        workgroupBarrier();
        bucket[gi] = temp;
    }
    if id < arrayLength(&output) {
        output[id] = bucket[gi];
    }
}
 