#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3u, 
    @builtin(local_invocation_index) gid: u32,
) {
    let idx = id.x * helpers::THREADS_PER_GROUP - 1u;
    
    var x = 0u;
    if (idx >= 0u && idx < arrayLength(&helpers::input)) {
        x = helpers::input[idx];
    }
 
    helpers::groupScan(id.x, gid, x);
}
