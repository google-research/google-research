#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3u, 
    @builtin(local_invocation_index) gid: u32,
) {
    var x = 0u;
    if (id.x < arrayLength(&helpers::input)) {
        x = helpers::input[id.x];
    }
 
    helpers::groupScan(id.x, gid, x);
}

