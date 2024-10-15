#import prefix_sum_helpers as helpers

@compute
@workgroup_size(helpers::THREADS_PER_GROUP, 1, 1)
fn main(
    @builtin(global_invocation_id) id: vec3u, 
    @builtin(workgroup_id) gid: vec3u
) {
    if (id.x < arrayLength(&helpers::output)) {
        helpers::output[id.x] += helpers::input[gid.x];
    }
}
