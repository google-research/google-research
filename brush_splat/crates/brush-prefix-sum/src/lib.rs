mod shaders;

use brush_kernel::calc_cube_count;
use brush_kernel::create_tensor;
use brush_kernel::kernel_source_gen;
use burn_wgpu::WgpuRuntime;
use shaders::prefix_sum_add_scanned_sums;
use shaders::prefix_sum_scan;
use shaders::prefix_sum_scan_sums;

kernel_source_gen!(PrefixSumScan {}, prefix_sum_scan);
kernel_source_gen!(PrefixSumScanSums {}, prefix_sum_scan_sums);
kernel_source_gen!(PrefixSumAddScannedSums {}, prefix_sum_add_scanned_sums);

use burn_wgpu::JitTensor;

pub fn prefix_sum(input: JitTensor<WgpuRuntime, u32>) -> JitTensor<WgpuRuntime, u32> {
    let threads_per_group = shaders::prefix_sum_helpers::THREADS_PER_GROUP as usize;
    let num = input.shape.dims[0];
    let client = &input.client;
    let outputs = create_tensor(input.shape.dims::<1>(), &input.device, client);

    unsafe {
        client.execute_unchecked(
            PrefixSumScan::task(),
            calc_cube_count([num as u32], PrefixSumScan::WORKGROUP_SIZE),
            vec![input.handle.binding(), outputs.handle.clone().binding()],
        );
    }

    if num <= threads_per_group {
        return outputs;
    }

    let mut group_buffer = vec![];
    let mut work_size = vec![];
    let mut work_sz = num;
    while work_sz > threads_per_group {
        work_sz = work_sz.div_ceil(threads_per_group);
        group_buffer.push(create_tensor::<u32, 1, WgpuRuntime>(
            [work_sz],
            &input.device,
            client,
        ));
        work_size.push(work_sz);
    }

    unsafe {
        client.execute_unchecked(
            PrefixSumScanSums::task(),
            calc_cube_count([work_size[0] as u32], PrefixSumScanSums::WORKGROUP_SIZE),
            vec![
                outputs.handle.clone().binding(),
                group_buffer[0].handle.clone().binding(),
            ],
        );
    }

    for l in 0..(group_buffer.len() - 1) {
        unsafe {
            client.execute_unchecked(
                PrefixSumScanSums::task(),
                calc_cube_count([work_size[l + 1] as u32], PrefixSumScanSums::WORKGROUP_SIZE),
                vec![
                    group_buffer[l].handle.clone().binding(),
                    group_buffer[l + 1].handle.clone().binding(),
                ],
            );
        }
    }

    for l in (1..group_buffer.len()).rev() {
        let work_sz = work_size[l - 1];

        unsafe {
            client.execute_unchecked(
                PrefixSumAddScannedSums::task(),
                calc_cube_count([work_sz as u32], PrefixSumAddScannedSums::WORKGROUP_SIZE),
                vec![
                    group_buffer[l].handle.clone().binding(),
                    group_buffer[l - 1].handle.clone().binding(),
                ],
            );
        }
    }

    unsafe {
        client.execute_unchecked(
            PrefixSumAddScannedSums::task(),
            calc_cube_count(
                [(work_size[0] * threads_per_group) as u32],
                PrefixSumAddScannedSums::WORKGROUP_SIZE,
            ),
            vec![
                group_buffer[0].handle.clone().binding(),
                outputs.handle.clone().binding(),
            ],
        );
    }

    outputs
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use crate::prefix_sum;
    use brush_kernel::bitcast_tensor;
    use burn::tensor::{Int, Tensor};
    use burn_wgpu::{JitBackend, WgpuRuntime};

    #[test]
    fn test_sum_tiny() {
        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data([1, 1, 1, 1], &device).into_primitive();
        let keys = bitcast_tensor(keys);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();
        let summed = summed.as_slice::<i32>().unwrap();
        assert_eq!(summed.len(), 4);
        assert_eq!(summed, [1, 2, 3, 4])
    }

    #[test]
    fn test_512_multiple() {
        const ITERS: usize = 1024;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(90 + i as i32);
        }
        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let keys = bitcast_tensor(keys);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();
        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();
        for (summed, reff) in summed.as_slice::<i32>().unwrap().iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff)
        }
    }

    #[test]
    fn test_sum() {
        const ITERS: usize = 512 * 16 + 123;
        let mut data = vec![];
        for i in 0..ITERS {
            data.push(2 + i as i32);
            data.push(0);
            data.push(32);
            data.push(512);
            data.push(30965);
        }

        type Backend = JitBackend<WgpuRuntime, f32, i32>;
        let device = Default::default();
        let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
        let keys = bitcast_tensor(keys);
        let summed = prefix_sum(keys.clone());
        let summed = Tensor::<Backend, 1, Int>::from_primitive(bitcast_tensor(summed)).to_data();

        let prefix_sum_ref: Vec<_> = data
            .into_iter()
            .scan(0, |x, y| {
                *x += y;
                Some(*x)
            })
            .collect();

        for (summed, reff) in summed.as_slice::<i32>().unwrap().iter().zip(prefix_sum_ref) {
            assert_eq!(*summed, reff)
        }
    }
}
