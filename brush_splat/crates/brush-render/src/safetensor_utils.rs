// TODO: This is rather gnarly, must be an easier way to do this.
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor, TensorData},
};
use safetensors::tensor::TensorView;

fn float_from_u8(data: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(data).to_vec()
}

pub(crate) fn safetensor_to_burn<B: Backend, const D: usize>(
    t: TensorView,
    device: &B::Device,
) -> Tensor<B, D, Float> {
    let data = TensorData::new::<f32, _>(float_from_u8(t.data()), t.shape());
    Tensor::from_data(data, device)
}
