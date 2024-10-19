use burn_jit::{JitElement, JitRuntime};
use burn_wgpu::JitTensor;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub(crate) enum DimBound {
    Exact(usize),
    Any,
    Matching(&'static str),
}

pub(crate) struct DimCheck<'a, R: JitRuntime> {
    bound: HashMap<&'a str, usize>,
    device: Option<R::Device>,
}

impl<'a, R: JitRuntime> DimCheck<'a, R> {
    pub fn new() -> Self {
        DimCheck {
            bound: HashMap::new(),
            device: None,
        }
    }

    pub fn check_dims<E: JitElement>(
        mut self,
        tensor: &JitTensor<R, E>,
        bounds: &[DimBound],
    ) -> Self {
        let dims = &tensor.shape.dims;

        match self.device.as_ref() {
            None => self.device = Some(tensor.device.clone()),
            Some(d) => assert_eq!(d, &tensor.device),
        }

        for (cur_dim, bound) in dims.into_iter().zip(bounds) {
            match bound {
                DimBound::Exact(dim) => assert_eq!(cur_dim, dim),
                DimBound::Any => (),
                DimBound::Matching(id) => {
                    let dim = self.bound.entry(id).or_insert(*cur_dim);
                    assert_eq!(cur_dim, dim);
                }
            }
        }
        self
    }
}

impl From<usize> for DimBound {
    fn from(value: usize) -> Self {
        DimBound::Exact(value)
    }
}

impl From<u32> for DimBound {
    fn from(value: u32) -> Self {
        DimBound::Exact(value as usize)
    }
}

impl From<i32> for DimBound {
    fn from(value: i32) -> Self {
        DimBound::Exact(value as usize)
    }
}

impl From<&'static str> for DimBound {
    fn from(value: &'static str) -> Self {
        match value {
            "*" => DimBound::Any,
            _ => DimBound::Matching(value),
        }
    }
}
