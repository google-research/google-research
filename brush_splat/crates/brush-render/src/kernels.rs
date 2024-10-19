use super::shaders::{
    get_tile_bin_edges, map_gaussian_to_intersects, project_backwards, project_forward,
    project_visible, rasterize, rasterize_backwards,
};
use crate::shaders::gather_grads;
use brush_kernel::kernel_source_gen;

kernel_source_gen!(ProjectSplats {}, project_forward);
kernel_source_gen!(ProjectVisible {}, project_visible);
kernel_source_gen!(MapGaussiansToIntersect {}, map_gaussian_to_intersects);
kernel_source_gen!(GetTileBinEdges {}, get_tile_bin_edges);
kernel_source_gen!(Rasterize { raster_u32 }, rasterize);
kernel_source_gen!(RasterizeBackwards { hard_float }, rasterize_backwards);
kernel_source_gen!(GatherGrads {}, gather_grads);
kernel_source_gen!(ProjectBackwards {}, project_backwards);
