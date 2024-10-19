#[derive(Debug, Default, Clone)]
pub struct Camera {
    pub fov: glam::Vec2,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
}

impl Camera {
    pub fn new(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov: glam::Vec2,
        center_uv: glam::Vec2,
    ) -> Self {
        Camera {
            fov,
            center_uv,
            position,
            rotation,
        }
    }

    pub fn focal(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fov.x, img_size.x),
            fov_to_focal(self.fov.y, img_size.y),
        )
    }

    pub fn center(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            self.center_uv.x * img_size.x as f32,
            self.center_uv.y * img_size.y as f32,
        )
    }

    pub fn local_to_world(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.rotation, self.position)
    }

    pub fn world_to_local(&self) -> glam::Mat4 {
        self.local_to_world().inverse()
    }
}

// Converts field of view to focal length
pub fn fov_to_focal(fov_rad: f32, pixels: u32) -> f32 {
    0.5 * (pixels as f32) / (fov_rad * 0.5).tan()
}

// Converts focal length to field of view.
pub fn focal_to_fov(focal: f32, pixels: u32) -> f32 {
    2.0 * ((pixels as f32) / (2.0 * focal)).atan()
}
