// rq-03830444
#[derive(Debug, Clone, Copy, PartialEq)]
// rq-b75afb31
pub struct SimulationBox {
    lx: f32,
    ly: f32,
    lz: f32,
    xy: f32,
    xz: f32,
    yz: f32,
    generation: u64,
}

// rq-aef9888b
#[derive(Debug, thiserror::Error)]
pub enum SimulationBoxError {
    #[error("non-finite simulation-box lattice value for `{name}`: {value}")]
    NonFiniteLatticeValue { name: &'static str, value: f32 },
    #[error("non-positive simulation-box diagonal for `{name}`: {value}")]
    NonPositiveDiagonal { name: &'static str, value: f32 },
}

fn check_finite(name: &'static str, value: f32) -> Result<(), SimulationBoxError> {
    if !value.is_finite() {
        return Err(SimulationBoxError::NonFiniteLatticeValue { name, value });
    }
    Ok(())
}

fn check_diagonal(name: &'static str, value: f32) -> Result<(), SimulationBoxError> {
    check_finite(name, value)?;
    if value <= 0.0 {
        return Err(SimulationBoxError::NonPositiveDiagonal { name, value });
    }
    Ok(())
}

fn check_tilt(name: &'static str, value: f32) -> Result<(), SimulationBoxError> {
    check_finite(name, value)
}

fn validate_lattice(
    lx: f32,
    ly: f32,
    lz: f32,
    xy: f32,
    xz: f32,
    yz: f32,
) -> Result<(), SimulationBoxError> {
    check_diagonal("lx", lx)?;
    check_diagonal("ly", ly)?;
    check_diagonal("lz", lz)?;
    check_tilt("xy", xy)?;
    check_tilt("xz", xz)?;
    check_tilt("yz", yz)?;
    Ok(())
}

impl SimulationBox {
    // rq-f0da71ea
    pub fn new(
        lx: f32,
        ly: f32,
        lz: f32,
        xy: f32,
        xz: f32,
        yz: f32,
    ) -> Result<Self, SimulationBoxError> {
        validate_lattice(lx, ly, lz, xy, xz, yz)?;
        Ok(SimulationBox {
            lx,
            ly,
            lz,
            xy,
            xz,
            yz,
            generation: 0,
        })
    }

    // rq-71fbbafb
    pub fn set_lattice(
        &mut self,
        lx: f32,
        ly: f32,
        lz: f32,
        xy: f32,
        xz: f32,
        yz: f32,
    ) -> Result<(), SimulationBoxError> {
        validate_lattice(lx, ly, lz, xy, xz, yz)?;
        self.lx = lx;
        self.ly = ly;
        self.lz = lz;
        self.xy = xy;
        self.xz = xz;
        self.yz = yz;
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    // rq-dc17132d
    pub fn generation(&self) -> u64 {
        self.generation
    }

    // rq-e8be1a1c
    pub fn lattice(&self) -> [f32; 6] {
        [self.lx, self.ly, self.lz, self.xy, self.xz, self.yz]
    }

    // rq-f73a0f99
    pub fn lx(&self) -> f32 {
        self.lx
    }

    // rq-f73a0f99
    pub fn ly(&self) -> f32 {
        self.ly
    }

    // rq-f73a0f99
    pub fn lz(&self) -> f32 {
        self.lz
    }

    // rq-f73a0f99
    pub fn xy(&self) -> f32 {
        self.xy
    }

    // rq-f73a0f99
    pub fn xz(&self) -> f32 {
        self.xz
    }

    // rq-f73a0f99
    pub fn yz(&self) -> f32 {
        self.yz
    }

    // rq-3b9ed390
    pub fn volume(&self) -> f32 {
        self.lx * self.ly * self.lz
    }

    // rq-9d8d96f1
    //
    // Closed-form perpendicular widths along each lattice direction:
    //   w_a = (lx·ly·lz) / sqrt((ly·lz)² + (xy·lz)² + (xy·yz − ly·xz)²)
    //   w_b = (ly·lz)    / sqrt(lz² + yz²)
    //   w_c = lz
    pub fn perpendicular_widths(&self) -> [f32; 3] {
        let lx = self.lx;
        let ly = self.ly;
        let lz = self.lz;
        let xy = self.xy;
        let xz = self.xz;
        let yz = self.yz;
        let vol = lx * ly * lz;
        let ly_lz = ly * lz;
        let xy_lz = xy * lz;
        let xy_yz_minus_ly_xz = xy * yz - ly * xz;
        let denom_a = (ly_lz * ly_lz + xy_lz * xy_lz + xy_yz_minus_ly_xz * xy_yz_minus_ly_xz).sqrt();
        let w_a = vol / denom_a;
        let denom_b = (lz * lz + yz * yz).sqrt();
        let w_b = ly_lz / denom_b;
        let w_c = lz;
        [w_a, w_b, w_c]
    }

    // rq-5fe22acb
    pub fn min_perpendicular_width(&self) -> f32 {
        let [w_a, w_b, w_c] = self.perpendicular_widths();
        w_a.min(w_b).min(w_c)
    }

    // rq-d49c9093
    pub fn minimum_image(&self, displacement: [f32; 3]) -> [f32; 3] {
        let (wrapped, _image) = self.wrap_with_image_count(displacement);
        wrapped
    }

    // rq-9b1c84c3
    pub fn wrap_position(&self, position: [f32; 3]) -> [f32; 3] {
        let (wrapped, _image) = self.wrap_with_image_count(position);
        wrapped
    }

    // rq-a4d5e711
    pub fn wrap_position_with_image_count(
        &self,
        position: [f32; 3],
    ) -> ([f32; 3], [i32; 3]) {
        self.wrap_with_image_count(position)
    }

    // rq-4ca9b179
    //
    // Fractional-coordinate wrap. Compute the fractional coordinates of
    // `v` via back-substitution (z-then-y-then-x), pick the integer
    // image triple that brings each component into `[-1/2, 1/2)`, and
    // apply the image-vector correction directly in Cartesian
    // coordinates. The output has fractional coordinates in
    // `[-1/2, 1/2)³` and therefore lies inside the primary
    // parallelepiped.
    //
    // For an orthorhombic box (xy = xz = yz = 0), s_d reduces to
    // v_d / L_d, and `floor(s_d + 0.5) = floor((v_d + L_d * 0.5) / L_d)`
    // — the algorithm collapses to three independent per-axis wraps
    // that match the v0 orthorhombic implementation bit-for-bit.
    #[inline]
    fn wrap_with_image_count(&self, v: [f32; 3]) -> ([f32; 3], [i32; 3]) {
        let s_c = v[2] / self.lz;
        let s_b = (v[1] - s_c * self.yz) / self.ly;
        let s_a = (v[0] - s_b * self.xy - s_c * self.xz) / self.lx;

        let k_a_f = (s_a + 0.5).floor();
        let k_b_f = (s_b + 0.5).floor();
        let k_c_f = (s_c + 0.5).floor();

        let vx = v[0] - k_a_f * self.lx - k_b_f * self.xy - k_c_f * self.xz;
        let vy = v[1] - k_b_f * self.ly - k_c_f * self.yz;
        let vz = v[2] - k_c_f * self.lz;

        ([vx, vy, vz], [k_a_f as i32, k_b_f as i32, k_c_f as i32])
    }

    // rq-1a3ec0c8
    pub fn fractional_coords(&self, position: [f32; 3]) -> [f32; 3] {
        let s_c = position[2] / self.lz;
        let s_b = (position[1] - s_c * self.yz) / self.ly;
        let s_a = (position[0] - s_b * self.xy - s_c * self.xz) / self.lx;
        [s_a, s_b, s_c]
    }

    // rq-be7b9fe6
    pub fn cartesian_coords(&self, fractional: [f32; 3]) -> [f32; 3] {
        let s_a = fractional[0];
        let s_b = fractional[1];
        let s_c = fractional[2];
        let v_z = s_c * self.lz;
        let v_y = s_b * self.ly + s_c * self.yz;
        let v_x = s_a * self.lx + s_b * self.xy + s_c * self.xz;
        [v_x, v_y, v_z]
    }
}
