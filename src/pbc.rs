// rq-03830444
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice};

use crate::gpu::GpuError;
use crate::precision::Real;

// rq-b75afb31
#[derive(Debug)]
pub struct SimulationBox {
    lx: Real,
    ly: Real,
    lz: Real,
    xy: Real,
    xz: Real,
    yz: Real,
    generation: u64,
    device: Arc<CudaDevice>,
    lattice_device: CudaSlice<Real>,
}

impl Clone for SimulationBox {
    fn clone(&self) -> Self {
        // Allocate a fresh device buffer and copy our values into it so
        // mutations through one handle never reach the other.
        let host = [self.lx, self.ly, self.lz, self.xy, self.xz, self.yz];
        let lattice_device = self
            .device
            .htod_sync_copy(&host)
            .expect("SimulationBox clone htod failed");
        SimulationBox {
            lx: self.lx,
            ly: self.ly,
            lz: self.lz,
            xy: self.xy,
            xz: self.xz,
            yz: self.yz,
            generation: self.generation,
            device: self.device.clone(),
            lattice_device,
        }
    }
}

impl PartialEq for SimulationBox {
    fn eq(&self, other: &Self) -> bool {
        self.lx == other.lx
            && self.ly == other.ly
            && self.lz == other.lz
            && self.xy == other.xy
            && self.xz == other.xz
            && self.yz == other.yz
            && self.generation == other.generation
    }
}

// rq-aef9888b
#[derive(Debug, thiserror::Error)]
pub enum SimulationBoxError {
    #[error("non-finite simulation-box lattice value for `{name}`: {value}")]
    NonFiniteLatticeValue { name: &'static str, value: Real },
    #[error("non-positive simulation-box diagonal for `{name}`: {value}")]
    NonPositiveDiagonal { name: &'static str, value: Real },
    #[error("simulation-box perpendicular width along lattice direction `{direction}` is {width}, below the required {required}")]
    PerpendicularWidthTooSmall {
        direction: &'static str,
        width: Real,
        required: Real,
    },
    #[error("{0}")]
    Gpu(#[from] GpuError),
}

fn check_finite(name: &'static str, value: Real) -> Result<(), SimulationBoxError> {
    if !value.is_finite() {
        return Err(SimulationBoxError::NonFiniteLatticeValue { name, value });
    }
    Ok(())
}

fn check_diagonal(name: &'static str, value: Real) -> Result<(), SimulationBoxError> {
    check_finite(name, value)?;
    if value <= 0.0 {
        return Err(SimulationBoxError::NonPositiveDiagonal { name, value });
    }
    Ok(())
}

fn check_tilt(name: &'static str, value: Real) -> Result<(), SimulationBoxError> {
    check_finite(name, value)
}

fn validate_lattice(
    lx: Real,
    ly: Real,
    lz: Real,
    xy: Real,
    xz: Real,
    yz: Real,
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
        device: &Arc<CudaDevice>,
        lx: Real,
        ly: Real,
        lz: Real,
        xy: Real,
        xz: Real,
        yz: Real,
    ) -> Result<Self, SimulationBoxError> {
        validate_lattice(lx, ly, lz, xy, xz, yz)?;
        let host = [lx, ly, lz, xy, xz, yz];
        let lattice_device = device.htod_sync_copy(&host).map_err(GpuError::from)?;
        Ok(SimulationBox {
            lx,
            ly,
            lz,
            xy,
            xz,
            yz,
            generation: 0,
            device: device.clone(),
            lattice_device,
        })
    }

    // rq-71fbbafb
    pub fn set_lattice(
        &mut self,
        lx: Real,
        ly: Real,
        lz: Real,
        xy: Real,
        xz: Real,
        yz: Real,
    ) -> Result<(), SimulationBoxError> {
        validate_lattice(lx, ly, lz, xy, xz, yz)?;
        let host = [lx, ly, lz, xy, xz, yz];
        self.device
            .htod_sync_copy_into(&host, &mut self.lattice_device)
            .map_err(GpuError::from)?;
        self.lx = lx;
        self.ly = ly;
        self.lz = lz;
        self.xy = xy;
        self.xz = xz;
        self.yz = yz;
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// Read-only access to the device-resident lattice mirror.
    /// Length 6, in `[lx, ly, lz, xy, xz, yz]` order. Kernel launchers
    /// pass this to CUDA kernels as a `const Real *lattice` argument.
    pub fn lattice_device(&self) -> &CudaSlice<Real> {
        &self.lattice_device
    }

    /// Mutable access to the device-resident lattice mirror. Bumps the
    /// generation counter; host fields become stale until the next
    /// `flush_from_device` call. Used by barostat kernels that compute
    /// the new lattice on device.
    pub fn lattice_device_mut(&mut self) -> &mut CudaSlice<Real> {
        self.generation = self.generation.wrapping_add(1);
        &mut self.lattice_device
    }

    /// Multiplies every component of the device-resident lattice mirror
    /// by `factor` via the `multiply_lattice_isotropic` kernel. Bumps
    /// the generation counter on success. Host fields are not updated.
    ///
    /// Returns `NonFiniteLatticeValue { name: "factor" }` if `factor`
    /// is non-finite or `NonPositiveDiagonal { name: "factor" }` if
    /// `factor <= 0`. On any error the device buffer and generation
    /// counter are left unchanged.
    pub fn multiply_lattice_isotropic(
        &mut self,
        factor: Real,
    ) -> Result<(), SimulationBoxError> {
        check_diagonal("factor", factor)?;
        let func = self
            .device
            .get_func("barostat", "multiply_lattice_isotropic")
            .ok_or_else(|| {
                GpuError(cudarc::driver::DriverError(
                    cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND,
                ))
            })?;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        use cudarc::driver::LaunchAsync;
        unsafe {
            func.launch(cfg, (&mut self.lattice_device, factor))
                .map_err(GpuError::from)?;
        }
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// Downloads the device-resident lattice into the host fields. The
    /// generation counter is left unchanged. After a successful return,
    /// every host accessor reflects the latest device state.
    pub fn flush_from_device(&mut self) -> Result<(), SimulationBoxError> {
        let mut host = [0.0 as Real; 6];
        self.device
            .dtoh_sync_copy_into(&self.lattice_device, &mut host)
            .map_err(GpuError::from)?;
        self.lx = host[0];
        self.ly = host[1];
        self.lz = host[2];
        self.xy = host[3];
        self.xz = host[4];
        self.yz = host[5];
        Ok(())
    }

    /// The `Arc<CudaDevice>` the box was constructed against. Kernel
    /// launchers needing a device handle (for buffer allocation or
    /// stream management) read it from here when no other handle is in
    /// scope.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    // Multiply all six lattice parameters by `factor` in a single
    // mutation that bumps the generation counter exactly once. Preserves
    // the triclinic shape (every angle ratio is invariant). Convenience
    // over `set_lattice(factor*lx, ...)`; provided so callers cannot
    // accidentally apply different scale factors to the orthogonal and
    // shear components.
    // rq-9e2e9d4e
    pub fn rescale_isotropic(&mut self, factor: Real) -> Result<(), SimulationBoxError> {
        self.set_lattice(
            self.lx * factor,
            self.ly * factor,
            self.lz * factor,
            self.xy * factor,
            self.xz * factor,
            self.yz * factor,
        )
    }

    // rq-dc17132d
    pub fn generation(&self) -> u64 {
        self.generation
    }

    // rq-e8be1a1c
    pub fn lattice(&self) -> [Real; 6] {
        [self.lx, self.ly, self.lz, self.xy, self.xz, self.yz]
    }

    // rq-f73a0f99
    pub fn lx(&self) -> Real {
        self.lx
    }

    // rq-f73a0f99
    pub fn ly(&self) -> Real {
        self.ly
    }

    // rq-f73a0f99
    pub fn lz(&self) -> Real {
        self.lz
    }

    // rq-f73a0f99
    pub fn xy(&self) -> Real {
        self.xy
    }

    // rq-f73a0f99
    pub fn xz(&self) -> Real {
        self.xz
    }

    // rq-f73a0f99
    pub fn yz(&self) -> Real {
        self.yz
    }

    // rq-3b9ed390
    pub fn volume(&self) -> Real {
        self.lx * self.ly * self.lz
    }

    // rq-9d8d96f1
    //
    // Closed-form perpendicular widths along each lattice direction:
    //   w_a = (lx·ly·lz) / sqrt((ly·lz)² + (xy·lz)² + (xy·yz − ly·xz)²)
    //   w_b = (ly·lz)    / sqrt(lz² + yz²)
    //   w_c = lz
    pub fn perpendicular_widths(&self) -> [Real; 3] {
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
    pub fn min_perpendicular_width(&self) -> Real {
        let [w_a, w_b, w_c] = self.perpendicular_widths();
        w_a.min(w_b).min(w_c)
    }

    // rq-1a7bd47a
    //
    // Scans the three perpendicular widths in lattice-direction order
    // (a, b, c) and returns the first one whose width is strictly less
    // than `required`. `required` is taken verbatim — no sign or
    // finiteness pre-check is applied.
    pub fn check_min_perpendicular_width(
        &self,
        required: Real,
    ) -> Result<(), SimulationBoxError> {
        let widths = self.perpendicular_widths();
        let directions: [&'static str; 3] = ["a", "b", "c"];
        for d in 0..3 {
            if widths[d] < required {
                return Err(SimulationBoxError::PerpendicularWidthTooSmall {
                    direction: directions[d],
                    width: widths[d],
                    required,
                });
            }
        }
        Ok(())
    }

    // rq-d49c9093
    pub fn minimum_image(&self, displacement: [Real; 3]) -> [Real; 3] {
        let (wrapped, _image) = self.wrap_with_image_count(displacement);
        wrapped
    }

    // rq-9b1c84c3
    pub fn wrap_position(&self, position: [Real; 3]) -> [Real; 3] {
        let (wrapped, _image) = self.wrap_with_image_count(position);
        wrapped
    }

    // rq-a4d5e711
    pub fn wrap_position_with_image_count(
        &self,
        position: [Real; 3],
    ) -> ([Real; 3], [i32; 3]) {
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
    fn wrap_with_image_count(&self, v: [Real; 3]) -> ([Real; 3], [i32; 3]) {
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
    pub fn fractional_coords(&self, position: [Real; 3]) -> [Real; 3] {
        let s_c = position[2] / self.lz;
        let s_b = (position[1] - s_c * self.yz) / self.ly;
        let s_a = (position[0] - s_b * self.xy - s_c * self.xz) / self.lx;
        [s_a, s_b, s_c]
    }

    // rq-be7b9fe6
    pub fn cartesian_coords(&self, fractional: [Real; 3]) -> [Real; 3] {
        let s_a = fractional[0];
        let s_b = fractional[1];
        let s_c = fractional[2];
        let v_z = s_c * self.lz;
        let v_y = s_b * self.ly + s_c * self.yz;
        let v_x = s_a * self.lx + s_b * self.xy + s_c * self.xz;
        [v_x, v_y, v_z]
    }
}
