// rq-e0a0553d rq-6cd635cd rq-6c5b4246
//
// Three orthogonal slot frameworks: integrator, thermostat, barostat.
// The runner chains the slots `apply_pre → step → apply_post → apply`
// per timestep (see `simulation-runner.md` and `framework.md`).

use crate::forces::{ForceField, ForceFieldError};
use crate::gpu::{GpuContext, GpuError, ParticleBuffers};
use crate::io::config::{BarostatKind, IntegratorKind, ThermostatKind};
use crate::pbc::SimulationBox;
use crate::timings::{Timings, TimingsError};

pub mod andersen;
pub mod berendsen;
pub mod berendsen_barostat;
pub mod c_rescale_barostat;
pub mod csvr;
pub mod langevin_baoab;
pub mod mtk_npt;
pub mod nose_hoover_chain;
pub mod philox;
pub mod velocity_verlet;

pub use andersen::{AndersenBuilder, AndersenThermostat};
pub use berendsen::{BerendsenBuilder, BerendsenThermostat};
pub use berendsen_barostat::{BerendsenBarostat, BerendsenBarostatBuilder};
pub use c_rescale_barostat::{CRescaleBarostat, CRescaleBarostatBuilder};
pub use csvr::{CsvrBuilder, CsvrThermostat};
pub use langevin_baoab::{LangevinBaoabBuilder, LangevinBaoabState};
pub use mtk_npt::{MtkNptBuilder, MtkNptIntegrator};
pub use nose_hoover_chain::{
    NoseHooverChainBuilder, NoseHooverChainThermostat, nhc_chain_sub_step,
};
pub use philox::{philox_4x32_10, philox_normal};
pub use velocity_verlet::{VelocityVerletBuilder, VelocityVerletState};

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum IntegratorError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("{0}")]
    ForceField(#[from] ForceFieldError),
    #[error("unknown integrator kind `{0}`")]
    UnknownKind(String),
}

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum ThermostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown thermostat kind `{0}`")]
    UnknownKind(String),
}

// rq-2ccf40de
#[derive(Debug, thiserror::Error)]
pub enum BarostatError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("{0}")]
    Timings(#[from] TimingsError),
    #[error("unknown barostat kind `{0}`")]
    UnknownKind(String),
}

// --- Integrator trait, builder, registry ------------------------------

// rq-78f484d9
pub trait Integrator: std::fmt::Debug + Send {
    // rq-aa68f468
    fn step(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        force_field: &mut ForceField,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), IntegratorError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait IntegratorBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &IntegratorKind,
    ) -> Result<Box<dyn Integrator>, IntegratorError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct IntegratorRegistry {
    pub builders: Vec<Box<dyn IntegratorBuilder>>,
}

impl IntegratorRegistry {
    pub fn new() -> Self {
        IntegratorRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        IntegratorRegistry {
            builders: vec![
                Box::new(VelocityVerletBuilder),
                Box::new(LangevinBaoabBuilder),
                Box::new(MtkNptBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.builders.push(builder);
    }

    // rq-24f6b8b9
    pub fn build(
        &self,
        kind: &IntegratorKind,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Box<dyn Integrator>, IntegratorError> {
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return b.build(gpu, particle_count, kind);
            }
        }
        Err(IntegratorError::UnknownKind(target.to_string()))
    }
}

impl Default for IntegratorRegistry {
    fn default() -> Self {
        IntegratorRegistry::with_builtins()
    }
}

// --- Thermostat trait, builder, registry ------------------------------

// rq-5d9ed248
pub trait Thermostat: std::fmt::Debug + Send {
    // rq-2fe47a86
    fn apply_pre(
        &mut self,
        _buffers: &mut ParticleBuffers,
        _dt: f32,
        _timings: &mut Timings,
    ) -> Result<(), ThermostatError> {
        Ok(())
    }

    // rq-7a124d43
    fn apply_post(
        &mut self,
        buffers: &mut ParticleBuffers,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), ThermostatError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait ThermostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &ThermostatKind,
    ) -> Result<Box<dyn Thermostat>, ThermostatError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct ThermostatRegistry {
    pub builders: Vec<Box<dyn ThermostatBuilder>>,
}

impl ThermostatRegistry {
    pub fn new() -> Self {
        ThermostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        ThermostatRegistry {
            builders: vec![
                Box::new(NoseHooverChainBuilder),
                Box::new(CsvrBuilder),
                Box::new(AndersenBuilder),
                Box::new(BerendsenBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn ThermostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-678c233d
    pub fn build_optional(
        &self,
        kind: Option<&ThermostatKind>,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Option<Box<dyn Thermostat>>, ThermostatError> {
        let Some(kind) = kind else { return Ok(None) };
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return Ok(Some(b.build(gpu, particle_count, kind)?));
            }
        }
        Err(ThermostatError::UnknownKind(target.to_string()))
    }
}

impl Default for ThermostatRegistry {
    fn default() -> Self {
        ThermostatRegistry::with_builtins()
    }
}

// --- Barostat trait, builder, registry --------------------------------

// rq-076617ab
pub trait Barostat: std::fmt::Debug + Send {
    // rq-1179e42f
    fn apply(
        &mut self,
        buffers: &mut ParticleBuffers,
        sim_box: &mut SimulationBox,
        dt: f32,
        timings: &mut Timings,
    ) -> Result<(), BarostatError>;

    fn log_column_names(&self) -> &'static [&'static str] {
        &[]
    }

    fn log_column_values(
        &self,
        _kinetic_energy: f64,
        _potential_energy: f64,
    ) -> Vec<f64> {
        Vec::new()
    }
}

// rq-29e08cb5
pub trait BarostatBuilder: std::fmt::Debug + Send + Sync {
    fn kind_name(&self) -> &'static str;
    fn build(
        &self,
        gpu: &GpuContext,
        particle_count: usize,
        kind: &BarostatKind,
    ) -> Result<Box<dyn Barostat>, BarostatError>;
}

// rq-4901507f
#[derive(Debug)]
pub struct BarostatRegistry {
    pub builders: Vec<Box<dyn BarostatBuilder>>,
}

impl BarostatRegistry {
    pub fn new() -> Self {
        BarostatRegistry { builders: Vec::new() }
    }

    // rq-4901507f
    pub fn with_builtins() -> Self {
        BarostatRegistry {
            builders: vec![
                Box::new(BerendsenBarostatBuilder),
                Box::new(CRescaleBarostatBuilder),
            ],
        }
    }

    pub fn register(&mut self, builder: Box<dyn BarostatBuilder>) {
        self.builders.push(builder);
    }

    // rq-9548bc1a
    pub fn build_optional(
        &self,
        kind: Option<&BarostatKind>,
        gpu: &GpuContext,
        particle_count: usize,
    ) -> Result<Option<Box<dyn Barostat>>, BarostatError> {
        let Some(kind) = kind else { return Ok(None) };
        let target = kind.name();
        for b in &self.builders {
            if b.kind_name() == target {
                return Ok(Some(b.build(gpu, particle_count, kind)?));
            }
        }
        Err(BarostatError::UnknownKind(target.to_string()))
    }
}

impl Default for BarostatRegistry {
    fn default() -> Self {
        BarostatRegistry::with_builtins()
    }
}
