// rq-74bb02cc
//
// Bundled handle to every open builder registry the runner consults:
// integrators, thermostats, barostats, constraint types, and
// potentials. Lives at the crate root so it does not appear to belong
// to any single subsystem.
//
// `run_simulation_with_registries` reads every field through this
// bundle, and `Config::validate_against` consumes the integrator /
// thermostat / barostat / constraint-type registries from it. Callers
// that want to register custom builders construct a `Registries`
// (either from `with_builtins()` and then call `register_*`, or from
// `new()` and register every builder explicitly) and pass it to
// `run_simulation_with_registries`.

use crate::analysis::{AnalysisBuilder, AnalysisRegistry};
use crate::forces::{PotentialBuilder, PotentialRegistry};
use crate::integrator::{
    BarostatBuilder, BarostatRegistry, ConstraintBuilder, ConstraintRegistry, IntegratorBuilder,
    IntegratorRegistry, ThermostatBuilder, ThermostatRegistry,
};
use crate::minimizer::{MinimizerBuilder, MinimizerRegistry};

// rq-32308250 rq-a7211dfd
#[derive(Debug)]
pub struct Registries {
    pub integrators: IntegratorRegistry,
    pub thermostats: ThermostatRegistry,
    pub barostats: BarostatRegistry,
    pub constraint_types: ConstraintRegistry,
    pub potentials: PotentialRegistry,
    pub minimizers: MinimizerRegistry,
    pub analyses: AnalysisRegistry,
}

impl Registries {
    pub fn with_builtins() -> Self {
        Registries {
            integrators: IntegratorRegistry::with_builtins(),
            thermostats: ThermostatRegistry::with_builtins(),
            barostats: BarostatRegistry::with_builtins(),
            constraint_types: ConstraintRegistry::with_builtins(),
            potentials: PotentialRegistry::with_builtins(),
            minimizers: MinimizerRegistry::with_builtins(),
            analyses: AnalysisRegistry::with_builtins(),
        }
    }

    pub fn new() -> Self {
        Registries {
            integrators: IntegratorRegistry::new(),
            thermostats: ThermostatRegistry::new(),
            barostats: BarostatRegistry::new(),
            constraint_types: ConstraintRegistry::new(),
            potentials: PotentialRegistry::new(),
            minimizers: MinimizerRegistry::new(),
            analyses: AnalysisRegistry::new(),
        }
    }

    pub fn register_integrator(&mut self, builder: Box<dyn IntegratorBuilder>) {
        self.integrators.register(builder);
    }

    pub fn register_thermostat(&mut self, builder: Box<dyn ThermostatBuilder>) {
        self.thermostats.register(builder);
    }

    pub fn register_barostat(&mut self, builder: Box<dyn BarostatBuilder>) {
        self.barostats.register(builder);
    }

    pub fn register_constraint_type(&mut self, builder: Box<dyn ConstraintBuilder>) {
        self.constraint_types.register(builder);
    }

    pub fn register_potential(&mut self, builder: Box<dyn PotentialBuilder>) {
        self.potentials.register(builder);
    }

    pub fn register_minimizer(&mut self, builder: Box<dyn MinimizerBuilder>) {
        self.minimizers.register(builder);
    }

    pub fn register_analysis(&mut self, builder: Box<dyn AnalysisBuilder>) {
        self.analyses.register(builder);
    }
}

impl Default for Registries {
    fn default() -> Self {
        Registries::with_builtins()
    }
}
