//! Unit-system selection for the input-file surface.
//!
//! The internal simulation is SI throughout; this module exists to let
//! the user write input files in either SI or Hartree atomic units. The
//! `units` top-level TOML field selects the system; every unit-bearing
//! value the parser reads is then multiplied by the appropriate
//! SI-per-atomic-unit factor on the way in.
//!
//! Conversion factors live in the generated `atomic_constants` submodule
//! and originate from QCElemental (CODATA-2018). Regenerate them with
//! `python3 scripts/gen_atomic_units.py`.

pub mod atomic_constants;

use atomic_constants::{
    CHARGE_E_TO_C, ENERGY_HARTREE_TO_J, FORCE_AU_TO_N, LENGTH_BOHR_TO_M, MASS_ME_TO_KG,
    PRESSURE_AU_TO_PA, TEMPERATURE_AU_TO_K, TIME_AU_TO_S, VELOCITY_AU_TO_M_PER_S,
};

/// The unit system the user's input file is written in. `Si` is the
/// default and a no-op for every conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnitSystem {
    #[default]
    Si,
    Atomic,
}

impl UnitSystem {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "si" => Some(UnitSystem::Si),
            "atomic" => Some(UnitSystem::Atomic),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            UnitSystem::Si => "si",
            UnitSystem::Atomic => "atomic",
        }
    }
}

/// The physical dimensions the parser knows how to rescale. Every
/// unit-bearing field in the config (or in the .xyz init state) maps to
/// exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    Length,
    InverseLength,
    Mass,
    Charge,
    Energy,
    Time,
    InverseTime,
    Force,
    Pressure,
    InversePressure,
    Temperature,
    Velocity,
}

impl UnitSystem {
    /// Multiplier that turns a value expressed in `self` into the
    /// equivalent SI value: `value_si = value * factor(dim)`.
    pub fn factor(self, dim: Dimension) -> f64 {
        if self == UnitSystem::Si {
            return 1.0;
        }
        match dim {
            Dimension::Length => LENGTH_BOHR_TO_M,
            Dimension::InverseLength => 1.0 / LENGTH_BOHR_TO_M,
            Dimension::Mass => MASS_ME_TO_KG,
            Dimension::Charge => CHARGE_E_TO_C,
            Dimension::Energy => ENERGY_HARTREE_TO_J,
            Dimension::Time => TIME_AU_TO_S,
            Dimension::InverseTime => 1.0 / TIME_AU_TO_S,
            Dimension::Force => FORCE_AU_TO_N,
            Dimension::Pressure => PRESSURE_AU_TO_PA,
            Dimension::InversePressure => 1.0 / PRESSURE_AU_TO_PA,
            Dimension::Temperature => TEMPERATURE_AU_TO_K,
            Dimension::Velocity => VELOCITY_AU_TO_M_PER_S,
        }
    }

    /// Convert a single scalar to SI.
    pub fn to_si(self, dim: Dimension, value: f64) -> f64 {
        value * self.factor(dim)
    }
}

/// Look up the unit-bearing fields of a slot kind. Returns `None` for
/// kinds this module doesn't know about — the slot will pass through
/// unchanged, and the builder registry will reject it later if the
/// kind is genuinely invalid.
///
/// Kinds that exist but carry no unit-bearing fields (e.g.
/// `velocity-verlet`) return `Some(&[])` so a maintainer adding a new
/// slot can spot at a glance that the entry was considered.
pub fn slot_kind_field_dims(kind: &str) -> Option<&'static [(&'static str, Dimension)]> {
    use Dimension::*;
    match kind {
        // Integrators
        "velocity-verlet" => Some(&[]),
        "langevin-baoab" => Some(&[
            ("friction", InverseTime),
            ("temperature", Temperature),
        ]),
        "nose-hoover-chain" => Some(&[
            ("temperature", Temperature),
            ("tau", Time),
        ]),
        "mtk-npt" => Some(&[
            ("temperature", Temperature),
            ("pressure", Pressure),
            ("tau_t", Time),
            ("tau_p", Time),
        ]),

        // Thermostats
        "berendsen" => Some(&[
            ("temperature", Temperature),
            ("tau", Time),
        ]),
        "csvr" => Some(&[
            ("temperature", Temperature),
            ("tau", Time),
        ]),
        "andersen" => Some(&[
            ("temperature", Temperature),
            ("collision_rate", InverseTime),
        ]),

        // Barostats
        "berendsen-barostat" => Some(&[
            ("pressure", Pressure),
            ("tau", Time),
            ("compressibility", InversePressure),
        ]),
        "c-rescale" => Some(&[
            ("pressure", Pressure),
            ("temperature", Temperature),
            ("tau", Time),
            ("compressibility", InversePressure),
        ]),

        // Constraints
        "settle-water" => Some(&[
            ("r_oh", Length),
            ("r_hh", Length),
        ]),

        // Minimizers
        "steepest-descent" => Some(&[
            ("initial_step", Length),
            ("max_step", Length),
            ("force_tolerance", Force),
            ("energy_tolerance", Energy),
        ]),

        _ => None,
    }
}

/// Rescale every unit-bearing field of a slot's `params` table in-place,
/// converting the user-supplied values to SI. No-op when `system` is SI
/// or when the kind is unknown.
pub fn convert_slot_params(system: UnitSystem, kind: &str, params: &mut toml::Value) {
    if system == UnitSystem::Si {
        return;
    }
    let Some(fields) = slot_kind_field_dims(kind) else {
        return;
    };
    let Some(table) = params.as_table_mut() else {
        return;
    };
    for (name, dim) in fields {
        let Some(slot) = table.get_mut(*name) else {
            continue;
        };
        if let Some(f) = slot.as_float() {
            *slot = toml::Value::Float(system.to_si(*dim, f));
        } else if let Some(i) = slot.as_integer() {
            *slot = toml::Value::Float(system.to_si(*dim, i as f64));
        }
        // else: leave non-numeric values alone — the builder's
        // validate_params will produce the right error message.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_factors_are_unity() {
        for dim in [
            Dimension::Length,
            Dimension::InverseLength,
            Dimension::Mass,
            Dimension::Charge,
            Dimension::Energy,
            Dimension::Time,
            Dimension::InverseTime,
            Dimension::Force,
            Dimension::Pressure,
            Dimension::InversePressure,
            Dimension::Temperature,
            Dimension::Velocity,
        ] {
            assert_eq!(UnitSystem::Si.factor(dim), 1.0);
        }
    }

    #[test]
    fn atomic_length_factor_matches_codata_2018_bohr() {
        // 5.29177210903e-11 m  — see atomic_constants.rs
        let f = UnitSystem::Atomic.factor(Dimension::Length);
        assert!((f - 5.29177210903e-11).abs() < 1e-25);
    }

    #[test]
    fn atomic_temperature_factor_matches_e_h_over_kb() {
        // E_h / k_B ≈ 3.158e5 K
        let f = UnitSystem::Atomic.factor(Dimension::Temperature);
        assert!((f - 315775.0248040668).abs() < 1e-7);
    }

    #[test]
    fn inverse_length_is_reciprocal_of_length() {
        let l = UnitSystem::Atomic.factor(Dimension::Length);
        let il = UnitSystem::Atomic.factor(Dimension::InverseLength);
        assert!((l * il - 1.0).abs() < 1e-15);
    }

    #[test]
    fn convert_slot_params_rescales_csvr_temperature_and_tau() {
        let mut params: toml::Value = toml::from_str(
            "temperature = 9.5e-4\ntau = 41.34\nseed = 11\n",
        )
        .unwrap();
        convert_slot_params(UnitSystem::Atomic, "csvr", &mut params);
        let t = params["temperature"].as_float().unwrap();
        let tau = params["tau"].as_float().unwrap();
        let seed = params["seed"].as_integer().unwrap();
        assert!((t - 9.5e-4 * 315775.0248040668).abs() < 1e-6);
        assert!((tau - 41.34 * 2.4188843265857195e-17).abs() < 1e-30);
        assert_eq!(seed, 11);
    }

    #[test]
    fn convert_slot_params_no_op_for_si() {
        let mut params: toml::Value = toml::from_str("temperature = 300.0\n").unwrap();
        convert_slot_params(UnitSystem::Si, "csvr", &mut params);
        assert_eq!(params["temperature"].as_float().unwrap(), 300.0);
    }

    #[test]
    fn convert_slot_params_no_op_for_unknown_kind() {
        let mut params: toml::Value = toml::from_str("temperature = 300.0\n").unwrap();
        convert_slot_params(UnitSystem::Atomic, "no-such-kind", &mut params);
        assert_eq!(params["temperature"].as_float().unwrap(), 300.0);
    }

    #[test]
    fn from_str_round_trips() {
        for s in ["si", "atomic"] {
            let u = UnitSystem::from_str(s).unwrap();
            assert_eq!(u.as_str(), s);
        }
        assert!(UnitSystem::from_str("imperial").is_none());
    }
}
