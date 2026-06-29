//! Unit-system selection for the I/O boundary.
//!
//! The simulation engine stores and computes in **Hartree atomic units**
//! internally. This module exists to let the user write input files and
//! read output files in either SI (the default) or Hartree atomic
//! units; the loader converts user → engine on read and the writers
//! convert engine → user on write. The `units` top-level TOML field
//! selects the system.
//!
//! Conversion factors live in the generated `atomic_constants` submodule
//! and originate from QCElemental (CODATA-2018). Regenerate them with
//! `python3 scripts/gen_atomic_units.py`.

pub mod atomic_constants;

use atomic_constants::{
    CHARGE_E_TO_C, ENERGY_HARTREE_TO_J, FORCE_AU_TO_N, LENGTH_BOHR_TO_M, MASS_ME_TO_KG,
    PRESSURE_AU_TO_PA, TEMPERATURE_AU_TO_K, TIME_AU_TO_S, VELOCITY_AU_TO_M_PER_S,
};

// rq-34446ef5
/// The unit system the user's I/O files are written in. `Atomic` is the
/// engine's native form and a no-op for every conversion; `Si` is the
/// default selector and triggers the SI ↔ atomic conversion at every
/// I/O boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnitSystem {
    #[default]
    Si,
    Atomic,
}

impl UnitSystem {
    // rq-1c857adf
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "si" => Some(UnitSystem::Si),
            "atomic" => Some(UnitSystem::Atomic),
            _ => None,
        }
    }

    // rq-7513f933
    pub fn as_str(self) -> &'static str {
        match self {
            UnitSystem::Si => "si",
            UnitSystem::Atomic => "atomic",
        }
    }
}

// rq-0d49f455
/// The physical dimensions the I/O conversion knows how to rescale.
/// Every unit-bearing scalar the loader handles or a writer emits maps
/// to exactly one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// A pure number or counter that should pass through unchanged
    /// under either `UnitSystem`. Both `to_user` and `from_user`
    /// return the value as-is.
    Dimensionless,
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
    // rq-fb7c4429
    /// The user-system value of one atomic unit of `dim`. Returns
    /// `1.0` for [`UnitSystem::Atomic`] (the user's view coincides
    /// with the engine), and the SI-per-atomic-unit factor for
    /// [`UnitSystem::Si`]. Returns `1.0` for
    /// [`Dimension::Dimensionless`] independent of `self`.
    pub fn factor(self, dim: Dimension) -> f64 {
        if dim == Dimension::Dimensionless {
            return 1.0;
        }
        if self == UnitSystem::Atomic {
            return 1.0;
        }
        match dim {
            Dimension::Dimensionless => 1.0,
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

    // rq-a7677c61
    /// Output-direction conversion: translates one engine-side
    /// atomic-unit scalar into the user's chosen unit system. Returns
    /// `value * self.factor(dim)`. Used by trajectory, CSV-log, and
    /// minlog writers.
    pub fn to_user(self, dim: Dimension, value: f64) -> f64 {
        value * self.factor(dim)
    }

    // rq-fdeba84b
    /// Input-direction conversion: translates one user-supplied scalar
    /// into the engine's atomic-unit representation. Returns
    /// `value / self.factor(dim)`. Used by `load_config` and
    /// `load_init_state`.
    pub fn from_user(self, dim: Dimension, value: f64) -> f64 {
        value / self.factor(dim)
    }
}

// rq-ecb8aa60
/// Look up the unit-bearing fields of a slot kind. Returns `None` for
/// kinds this module doesn't know about — the slot will pass through
/// unchanged, and the builder registry will reject it later if the
/// kind is genuinely invalid.
///
/// Kinds that exist but carry no unit-bearing fields (e.g.
/// `velocity-verlet`) return `Some(&[])` so a maintainer adding a new
/// slot can spot at a glance that the entry was considered.
///
/// The table below mirrors the kinds registered by
/// `Registries::with_builtins()` (see `src/registries.rs`) and by
/// each `XRegistry::with_builtins()` (in `src/integrator/mod.rs` and
/// `src/minimizer/mod.rs`). When adding a new built-in builder, add
/// the matching arm here. The `convert_slot_params` special-case
/// block immediately below this function handles nested-array shapes
/// (`shake`'s `constraints[k].d`) that the flat table can't express.
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

        // Constraints — the SHAKE entry is special-cased by
        // `convert_slot_params` (its `constraints` field is a nested
        // array of tables) and therefore declares an empty top-level
        // field set here.
        "shake" => Some(&[]),
        // rq-eecd4961 — SETTLE's two water distances are plain
        // top-level length fields, converted by the table-driven path.
        "settle" => Some(&[
            ("d_OH", Length),
            ("d_HH", Length),
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

// rq-8f5ebdc1
/// Rescale every unit-bearing field of a slot's `params` table in-place,
/// converting the user-supplied values into engine-side atomic units.
/// No-op when `system` is `Atomic` (identity) or when the kind is
/// unknown.
pub fn convert_slot_params(system: UnitSystem, kind: &str, params: &mut toml::Value) {
    if system == UnitSystem::Atomic {
        return;
    }
    // Kinds whose params include nested arrays of tables (e.g. the
    // SHAKE `constraints` array) cannot be rescaled by the
    // table-driven `slot_kind_field_dims` path. They are converted
    // explicitly here before the regular field walk.
    if kind == "shake" {
        if let Some(table) = params.as_table_mut() {
            if let Some(arr) = table.get_mut("constraints").and_then(|v| v.as_array_mut()) {
                for entry in arr.iter_mut() {
                    let Some(sub) = entry.as_table_mut() else {
                        continue;
                    };
                    let Some(d_slot) = sub.get_mut("d") else {
                        continue;
                    };
                    if let Some(f) = d_slot.as_float() {
                        *d_slot = toml::Value::Float(system.from_user(Dimension::Length, f));
                    } else if let Some(i) = d_slot.as_integer() {
                        *d_slot = toml::Value::Float(
                            system.from_user(Dimension::Length, i as f64),
                        );
                    }
                }
            }
        }
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
            *slot = toml::Value::Float(system.from_user(*dim, f));
        } else if let Some(i) = slot.as_integer() {
            *slot = toml::Value::Float(system.from_user(*dim, i as f64));
        }
        // else: leave non-numeric values alone — the builder's
        // validate_params will produce the right error message.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // rq-4c40f859
    #[test]
    fn atomic_factors_are_unity() {
        for dim in [
            Dimension::Dimensionless,
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
            assert_eq!(UnitSystem::Atomic.factor(dim), 1.0);
        }
    }

    #[test]
    fn si_dimensionless_factor_is_unity() {
        assert_eq!(UnitSystem::Si.factor(Dimension::Dimensionless), 1.0);
    }

    #[test]
    fn si_length_factor_matches_codata_2018_bohr() {
        // 5.29177210903e-11 m  — see atomic_constants.rs
        let f = UnitSystem::Si.factor(Dimension::Length);
        assert!((f - 5.29177210903e-11).abs() < 1e-25);
    }

    #[test]
    fn si_temperature_factor_matches_e_h_over_kb() {
        // E_h / k_B ≈ 3.158e5 K
        let f = UnitSystem::Si.factor(Dimension::Temperature);
        assert!((f - 315775.0248040668).abs() < 1e-7);
    }

    #[test]
    fn si_inverse_length_is_reciprocal_of_length() {
        let l = UnitSystem::Si.factor(Dimension::Length);
        let il = UnitSystem::Si.factor(Dimension::InverseLength);
        assert!((l * il - 1.0).abs() < 1e-15);
    }

    #[test]
    fn from_user_converts_si_to_atomic() {
        // 300 K → ~9.5e-4 E_h / k_B
        let t = UnitSystem::Si.from_user(Dimension::Temperature, 300.0);
        let expected = 300.0 / 315775.0248040668;
        assert!((t - expected).abs() < 1e-15);
    }

    #[test]
    fn to_user_converts_atomic_to_si() {
        // 9.5e-4 E_h / k_B → ~300 K in SI mode
        let t = UnitSystem::Si.to_user(Dimension::Temperature, 9.5e-4);
        let expected = 9.5e-4 * 315775.0248040668;
        assert!((t - expected).abs() < 1e-7);
    }

    #[test]
    fn atomic_mode_to_user_is_identity() {
        assert_eq!(UnitSystem::Atomic.to_user(Dimension::Length, 1.5), 1.5);
        assert_eq!(UnitSystem::Atomic.to_user(Dimension::Temperature, 9.5e-4), 9.5e-4);
    }

    #[test]
    fn atomic_mode_from_user_is_identity() {
        assert_eq!(UnitSystem::Atomic.from_user(Dimension::Length, 1.5), 1.5);
        assert_eq!(UnitSystem::Atomic.from_user(Dimension::Temperature, 9.5e-4), 9.5e-4);
    }

    #[test]
    fn round_trip_to_user_then_from_user_is_identity() {
        let dim = Dimension::Length;
        let value = 18.9_f64;
        let user_value = UnitSystem::Si.to_user(dim, value);
        let back = UnitSystem::Si.from_user(dim, user_value);
        assert!((back - value).abs() < 1e-12 * value);
    }

    #[test]
    fn convert_slot_params_rescales_si_csvr_temperature_and_tau() {
        let mut params: toml::Value = toml::from_str(
            "temperature = 300.0\ntau = 1.0e-13\nseed = 11\n",
        )
        .unwrap();
        convert_slot_params(UnitSystem::Si, "csvr", &mut params);
        let t = params["temperature"].as_float().unwrap();
        let tau = params["tau"].as_float().unwrap();
        let seed = params["seed"].as_integer().unwrap();
        let expected_t = 300.0 / 315775.0248040668;
        let expected_tau = 1.0e-13 / 2.4188843265857195e-17;
        assert!((t - expected_t).abs() < 1e-12);
        assert!((tau - expected_tau).abs() < 1e-3); // tau is ~4e3
        assert_eq!(seed, 11);
    }

    // rq-eecd4961 — SETTLE's d_OH / d_HH must be rescaled SI->atomic so
    // the kernels (which work in Bohr) see Bohr targets, not metres.
    #[test]
    fn convert_slot_params_rescales_si_settle_distances() {
        let mut params: toml::Value =
            toml::from_str("d_OH = 1.0e-10\nd_HH = 1.633e-10\n").unwrap();
        convert_slot_params(UnitSystem::Si, "settle", &mut params);
        let bohr = 5.29177210903e-11;
        let d_oh = params["d_OH"].as_float().unwrap();
        let d_hh = params["d_HH"].as_float().unwrap();
        assert!((d_oh - 1.0e-10 / bohr).abs() < 1e-6, "d_OH = {d_oh}");
        assert!((d_hh - 1.633e-10 / bohr).abs() < 1e-6, "d_HH = {d_hh}");
        // Atomic-units input is left unchanged.
        let mut atomic: toml::Value =
            toml::from_str("d_OH = 1.889726\nd_HH = 3.085926\n").unwrap();
        convert_slot_params(UnitSystem::Atomic, "settle", &mut atomic);
        assert_eq!(atomic["d_OH"].as_float().unwrap(), 1.889726);
    }

    #[test]
    fn convert_slot_params_no_op_for_atomic() {
        let mut params: toml::Value =
            toml::from_str("temperature = 9.5e-4\n").unwrap();
        convert_slot_params(UnitSystem::Atomic, "csvr", &mut params);
        assert_eq!(params["temperature"].as_float().unwrap(), 9.5e-4);
    }

    // rq-aeee8e44
    #[test]
    fn convert_slot_params_no_op_for_unknown_kind() {
        let mut params: toml::Value = toml::from_str("temperature = 300.0\n").unwrap();
        convert_slot_params(UnitSystem::Si, "no-such-kind", &mut params);
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
