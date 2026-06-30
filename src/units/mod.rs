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

// The `#[derive(Convert)]` macro (macro namespace) shares its name with
// the `Convert` trait below (type namespace), the same way `serde`'s
// `Serialize` is both a derive and a trait.
pub use heddle_md_derive::Convert;

// rq-bf5df23e
/// Applies the I/O-boundary unit rescaling to a value or an aggregate of
/// values: `from_user` rescales user → atomic (input), `to_user`
/// rescales atomic → user (output). The dimensioned newtypes below are
/// the leaves; `#[derive(Convert)]` recurses over the fields of a struct
/// or enum.
pub trait Convert {
    fn from_user(&mut self, units: UnitSystem);
    fn to_user(&mut self, units: UnitSystem);
}

// rq-bf5df23e
/// Dimensioned scalar newtypes — one transparent `f64` wrapper per
/// unit-bearing dimension. Each (de)serialises as a bare number, so TOML
/// and extended-XYZ syntax are unchanged, and carries its `Dimension` on
/// the type.
macro_rules! dimensioned_scalars {
    ($($name:ident => $dim:ident),* $(,)?) => {
        $(
            #[derive(Clone, Copy, Debug, PartialEq, Default, serde::Deserialize, serde::Serialize)]
            #[serde(transparent)]
            pub struct $name(pub f64);

            impl Convert for $name {
                fn from_user(&mut self, units: UnitSystem) {
                    self.0 = units.from_user(Dimension::$dim, self.0);
                }
                fn to_user(&mut self, units: UnitSystem) {
                    self.0 = units.to_user(Dimension::$dim, self.0);
                }
            }
        )*
    };
}

dimensioned_scalars! {
    Length => Length,
    InverseLength => InverseLength,
    Mass => Mass,
    Charge => Charge,
    Energy => Energy,
    Time => Time,
    InverseTime => InverseTime,
    Force => Force,
    Pressure => Pressure,
    InversePressure => InversePressure,
    Temperature => Temperature,
    Velocity => Velocity,
}

// rq-bf5df23e
/// Unit-free leaf types convert as no-ops, so a derived struct may hold
/// counts, seeds, flags, and names alongside its dimensioned fields.
macro_rules! convert_noop {
    ($($t:ty),* $(,)?) => {
        $(
            impl Convert for $t {
                fn from_user(&mut self, _units: UnitSystem) {}
                fn to_user(&mut self, _units: UnitSystem) {}
            }
        )*
    };
}

convert_noop!(f64, f32, i64, u64, u32, usize, i32, bool, String);

// rq-bf5df23e — aggregate blanket impls.
impl<T: Convert> Convert for Option<T> {
    fn from_user(&mut self, units: UnitSystem) {
        if let Some(v) = self {
            v.from_user(units);
        }
    }
    fn to_user(&mut self, units: UnitSystem) {
        if let Some(v) = self {
            v.to_user(units);
        }
    }
}

impl<T: Convert> Convert for Vec<T> {
    fn from_user(&mut self, units: UnitSystem) {
        for v in self.iter_mut() {
            v.from_user(units);
        }
    }
    fn to_user(&mut self, units: UnitSystem) {
        for v in self.iter_mut() {
            v.to_user(units);
        }
    }
}

impl<T: Convert, const N: usize> Convert for [T; N] {
    fn from_user(&mut self, units: UnitSystem) {
        for v in self.iter_mut() {
            v.from_user(units);
        }
    }
    fn to_user(&mut self, units: UnitSystem) {
        for v in self.iter_mut() {
            v.to_user(units);
        }
    }
}

// A `toml::Spanned<T>` (used for source-ordered phase entries) converts
// through its inner value.
impl<T: Convert> Convert for toml::Spanned<T> {
    fn from_user(&mut self, units: UnitSystem) {
        self.get_mut().from_user(units);
    }
    fn to_user(&mut self, units: UnitSystem) {
        self.get_mut().to_user(units);
    }
}

// An open-shaped `toml::Value` carries slot params that are converted by
// the owning builder's `convert_params`, not by the typed-field pass, so
// the typed pass treats it as a no-op leaf.
impl Convert for toml::Value {
    fn from_user(&mut self, _units: UnitSystem) {}
    fn to_user(&mut self, _units: UnitSystem) {}
}


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Convert)]
    struct Sub {
        d: Length,
    }

    #[derive(Convert)]
    struct Outer {
        len: Length,
        maybe_e: Option<Energy>,
        subs: Vec<Sub>,
        name: String,
        count: u64,
        flag: bool,
    }

    // rq-b45b09f9 — the Convert derive recurses into nested struct, Option,
    // and Vec fields, and leaves unit-free leaf fields untouched.
    #[test]
    fn convert_derive_recurses_into_nested_fields() {
        let bohr = LENGTH_BOHR_TO_M;
        let hartree = ENERGY_HARTREE_TO_J;
        let mut o = Outer {
            len: Length(1.0e-9),
            maybe_e: Some(Energy(2.0e-21)),
            subs: vec![Sub { d: Length(3.0e-10) }, Sub { d: Length(4.0e-10) }],
            name: "water".to_string(),
            count: 7,
            flag: true,
        };
        o.from_user(UnitSystem::Si);
        assert!((o.len.0 - 1.0e-9 / bohr).abs() < 1e-6);
        assert!((o.maybe_e.unwrap().0 - 2.0e-21 / hartree).abs() < 1e-30);
        assert!((o.subs[0].d.0 - 3.0e-10 / bohr).abs() < 1e-6);
        assert!((o.subs[1].d.0 - 4.0e-10 / bohr).abs() < 1e-6);
        assert_eq!(o.name, "water");
        assert_eq!(o.count, 7);
        assert!(o.flag);
        // Atomic mode is the identity.
        let mut a = Outer {
            len: Length(1.5),
            maybe_e: None,
            subs: vec![],
            name: String::new(),
            count: 0,
            flag: false,
        };
        a.from_user(UnitSystem::Atomic);
        assert_eq!(a.len.0, 1.5);
        // Round-trip: to_user undoes from_user.
        let mut r = Length(2.5e-10);
        r.from_user(UnitSystem::Si);
        r.to_user(UnitSystem::Si);
        assert!((r.0 - 2.5e-10).abs() < 1e-24);
    }

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


    // rq-eecd4961 — SETTLE's d_OH / d_HH must be rescaled SI->atomic so



    #[test]
    fn from_str_round_trips() {
        for s in ["si", "atomic"] {
            let u = UnitSystem::from_str(s).unwrap();
            assert_eq!(u.as_str(), s);
        }
        assert!(UnitSystem::from_str("imperial").is_none());
    }
}
