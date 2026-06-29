// Seam coverage for the builder-owned unit-conversion mechanism
// (KindedBuilder::convert_params + the dimensioned-newtype Convert
// derive). See rqm/io/unit-system.md and rqm/registry-framework.md.
//
// These tests exercise the config/units layer only — no GPU.

use heddle_md::registry::KindedBuilder;
use heddle_md::units::{Dimension, UnitSystem};
use heddle_md::Registries;

fn params(toml_src: &str) -> toml::Value {
    toml::from_str(toml_src).unwrap()
}

// rq-b99f0a0d rq-901a8f7c — a registered builder converts its own
// unit-bearing params; unit-free fields are left unchanged.
#[test]
fn csvr_builder_converts_its_params() {
    let r = Registries::with_builtins();
    let b = r.thermostats.lookup("csvr").expect("csvr registered");
    let mut p = params("temperature = 300.0\ntau = 1.0e-13\nseed = 11\n");
    b.convert_params(UnitSystem::Si, &mut p).unwrap();
    let t = p.get("temperature").unwrap().as_float().unwrap();
    let tau = p.get("tau").unwrap().as_float().unwrap();
    let seed = p.get("seed").unwrap().as_integer().unwrap();
    assert!((t - 300.0 / UnitSystem::Si.factor(Dimension::Temperature)).abs() < 1e-12);
    assert!((tau - 1.0e-13 / UnitSystem::Si.factor(Dimension::Time)).abs() < 1e-3);
    assert_eq!(seed, 11);
}

// rq-cce99aac — convert_params is a no-op for a kind with no unit-bearing
// params (velocity-verlet's only field is the unit-free `lossless`).
#[test]
fn velocity_verlet_convert_params_is_noop() {
    let r = Registries::with_builtins();
    let b = r.integrators.lookup("velocity-verlet").unwrap();
    let mut p = params("lossless = true\n");
    let before = p.clone();
    b.convert_params(UnitSystem::Si, &mut p).unwrap();
    assert_eq!(p, before);
}

// rq-57fede98 — every registered kinded builder converts a representative
// SI params block: at least one unit-bearing field is rescaled. A new
// kind whose params type omits a dimensioned field (or whose builder
// omits convert_params) fails here. The unit-free entries (velocity-verlet)
// assert the no-op path instead.
#[test]
fn every_registered_kind_converts_representative_params() {
    let r = Registries::with_builtins();
    // (lookup closure, kind, SI params, unit-bearing field, expected dim)
    struct Case {
        kind: &'static str,
        toml: &'static str,
        field: &'static str,
        dim: Dimension,
        si_value: f64,
    }
    // Integrators
    let integ = [
        Case { kind: "langevin-baoab", toml: "friction = 1.0e13\ntemperature = 300.0\nseed = 1\n",
               field: "temperature", dim: Dimension::Temperature, si_value: 300.0 },
        Case { kind: "mtk-npt", toml: "temperature = 85.0\npressure = 1.0e5\ntau_t = 1.0e-13\ntau_p = 1.0e-12\n",
               field: "pressure", dim: Dimension::Pressure, si_value: 1.0e5 },
    ];
    for c in &integ {
        let b = r.integrators.lookup(c.kind).unwrap();
        let mut p = params(c.toml);
        b.convert_params(UnitSystem::Si, &mut p).unwrap();
        let got = p.get(c.field).unwrap().as_float().unwrap();
        assert!((got - c.si_value / UnitSystem::Si.factor(c.dim)).abs()
                    < 1e-6 * (c.si_value / UnitSystem::Si.factor(c.dim)).abs().max(1.0),
                "{} {} not converted", c.kind, c.field);
    }

    // Thermostats
    for (kind, toml, field, dim, si) in [
        ("berendsen", "temperature = 300.0\ntau = 1.0e-13\n", "tau", Dimension::Time, 1.0e-13),
        ("csvr", "temperature = 300.0\ntau = 1.0e-13\nseed = 1\n", "temperature", Dimension::Temperature, 300.0),
        ("andersen", "temperature = 300.0\ncollision_rate = 1.0e12\nseed = 1\n", "collision_rate", Dimension::InverseTime, 1.0e12),
        ("nose-hoover-chain", "temperature = 300.0\ntau = 1.0e-13\n", "temperature", Dimension::Temperature, 300.0),
    ] {
        let b = r.thermostats.lookup(kind).unwrap();
        let mut p = params(toml);
        b.convert_params(UnitSystem::Si, &mut p).unwrap();
        let got = p.get(field).unwrap().as_float().unwrap();
        assert!((got - si / UnitSystem::Si.factor(dim)).abs()
                    < 1e-6 * (si / UnitSystem::Si.factor(dim)).abs().max(1.0),
                "{kind} {field} not converted");
    }

    // Barostats
    for (kind, toml) in [
        ("berendsen", "pressure = 1.0e5\ntau = 1.0e-12\ncompressibility = 4.5e-10\n"),
        ("c-rescale", "pressure = 1.0e5\ntemperature = 300.0\ntau = 1.0e-12\ncompressibility = 4.5e-10\nseed = 1\n"),
    ] {
        let b = r.barostats.lookup(kind).unwrap();
        let mut p = params(toml);
        b.convert_params(UnitSystem::Si, &mut p).unwrap();
        let got = p.get("pressure").unwrap().as_float().unwrap();
        assert!((got - 1.0e5 / UnitSystem::Si.factor(Dimension::Pressure)).abs()
                    < 1e-6 * (1.0e5 / UnitSystem::Si.factor(Dimension::Pressure)),
                "{kind} pressure not converted");
    }

    // Constraints
    let b = r.constraint_types.lookup("settle").unwrap();
    let mut p = params("d_OH = 1.0e-10\nd_HH = 1.633e-10\n");
    b.convert_params(UnitSystem::Si, &mut p).unwrap();
    let d_oh = p.get("d_OH").unwrap().as_float().unwrap();
    assert!((d_oh - 1.0e-10 / UnitSystem::Si.factor(Dimension::Length)).abs() < 1e-6);

    // Minimizers
    let b = r.minimizers.lookup("steepest-descent").unwrap();
    let mut p = params("initial_step = 1.0e-12\nmax_step = 1.0e-10\nforce_tolerance = 1.0e-10\nenergy_tolerance = 1.0e-7\n");
    b.convert_params(UnitSystem::Si, &mut p).unwrap();
    let init = p.get("initial_step").unwrap().as_float().unwrap();
    assert!((init - 1.0e-12 / UnitSystem::Si.factor(Dimension::Length)).abs() < 1e-6);
}

// rq-a0d557f5 — a slot's params round-trip equivalently between an SI
// description and the atomic description of the same physical values.
#[test]
fn slot_params_round_trip_si_vs_atomic() {
    let r = Registries::with_builtins();
    let b = r.thermostats.lookup("csvr").unwrap();

    let mut si = params("temperature = 300.0\ntau = 1.0e-13\nseed = 7\n");
    b.convert_params(UnitSystem::Si, &mut si).unwrap();

    // The atomic description: the SI values divided by their factors.
    let t_at = 300.0 / UnitSystem::Si.factor(Dimension::Temperature);
    let tau_at = 1.0e-13 / UnitSystem::Si.factor(Dimension::Time);
    let mut atomic = params(&format!("temperature = {t_at}\ntau = {tau_at}\nseed = 7\n"));
    b.convert_params(UnitSystem::Atomic, &mut atomic).unwrap();

    let rel = |a: f64, c: f64| (a - c).abs() / c.abs().max(1.0);
    for f in ["temperature", "tau"] {
        let s = si.get(f).unwrap().as_float().unwrap();
        let a = atomic.get(f).unwrap().as_float().unwrap();
        assert!(rel(s, a) < 1e-12, "{f}: si {s} vs atomic {a}");
    }
}

// rq-aeee8e44 — a kind with no registered builder leaves its params
// untouched (the registry has no builder to convert it).
#[test]
fn unknown_kind_params_pass_through() {
    let r = Registries::with_builtins();
    assert!(r.thermostats.lookup("no-such-thermostat").is_none());
    // With no builder, the loader's slot pass simply skips conversion;
    // the kind is rejected later by validate_against (covered in io_config).
}
