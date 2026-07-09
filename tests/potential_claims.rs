//! Category-scoped potential params claims (rq-73801d98): builder
//! claim declarations, claim dispatch, builder-owned validation and
//! unit conversion of open-shaped potential-table entry params, and
//! the `validate_against` integration. See
//! `rqm/forces/framework.md` (*Potential Params Claims*) and
//! `rqm/io/config-schema.md` (*Validation* item 13a).

use std::path::{Path, PathBuf};

use heddle_md::Registries;
use heddle_md::forces::{
    HarmonicAngleBuilder, HarmonicBondBuilder, LennardJonesBuilder, MorseBondedBuilder,
    PeriodicDihedralBuilder, Potential, PotentialBuildContext, PotentialBuilder,
    PotentialConfigEntry, PotentialParamsCategory, PotentialParamsClaim, PotentialRegistry,
    SpmeRealBuilder, SpmeReciprocalBuilder,
};
use heddle_md::forces::ForceFieldError;
use heddle_md::io::config::{
    BondTypeConfig, ConfigError, PairInteractionConfig, load_config_raw,
};
use heddle_md::units::{Dimension, UnitSystem};

fn tmp_path(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "heddle_potential_claims_{name}_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_config(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("sim.in.toml");
    std::fs::write(&path, contents).unwrap();
    path
}

fn minimal_config() -> String {
    r#"schema_version = 1
units = "atomic"
init = "argon.in.xyz"

[simulation]
seed = 12345
temperature = 300.0

[[phase]]
name = "run"
n_steps = 10
dt = 1.0e-15

[phase.integrator]
kind = "velocity-verlet"
lossless = false

[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
kind = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 1.0e-9
"#
    .to_string()
}

// rq-165037d6
#[test]
fn builtin_builders_declare_category_scoped_claims() {
    use PotentialParamsCategory::*;
    let cases: [(Box<dyn PotentialBuilder>, Option<(PotentialParamsCategory, &str)>); 7] = [
        (Box::new(LennardJonesBuilder), Some((PairInteraction, "lennard-jones"))),
        (Box::new(SpmeRealBuilder), None),
        (Box::new(SpmeReciprocalBuilder), None),
        (Box::new(MorseBondedBuilder), Some((BondType, "morse"))),
        (Box::new(HarmonicBondBuilder), Some((BondType, "harmonic"))),
        (Box::new(HarmonicAngleBuilder), Some((AngleType, "harmonic"))),
        (Box::new(PeriodicDihedralBuilder), Some((DihedralType, "periodic"))),
    ];
    for (builder, expected) in cases {
        let claim = builder.params_claim();
        match expected {
            Some((category, kind)) => {
                let c = claim.unwrap_or_else(|| panic!("{builder:?} should claim"));
                assert_eq!(c.category, category, "{builder:?}");
                assert_eq!(c.kind, kind, "{builder:?}");
            }
            None => assert!(claim.is_none(), "{builder:?} should claim nothing"),
        }
    }
}

/// A claimless custom builder relying on every trait default.
#[derive(Debug, Clone)]
struct ClaimlessBuilder;

impl PotentialBuilder for ClaimlessBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(None)
    }
}

// rq-d6888c00
#[test]
fn params_claim_defaults_to_none_with_noop_validate_and_convert() {
    let b = ClaimlessBuilder;
    assert!(b.params_claim().is_none());
    // Defaults are inert: validation accepts anything and conversion
    // leaves the params untouched.
    let entry = BondTypeConfig::morse("X", 1.0, 1.0, 1.0);
    assert!(b.validate_params(PotentialConfigEntry::BondType(&entry)).is_ok());
    let mut params = entry.params.clone();
    let before = params.clone();
    b.convert_params(UnitSystem::Si, &mut params).unwrap();
    assert_eq!(params, before);
}

/// Custom builder claiming `(BondType, "k")` that rejects every entry
/// with a marker error, used to observe dispatch order.
#[derive(Debug, Clone)]
struct MarkerBuilder {
    marker: &'static str,
}

impl PotentialBuilder for MarkerBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(None)
    }

    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::BondType,
            kind: "k",
        })
    }

    fn validate_params(&self, _entry: PotentialConfigEntry<'_>) -> Result<(), ConfigError> {
        Err(ConfigError::InvalidValue {
            field: self.marker.to_string(),
            reason: "marker".to_string(),
        })
    }
}

// rq-38fd4430
#[test]
fn claim_dispatch_selects_first_matching_builder_in_registration_order() {
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(MarkerBuilder { marker: "A" }));
    registry.register(Box::new(MarkerBuilder { marker: "B" }));
    let builder = registry
        .lookup_claim(PotentialParamsCategory::BondType, "k")
        .expect("claimed kind resolves");
    let entry = BondTypeConfig {
        name: "X".to_string(),
        kind: "k".to_string(),
        params: toml::Value::Table(toml::Table::new()),
    };
    match builder.validate_params(PotentialConfigEntry::BondType(&entry)) {
        Err(ConfigError::InvalidValue { field, .. }) => {
            assert_eq!(field, "A", "first registration wins");
        }
        other => panic!("unexpected: {other:?}"),
    }
    // No builder claims the same kind in a *different* category.
    assert!(
        registry
            .lookup_claim(PotentialParamsCategory::AngleType, "k")
            .is_none()
    );
}

// rq-8670941b
#[test]
fn claiming_builder_converts_its_own_unit_bearing_entry_params() {
    let builder = MorseBondedBuilder;
    let mut params: toml::Value = toml::from_str(
        "de = 1.65e-21\na = 1.9e10\nre = 3.4e-10\n",
    )
    .unwrap();
    builder.convert_params(UnitSystem::Si, &mut params).unwrap();
    let table = params.as_table().unwrap();
    let de = table.get("de").unwrap().as_float().unwrap();
    let a = table.get("a").unwrap().as_float().unwrap();
    let re = table.get("re").unwrap().as_float().unwrap();
    let f_e = UnitSystem::Si.factor(Dimension::Energy);
    let f_il = UnitSystem::Si.factor(Dimension::InverseLength);
    let f_l = UnitSystem::Si.factor(Dimension::Length);
    assert!((de - 1.65e-21 / f_e).abs() <= 1e-12 * de.abs());
    assert!((a - 1.9e10 / f_il).abs() <= 1e-12 * a.abs());
    assert!((re - 3.4e-10 / f_l).abs() <= 1e-12 * re.abs());
}

// rq-c3b69180
#[test]
fn claiming_builder_reads_a_common_field_during_validation() {
    // r_switch <= cutoff is a cross-field check against the entry's
    // common cutoff field, exposed through the entry view.
    let entry =
        PairInteractionConfig::lennard_jones(("A", "A"), 1.0, 1.0, 1.0e-9, Some(1.1e-9));
    let builder = LennardJonesBuilder;
    match builder.validate_params(PotentialConfigEntry::PairInteraction(&entry)) {
        Err(ConfigError::InvalidValue { field, .. }) => assert_eq!(field, "r_switch"),
        other => panic!("unexpected: {other:?}"),
    }
    // The same params under a large-enough cutoff validate cleanly.
    let ok_entry =
        PairInteractionConfig::lennard_jones(("A", "A"), 1.0, 1.0, 2.0e-9, Some(1.1e-9));
    assert!(
        builder
            .validate_params(PotentialConfigEntry::PairInteraction(&ok_entry))
            .is_ok()
    );
}

// rq-03b385ae
#[test]
fn validate_against_rejects_potential_table_kind_no_builder_claims() {
    let dir = tmp_path("unclaimed_kind");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"CC\"\nkind = \"fene\"\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config_raw(&path).unwrap();
    match cfg.validate_against(&Registries::with_builtins()).unwrap_err() {
        ConfigError::UnknownKind { slot, kind } => {
            assert_eq!(slot, "bond_types");
            assert_eq!(kind, "fene");
        }
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

/// Custom builder claiming `(BondType, "fene")` whose validate_params
/// accepts the entry.
#[derive(Debug, Clone)]
struct FeneBuilder;

impl PotentialBuilder for FeneBuilder {
    fn build(
        &self,
        _cx: &PotentialBuildContext<'_>,
    ) -> Result<Option<Box<dyn Potential>>, ForceFieldError> {
        Ok(None)
    }

    fn params_claim(&self) -> Option<PotentialParamsClaim> {
        Some(PotentialParamsClaim {
            category: PotentialParamsCategory::BondType,
            kind: "fene",
        })
    }
}

// rq-6b721a52
#[test]
fn validate_against_accepts_kind_claimed_by_registered_custom_builder() {
    let dir = tmp_path("custom_claim");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"CC\"\nkind = \"fene\"\nk = 1.0\nr_max = 2.0\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    let cfg = load_config_raw(&path).unwrap();
    let mut registries = Registries::with_builtins();
    registries.register_potential(Box::new(FeneBuilder));
    cfg.validate_against(&registries)
        .expect("custom claim satisfies the kind check");
}

// rq-c118d720
#[test]
fn validate_against_surfaces_claiming_builder_validate_params_error() {
    let dir = tmp_path("claim_validate_err");
    let body = format!(
        "{}\n[[bond_types]]\nname = \"CC\"\nkind = \"morse\"\nde = 0.0\na = 1.9e10\nre = 3.4e-10\n",
        minimal_config()
    );
    let path = write_config(&dir, &body);
    match heddle_md::io::config::load_config(&path).unwrap_err() {
        ConfigError::InvalidValue { field, .. } => {
            assert_eq!(field, "bond_types[0].de");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// rq-47be884d
#[test]
fn potential_params_claims_are_category_scoped() {
    // "harmonic" is claimed for the bond-type and angle-type categories;
    // a claim in one category never satisfies another.
    let dir = tmp_path("category_scoped");
    let body = minimal_config().replace(
        "kind = \"lennard-jones\"",
        "kind = \"harmonic\"",
    );
    let path = write_config(&dir, &body);
    let cfg = load_config_raw(&path).unwrap();
    match cfg.validate_against(&Registries::with_builtins()).unwrap_err() {
        ConfigError::UnknownKind { slot, kind } => {
            assert_eq!(slot, "pair_interactions");
            assert_eq!(kind, "harmonic");
        }
        other => panic!("expected UnknownKind, got {other:?}"),
    }
}

// rq-943724af
#[test]
fn build_deserialises_typed_params_from_claimed_entries() {
    use heddle_md::forces::{
        AngleList, Bond, BondList, DihedralList, ExclusionList, ForceField,
    };
    use heddle_md::gpu::init_device;
    use heddle_md::io::config::NeighborListConfig;
    use heddle_md::pbc::SimulationBox;

    let gpu = init_device().unwrap();
    let sim_box = SimulationBox::new(&gpu.device, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap();
    let bond_types = [BondTypeConfig::morse("CC", 1.0, 2.0, 1.0)];
    let bonds = BondList {
        bonds: vec![Bond { atom_i: 0, atom_j: 1, bond_type_index: 0 }],
        atom_bond_offsets: vec![0, 1, 2],
        atom_bond_indices: vec![0, 1],
        particle_count: 2,
    };
    let mut registry = PotentialRegistry::new();
    registry.register(Box::new(MorseBondedBuilder));
    let ff = ForceField::new(
        &registry,
        &gpu,
        2,
        &sim_box,
        &[],
        &[],
        &bond_types,
        &[],
        &[],
        None,
        &[],
        &bonds,
        &AngleList::empty(0),
        &DihedralList::empty(0),
        &ExclusionList::empty(2),
        &NeighborListConfig::AllPairs,
    )
    .unwrap();
    // The builder deserialised MorseBondParams from the claimed entry's
    // params and activated the slot.
    assert_eq!(ff.slots.len(), 1);
    assert_eq!(ff.slots[0].label(), "morse_bonded");
}
