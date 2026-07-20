//! Potential consistency harness. See `rqm/forces/potential-consistency-harness.md`.
//!
//! Verifies that a force-field potential's kernel produces forces consistent
//! with its energy, obeys Newton's third law, reports a scalar virial matching
//! the force, joins smoothly and vanishes at its cutoff (pair), and matches
//! known analytic reference points. The checking routine is generic over an
//! `Evaluator`, so the same logic drives the real GPU force pipeline (the
//! built-in fixtures) and, in tests, a CPU closure with an injected defect (the
//! negative scenarios).

#![allow(dead_code)]

use std::collections::HashSet;

use heddle_md::forces::topology::{Angle, Dihedral};
use heddle_md::forces::{
    AngleList, Bond, BondList, DihedralList, ExclusionList, ForceField, HarmonicAngleBuilder,
    HarmonicBondBuilder, MorseBondedBuilder, PeriodicDihedralBuilder, PotentialRegistry,
    SpmeExclusionBuilder, SpmeRealBuilder,
};
use heddle_md::forces::{AggregateLevel, LennardJonesBuilder};
use heddle_md::gpu::{GpuContext, ParticleBuffers};
use heddle_md::io::config::{
    AngleTypeConfig, BondTypeConfig, DihedralTypeConfig, NeighborListConfig, PairInteractionConfig,
    ParticleTypeConfig, SpmeConfig,
};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

// =====================================================================
// Public API types
// =====================================================================

/// The four fragment-composed shapes the harness drives. rq-45ceac42
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PotentialShape {
    Pair,
    Bond,
    Angle,
    Dihedral,
}

/// A relative tolerance with an absolute floor. `Default` supplies values
/// tuned for the f32 pipeline. rq-7c83a420
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub rel: f64,
    pub abs: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Tolerance { rel: 2.0e-2, abs: 2.0e-3 }
    }
}

impl Tolerance {
    pub fn new(rel: f64, abs: f64) -> Self {
        Tolerance { rel, abs }
    }
    /// True when `actual` matches `expected` within this tolerance.
    pub fn close(&self, expected: f64, actual: f64) -> bool {
        (expected - actual).abs() <= self.abs + self.rel * expected.abs().max(actual.abs())
    }
}

/// An analytic check at one coordinate value. rq-96809580
#[derive(Clone, Copy, Debug)]
pub struct ReferencePoint {
    pub coordinate: f64,
    pub energy: Option<f64>,
    pub coord_force: Option<f64>, // expected generalized force −dU/dq
    pub tol: Tolerance,
}

// =====================================================================
// Evaluator: maps a geometry to (energy, per-atom forces, scalar virial)
// =====================================================================

/// One evaluation of a potential on a set of atom positions.
#[derive(Clone, Debug)]
pub struct Eval {
    pub energy: f64,
    pub forces: Vec<[f64; 3]>,
    pub virial: f64,
}

/// Evaluates a potential's total energy, per-atom forces, and total scalar
/// virial at arbitrary atom positions.
pub trait Evaluator {
    fn eval(&mut self, positions: &[[f64; 3]]) -> Eval;
}

/// Drives the real GPU force pipeline: a `ForceField` carrying exactly one
/// slot, re-evaluated on a fresh `ParticleBuffers` per geometry.
pub struct GpuEvaluator<'a> {
    gpu: &'a GpuContext,
    ff: ForceField,
    sim_box: SimulationBox,
    masses: Vec<Real>,
    charges: Vec<Real>,
    type_indices: Vec<u32>,
}

impl<'a> GpuEvaluator<'a> {
    fn n(&self) -> usize {
        self.type_indices.len()
    }
}

impl Evaluator for GpuEvaluator<'_> {
    fn eval(&mut self, positions: &[[f64; 3]]) -> Eval {
        let n = positions.len();
        assert_eq!(n, self.n(), "geometry atom count must match the built system");
        let px: Vec<Real> = positions.iter().map(|p| p[0] as Real).collect();
        let py: Vec<Real> = positions.iter().map(|p| p[1] as Real).collect();
        let pz: Vec<Real> = positions.iter().map(|p| p[2] as Real).collect();
        let state = ParticleState::new(
            px,
            py,
            pz,
            vec![0.0 as Real; n],
            vec![0.0 as Real; n],
            vec![0.0 as Real; n],
            self.masses.clone(),
            self.charges.clone(),
            self.type_indices.clone(),
            None,
            None,
        )
        .expect("build ParticleState");
        let mut buffers = ParticleBuffers::new(self.gpu, &state).expect("ParticleBuffers");
        let mut timings = Timings::new(self.gpu).expect("Timings");
        self.ff
            .step(&mut buffers, &self.sim_box, &mut timings, AggregateLevel::ForcesAndScalars)
            .expect("force-field step");
        let fx: Vec<Real> = self.gpu.device.dtoh_sync_copy(&buffers.forces_x).unwrap();
        let fy: Vec<Real> = self.gpu.device.dtoh_sync_copy(&buffers.forces_y).unwrap();
        let fz: Vec<Real> = self.gpu.device.dtoh_sync_copy(&buffers.forces_z).unwrap();
        let en: Vec<Real> = self.gpu.device.dtoh_sync_copy(&buffers.potential_energies).unwrap();
        let vi: Vec<Real> = self.gpu.device.dtoh_sync_copy(&buffers.virials).unwrap();
        Eval {
            energy: en.iter().map(|&v| v as f64).sum(),
            forces: (0..n).map(|i| [fx[i] as f64, fy[i] as f64, fz[i] as f64]).collect(),
            virial: vi.iter().map(|&v| v as f64).sum(),
        }
    }
}

// =====================================================================
// Fixture: a shape, a geometry generator, and a GPU system builder
// =====================================================================

type GeometryFn = Box<dyn Fn(f64) -> Vec<[f64; 3]>>;
/// Returns (ForceField, SimulationBox, masses, charges, type_indices).
type BuildFn = Box<dyn Fn(&GpuContext) -> (ForceField, SimulationBox, Vec<Real>, Vec<Real>, Vec<u32>)>;

/// Everything needed to drive one potential through the harness. rq-e8b786b3
pub struct ConsistencyFixture {
    pub label: &'static str,
    pub shape: PotentialShape,
    pub samples: Vec<f64>,
    pub reference_points: Vec<ReferencePoint>,
    pub fd_tol: Tolerance,
    pub newton_tol: Tolerance,
    pub virial_tol: Tolerance,
    pub continuity_tol: Tolerance,
    pub r_switch: f64,
    pub cutoff: f64,
    /// The fragment declares `CutoffHandling::Unbounded`: it has no cutoff to be
    /// continuous across and is required *not* to vanish beyond one, so the
    /// switch/cutoff continuity check does not apply. See
    /// `rqm/forces/potential-consistency-harness.md`.
    pub unbounded: bool,
    pub fd_step: f64,
    pub scale_eps: f64,
    pub coord_fd_step: f64,
    pub continuity_jump_ratio: f64,
    geometry: GeometryFn,
    build: BuildFn,
}

const HARNESS_BOX: Real = 40.0;

/// `erf(0.6 r) / r` at r = 1 and r = 2 — the closed form of the SPME
/// excluded-pair correction's measured energy in the fixture below.
const REF_U_1: f64 = 6.038560908479259e-01;
const REF_U_2: f64 = 4.551569891148177e-01;

impl ConsistencyFixture {
    fn base(label: &'static str, shape: PotentialShape, samples: Vec<f64>, geometry: GeometryFn, build: BuildFn) -> Self {
        ConsistencyFixture {
            label,
            shape,
            samples,
            reference_points: Vec::new(),
            fd_tol: Tolerance::default(),
            newton_tol: Tolerance::new(1.0e-3, 1.0e-3),
            virial_tol: Tolerance::new(3.0e-2, 1.0e-2),
            continuity_tol: Tolerance::new(0.0, 1.0e-4),
            r_switch: 0.0,
            cutoff: 0.0,
            unbounded: false,
            fd_step: 1.0e-3,
            scale_eps: 1.0e-3,
            coord_fd_step: 1.0e-3,
            continuity_jump_ratio: 30.0,
            geometry,
            build,
        }
    }

    pub fn with_reference_points(mut self, points: Vec<ReferencePoint>) -> Self {
        self.reference_points = points;
        self
    }

    pub fn with_fd_tol(mut self, tol: Tolerance) -> Self {
        self.fd_tol = tol;
        self
    }
    pub fn with_virial_tol(mut self, tol: Tolerance) -> Self {
        self.virial_tol = tol;
        self
    }
    pub fn with_newton_tol(mut self, tol: Tolerance) -> Self {
        self.newton_tol = tol;
        self
    }

    /// A fixture with no GPU system builder, for driving the checks against a
    /// custom `Evaluator` (used by tests that inject a defect at the evaluator
    /// level). `build_system` / `assert_potential_consistent` must not be
    /// called on it.
    pub fn for_checks(
        label: &'static str,
        shape: PotentialShape,
        samples: Vec<f64>,
        geometry: impl Fn(f64) -> Vec<[f64; 3]> + 'static,
        r_switch: f64,
        cutoff: f64,
    ) -> Self {
        let build: BuildFn = Box::new(|_| panic!("for_checks fixture has no GPU system builder"));
        let mut f = Self::base(label, shape, samples, Box::new(geometry), build);
        f.r_switch = r_switch;
        f.cutoff = cutoff;
        f
    }

    /// Build the isolated single-slot GPU system for this fixture.
    pub fn build_system(&self, gpu: &GpuContext) -> (ForceField, SimulationBox, Vec<Real>, Vec<Real>, Vec<u32>) {
        (self.build)(gpu)
    }

    /// Construct the GPU evaluator that drives this fixture's isolated slot.
    pub fn gpu_evaluator<'a>(&self, gpu: &'a GpuContext) -> GpuEvaluator<'a> {
        let (ff, sim_box, masses, charges, type_indices) = (self.build)(gpu);
        GpuEvaluator { gpu, ff, sim_box, masses, charges, type_indices }
    }

    // --- Shape constructors ---------------------------------------

    /// A pairwise van-der-Waals / screened-Coulomb fixture. `register`
    /// populates the single-builder registry; `pair_interactions`, charges,
    /// and `spme` supply its activation data.
    /// A fixture for a `FragmentPasses::CorrectionOnly` fragment.
    ///
    /// Such a fragment is evaluated only by the correction pass, which walks the
    /// modified-pair list — so the fixture carries an exclusion between the two
    /// atoms, and what the harness measures is `(scale_coul - 1) x evaluate`.
    /// The finite-difference, Newton and virial invariants are unaffected by
    /// that constant factor; a reference point must account for it.
    ///
    /// The correction pass only launches when a neighbour list exists, and a
    /// neighbour list only exists when some slot reports a cutoff. A
    /// `CorrectionOnly` fragment reports none (it has no cutoff — that is the
    /// point of it), so the fixture registers a zero-epsilon Lennard-Jones slot
    /// alongside it: it supplies the cutoff and contributes exactly zero energy
    /// and zero force, leaving the measurement attributable to the fragment
    /// under test.
    #[allow(clippy::too_many_arguments)]
    pub fn pair_correction(
        label: &'static str,
        register: impl Fn(&mut PotentialRegistry) + 'static,
        particle_types: Vec<ParticleTypeConfig>,
        pair_interactions: Vec<PairInteractionConfig>,
        charges: Vec<Real>,
        spme: Option<SpmeConfig>,
        scale_coul: Real,
        cutoff: f64,
        samples: Vec<f64>,
    ) -> Self {
        let geometry: GeometryFn = Box::new(|r| vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]]);
        let type_indices = vec![0u32, if particle_types.len() > 1 { 1 } else { 0 }];
        let build: BuildFn = Box::new(move |gpu| {
            let mut reg = PotentialRegistry::new();
            reg.register(Box::new(heddle_md::forces::LennardJonesBuilder));
            register(&mut reg);
            let sim_box = SimulationBox::new(
                &gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0,
            )
            .unwrap();
            let exclusions =
                super::host_exclusions_from_entries(2, &[(0, 1, 1.0, scale_coul)]);
            let ff = ForceField::new(
                &reg,
                gpu,
                2,
                &sim_box,
                &particle_types,
                &pair_interactions,
                &[],
                &[],
                &[],
                spme.as_ref(),
                &charges,
                &BondList::empty(2),
                &AngleList::empty(0),
                &DihedralList::empty(0),
                &exclusions,
                &NeighborListConfig::AllPairs,
            )
            .expect("build pair-correction ForceField");
            (ff, sim_box, vec![1.0 as Real; 2], charges.clone(), type_indices.clone())
        });
        let mut f = Self::base(label, PotentialShape::Pair, samples, geometry, build);
        f.cutoff = cutoff;
        f.r_switch = cutoff;
        f.unbounded = true;
        f
    }

    pub fn pair(
        label: &'static str,
        register: impl Fn(&mut PotentialRegistry) + 'static,
        particle_types: Vec<ParticleTypeConfig>,
        pair_interactions: Vec<PairInteractionConfig>,
        charges: Vec<Real>,
        spme: Option<SpmeConfig>,
        cutoff: f64,
        r_switch: f64,
        samples: Vec<f64>,
    ) -> Self {
        let geometry: GeometryFn = Box::new(|r| vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]]);
        let type_indices = vec![0u32, if particle_types.len() > 1 { 1 } else { 0 }];
        let build: BuildFn = Box::new(move |gpu| {
            let mut reg = PotentialRegistry::new();
            register(&mut reg);
            let sim_box = SimulationBox::new(&gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0).unwrap();
            let ff = ForceField::new(
                &reg,
                gpu,
                2,
                &sim_box,
                &particle_types,
                &pair_interactions,
                &[],
                &[],
                &[],
                spme.as_ref(),
                &charges,
                &BondList::empty(2),
                &AngleList::empty(0),
                &DihedralList::empty(0),
                &ExclusionList::empty(2),
                &NeighborListConfig::AllPairs,
            )
            .expect("build pair ForceField");
            (ff, sim_box, vec![1.0 as Real; 2], charges.clone(), type_indices.clone())
        });
        let mut f = Self::base(label, PotentialShape::Pair, samples, geometry, build);
        f.cutoff = cutoff;
        f.r_switch = r_switch;
        f
    }

    pub fn bond(
        label: &'static str,
        register: impl Fn(&mut PotentialRegistry) + 'static,
        bond_types: Vec<BondTypeConfig>,
        samples: Vec<f64>,
    ) -> Self {
        let geometry: GeometryFn = Box::new(|r| vec![[0.0, 0.0, 0.0], [r, 0.0, 0.0]]);
        let build: BuildFn = Box::new(move |gpu| {
            let mut reg = PotentialRegistry::new();
            register(&mut reg);
            let sim_box = SimulationBox::new(&gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0).unwrap();
            let bonds = single_bond_list(2);
            let ff = ForceField::new(
                &reg, gpu, 2, &sim_box, &[], &[], &bond_types, &[], &[], None, &[],
                &bonds, &AngleList::empty(0), &DihedralList::empty(0), &ExclusionList::empty(2),
                &NeighborListConfig::AllPairs,
            )
            .expect("build bond ForceField");
            (ff, sim_box, vec![1.0 as Real; 2], vec![0.0 as Real; 2], vec![0u32; 2])
        });
        Self::base(label, PotentialShape::Bond, samples, geometry, build)
    }

    pub fn angle(
        label: &'static str,
        register: impl Fn(&mut PotentialRegistry) + 'static,
        angle_types: Vec<AngleTypeConfig>,
        samples: Vec<f64>,
    ) -> Self {
        // Coordinate is the angle θ; j (index 1) is the central atom.
        let geometry: GeometryFn = Box::new(|theta| {
            vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [theta.cos(), theta.sin(), 0.0]]
        });
        let build: BuildFn = Box::new(move |gpu| {
            let mut reg = PotentialRegistry::new();
            register(&mut reg);
            let sim_box = SimulationBox::new(&gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0).unwrap();
            let angles = single_angle_list(3, 0, 1, 2);
            let ff = ForceField::new(
                &reg, gpu, 3, &sim_box, &[], &[], &[], &angle_types, &[], None, &[],
                &BondList::empty(3), &angles, &DihedralList::empty(0), &ExclusionList::empty(3),
                &NeighborListConfig::AllPairs,
            )
            .expect("build angle ForceField");
            (ff, sim_box, vec![1.0 as Real; 3], vec![0.0 as Real; 3], vec![0u32; 3])
        });
        Self::base(label, PotentialShape::Angle, samples, geometry, build)
    }

    pub fn dihedral(
        label: &'static str,
        register: impl Fn(&mut PotentialRegistry) + 'static,
        dihedral_types: Vec<DihedralTypeConfig>,
        samples: Vec<f64>,
    ) -> Self {
        // Coordinate is the torsion φ; atoms i,j,k,l are indices 0..3.
        let geometry: GeometryFn = Box::new(|phi| {
            vec![
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, phi.cos(), phi.sin()],
            ]
        });
        let build: BuildFn = Box::new(move |gpu| {
            let mut reg = PotentialRegistry::new();
            register(&mut reg);
            let sim_box = SimulationBox::new(&gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0).unwrap();
            let dihedrals = single_dihedral_list(4, 0, 1, 2, 3);
            let ff = ForceField::new(
                &reg, gpu, 4, &sim_box, &[], &[], &[], &[], &dihedral_types, None, &[],
                &BondList::empty(4), &AngleList::empty(0), &dihedrals, &ExclusionList::empty(4),
                &NeighborListConfig::AllPairs,
            )
            .expect("build dihedral ForceField");
            (ff, sim_box, vec![1.0 as Real; 4], vec![0.0 as Real; 4], vec![0u32; 4])
        });
        Self::base(label, PotentialShape::Dihedral, samples, geometry, build)
    }
}

// =====================================================================
// Topology construction helpers (single-interaction reduction maps)
// =====================================================================

fn single_bond_list(n: usize) -> BondList {
    let bonds = vec![Bond { atom_i: 0, atom_j: 1, bond_type_index: 0 }];
    let mut atom_bond_offsets = vec![0u32; n + 1];
    atom_bond_offsets[1] = 1;
    for i in 2..=n {
        atom_bond_offsets[i] = 2;
    }
    BondList {
        bonds,
        atom_bond_offsets,
        atom_bond_indices: vec![0u32, 1u32],
        particle_count: n,
    }
}

fn single_angle_list(n: usize, i: u32, j: u32, k: u32) -> AngleList {
    let angles = vec![Angle { atom_i: i, atom_j: j, atom_k: k, angle_type_index: 0 }];
    let mut counts = vec![0u32; n];
    counts[i as usize] += 1;
    counts[j as usize] += 1;
    counts[k as usize] += 1;
    let mut atom_angle_offsets = vec![0u32; n + 1];
    let mut running = 0u32;
    for a in 0..n {
        atom_angle_offsets[a] = running;
        running += counts[a];
    }
    atom_angle_offsets[n] = running;
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    per_atom[i as usize].push(0);
    per_atom[j as usize].push(1);
    per_atom[k as usize].push(2);
    let mut atom_angle_indices = Vec::new();
    for a in 0..n {
        for &idx in &per_atom[a] {
            atom_angle_indices.push(idx);
        }
    }
    AngleList { angles, atom_angle_offsets, atom_angle_indices, particle_count: n }
}

fn single_dihedral_list(n: usize, i: u32, j: u32, k: u32, l: u32) -> DihedralList {
    let dihedrals = vec![Dihedral { atom_i: i, atom_j: j, atom_k: k, atom_l: l, dihedral_type_index: 0 }];
    let mut counts = vec![0u32; n];
    for &a in &[i, j, k, l] {
        counts[a as usize] += 1;
    }
    let mut atom_dihedral_offsets = vec![0u32; n + 1];
    let mut running = 0u32;
    for a in 0..n {
        atom_dihedral_offsets[a] = running;
        running += counts[a];
    }
    atom_dihedral_offsets[n] = running;
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    per_atom[i as usize].push(0);
    per_atom[j as usize].push(1);
    per_atom[k as usize].push(2);
    per_atom[l as usize].push(3);
    let mut atom_dihedral_indices = Vec::new();
    for a in 0..n {
        for &idx in &per_atom[a] {
            atom_dihedral_indices.push(idx);
        }
    }
    DihedralList { dihedrals, atom_dihedral_offsets, atom_dihedral_indices, particle_count: n }
}

// =====================================================================
// The invariant checks (generic over an Evaluator) — rq-b9fde79c
// =====================================================================

fn mean(pos: &[[f64; 3]]) -> [f64; 3] {
    let n = pos.len() as f64;
    let mut c = [0.0; 3];
    for p in pos {
        for d in 0..3 {
            c[d] += p[d];
        }
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

fn scale_about(pos: &[[f64; 3]], center: &[f64; 3], lambda: f64) -> Vec<[f64; 3]> {
    pos.iter()
        .map(|p| [
            center[0] + lambda * (p[0] - center[0]),
            center[1] + lambda * (p[1] - center[1]),
            center[2] + lambda * (p[2] - center[2]),
        ])
        .collect()
}

/// Run every applicable invariant for one fixture-shaped potential against
/// `ev`. Panics (naming the invariant) on the first failure. Shared by the
/// GPU path and the CPU-closure negative tests.
pub fn check_consistency(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    check_force_energy(fixture, ev);
    check_newton(fixture, ev);
    check_virial(fixture, ev);
    // An `Unbounded` fragment has no cutoff to be continuous across, and is
    // required NOT to vanish beyond one: the SPME excluded-pair correction
    // offsets a cutoff-free reciprocal-space mesh sum, so a non-zero force at
    // large r is the specified behaviour, not a defect.
    if fixture.shape == PotentialShape::Pair && !fixture.unbounded {
        check_pair_continuity(fixture, ev);
    }
    check_reference_points(fixture, ev);
}

/// Force–energy finite-difference consistency: `−ΔU/2h ≈ F` per atom, per
/// axis. Panics naming the force-energy invariant. rq-b9fde79c
pub fn check_force_energy(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    let label = fixture.label;
    let h = fixture.fd_step;
    for &coord in &fixture.samples {
        let pos = (fixture.geometry)(coord);
        let base = ev.eval(&pos);
        for a in 0..pos.len() {
            for d in 0..3 {
                let mut up = pos.clone();
                up[a][d] += h;
                let mut dn = pos.clone();
                dn[a][d] -= h;
                let fd = -(ev.eval(&up).energy - ev.eval(&dn).energy) / (2.0 * h);
                let f = base.forces[a][d];
                assert!(
                    fixture.fd_tol.close(f, fd),
                    "[{label}] force-energy FD mismatch at coord {coord}, atom {a}, axis {d}: F={f}, -dU/dx={fd}"
                );
            }
        }
    }
}

/// Newton's third law: the per-atom forces sum to zero. Panics naming the
/// Newton invariant. rq-b9fde79c
pub fn check_newton(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    let label = fixture.label;
    for &coord in &fixture.samples {
        let base = ev.eval(&(fixture.geometry)(coord));
        let mut sum = [0.0f64; 3];
        let mut max_f = 0.0f64;
        for f in &base.forces {
            for d in 0..3 {
                sum[d] += f[d];
                max_f = max_f.max(f[d].abs());
            }
        }
        for d in 0..3 {
            assert!(
                sum[d].abs() <= fixture.newton_tol.abs + fixture.newton_tol.rel * max_f,
                "[{label}] Newton's third law violated at coord {coord}, axis {d}: sum F={}",
                sum[d]
            );
        }
    }
}

/// Virial–force consistency: reported scalar virial ≈ `−dU/d(ln λ)` under
/// isotropic scaling. Panics naming the virial invariant. rq-b9fde79c
pub fn check_virial(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    let label = fixture.label;
    for &coord in &fixture.samples {
        let pos = (fixture.geometry)(coord);
        let base = ev.eval(&pos);
        let c = mean(&pos);
        let eps = fixture.scale_eps;
        let e_up = ev.eval(&scale_about(&pos, &c, 1.0 + eps)).energy;
        let e_dn = ev.eval(&scale_about(&pos, &c, 1.0 - eps)).energy;
        let w_fd = -(e_up - e_dn) / (2.0 * eps);
        assert!(
            fixture.virial_tol.close(w_fd, base.virial),
            "[{label}] virial mismatch at coord {coord}: reported W={}, -dU/dlnλ={w_fd}",
            base.virial
        );
    }
}

/// Analytic reference points: expected energy and/or coordinate-conjugate
/// force at declared coordinate values. Panics naming the reference-point
/// invariant. rq-b9fde79c
pub fn check_reference_points(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    let label = fixture.label;
    for rp in &fixture.reference_points {
        let e = ev.eval(&(fixture.geometry)(rp.coordinate)).energy;
        if let Some(exp) = rp.energy {
            assert!(
                rp.tol.close(exp, e),
                "[{label}] reference-point energy mismatch at coord {}: expected {exp}, got {e}",
                rp.coordinate
            );
        }
        if let Some(expf) = rp.coord_force {
            let hc = fixture.coord_fd_step;
            let eu = ev.eval(&(fixture.geometry)(rp.coordinate + hc)).energy;
            let ed = ev.eval(&(fixture.geometry)(rp.coordinate - hc)).energy;
            let cf = -(eu - ed) / (2.0 * hc);
            assert!(
                rp.tol.close(expf, cf),
                "[{label}] reference-point force mismatch at coord {}: expected {expf}, got {cf}",
                rp.coordinate
            );
        }
    }
}

/// Switch/cutoff continuity (pair only). Panics naming the continuity
/// invariant. rq-b9fde79c
pub fn check_pair_continuity(fixture: &ConsistencyFixture, ev: &mut dyn Evaluator) {
    let label = fixture.label;
    let cutoff = fixture.cutoff;
    // Force (and, for a switched potential, energy) must be exactly zero
    // beyond the cutoff.
    for m in [1.02_f64, 1.1, 1.3] {
        let e = ev.eval(&(fixture.geometry)(cutoff * m));
        let fmag = e.forces[0].iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        assert!(
            fmag <= fixture.continuity_tol.abs,
            "[{label}] continuity: force nonzero beyond cutoff at r={}: |F|={fmag}",
            cutoff * m
        );
    }

    // Smoothness only applies when a genuine switching region exists.
    if fixture.r_switch >= fixture.cutoff {
        return;
    }
    // Jump detector across [0.98·r_switch, cutoff]: a discontinuity produces a
    // single |ΔE| far larger than the median step.
    let lo = 0.98 * fixture.r_switch;
    let n = 64usize;
    let mut es = Vec::with_capacity(n);
    for i in 0..n {
        let r = lo + (cutoff - lo) * (i as f64) / ((n - 1) as f64);
        es.push(ev.eval(&(fixture.geometry)(r)).energy);
    }
    let mut diffs: Vec<f64> = es.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
    let max_d = diffs.iter().cloned().fold(0.0_f64, f64::max);
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = diffs[diffs.len() / 2];
    assert!(
        max_d <= fixture.continuity_jump_ratio * median + fixture.continuity_tol.abs,
        "[{label}] continuity: energy discontinuity detected across the switching region \
         (max step {max_d}, median {median})"
    );
}

/// Run every applicable invariant for `fixture` against the real GPU force
/// pipeline. rq-9cd51562
pub fn assert_potential_consistent(fixture: &ConsistencyFixture, gpu: &GpuContext) {
    let mut ev = fixture.gpu_evaluator(gpu);
    check_consistency(fixture, &mut ev);
}

// =====================================================================
// Built-in fixtures and coverage — rq-da0e909a rq-6dbbbccc
// =====================================================================

/// One fixture per built-in fragment-composed potential. rq-da0e909a
pub fn builtin_consistency_fixtures() -> Vec<ConsistencyFixture> {
    let two_pow_1_6 = 2.0_f64.powf(1.0 / 6.0);
    let mut out = Vec::new();

    // Lennard-Jones: σ=1, ε=1, cutoff 4, switch at 3.5. rq-2d06b5d2
    out.push(
        ConsistencyFixture::pair(
            "lennard_jones",
            |reg| reg.register(Box::new(LennardJonesBuilder)),
            vec![lj_type("Ar", 0.0)],
            vec![PairInteractionConfig::lennard_jones(("Ar", "Ar"), 1.0, 1.0, 4.0, Some(3.5))],
            vec![0.0 as Real; 2],
            None,
            4.0,
            3.5,
            vec![0.95, 1.05, 1.2, 1.5, 2.0, 3.6],
        )
        .with_reference_points(vec![
            ReferencePoint { coordinate: 1.0, energy: Some(0.0), coord_force: None, tol: Tolerance::new(2.0e-2, 2.0e-2) },
            ReferencePoint { coordinate: two_pow_1_6, energy: Some(-1.0), coord_force: Some(0.0), tol: Tolerance::new(2.0e-2, 2.0e-2) },
        ]),
    );

    // SPME real-space: two opposite unit charges. Hard-truncated screened
    // Coulomb (no switch), so continuity smoothness is not applied.
    out.push(
        ConsistencyFixture::pair(
            "spme_real",
            |reg| reg.register(Box::new(SpmeRealBuilder)),
            vec![lj_type("P", 1.0), lj_type("M", -1.0)],
            vec![],
            vec![1.0 as Real, -1.0],
            Some(SpmeConfig { alpha: 0.6, r_cut_real: 5.0, grid: [16, 16, 16], spline_order: 5 }),
            5.0,
            5.0, // r_switch == cutoff → smoothness check skipped
            vec![1.0, 1.5, 2.0, 3.0, 4.0],
        ),
    );

    // SPME excluded-pair correction. The reciprocal mesh carries erf(a*r)/r for
    // every pair including the excluded ones; this fragment removes the unwanted
    // share. It is a CorrectionOnly, Unbounded fragment, so the harness measures
    // `(scale_coul - 1) x evaluate` and the continuity check does not apply.
    //
    // alpha = 0.6, k_C = 1 (atomic units), charges (+1, -1) so qq = -1, and
    // scale_coul = 0 (a full exclusion). The measured energy is therefore
    //     (0 - 1) * 1 * (-1) * erf(0.6 r) / r  =  +erf(0.6 r) / r
    // The reference point pins the closed form, which no finite-difference check
    // could: a wrong U with its matching wrong F still satisfies F = -grad U.
    out.push(
        ConsistencyFixture::pair_correction(
            "spme_exclusion",
            |reg| reg.register(Box::new(SpmeExclusionBuilder)),
            vec![lj_type("P", 1.0), lj_type("M", -1.0)],
            // Zero-epsilon LJ: supplies the neighbour list's cutoff and
            // contributes exactly nothing. Every unordered type pair must be
            // declared (the loader's config invariant).
            vec![
                PairInteractionConfig::lennard_jones(("P", "P"), 1.0, 0.0, 5.0, Some(5.0)),
                PairInteractionConfig::lennard_jones(("M", "M"), 1.0, 0.0, 5.0, Some(5.0)),
                PairInteractionConfig::lennard_jones(("P", "M"), 1.0, 0.0, 5.0, Some(5.0)),
            ],
            vec![1.0 as Real, -1.0],
            Some(SpmeConfig { alpha: 0.6, r_cut_real: 5.0, grid: [16, 16, 16], spline_order: 5 }),
            0.0, // scale_coul: a full exclusion
            5.0,
            vec![0.5, 1.0, 2.0, 3.0, 6.0], // 6.0 is BEYOND the cutoff: still corrected
        )
        .with_reference_points(vec![
            ReferencePoint {
                coordinate: 1.0,
                energy: Some(REF_U_1),
                coord_force: None,
                tol: Tolerance { rel: 2e-2, abs: 2e-3 },
            },
            ReferencePoint {
                coordinate: 2.0,
                energy: Some(REF_U_2),
                coord_force: None,
                tol: Tolerance { rel: 2e-2, abs: 2e-3 },
            },
        ]),
    );

    // Morse bond: De=2, a=1.5, re=1. rq-f417a0f5
    out.push(
        ConsistencyFixture::bond(
            "morse_bonded",
            |reg| reg.register(Box::new(MorseBondedBuilder)),
            vec![BondTypeConfig::morse("MM", 2.0, 1.5, 1.0)],
            vec![0.8, 1.0, 1.3, 1.7],
        )
        .with_reference_points(vec![ReferencePoint {
            coordinate: 1.0,
            energy: Some(0.0),
            coord_force: Some(0.0),
            tol: Tolerance::new(2.0e-2, 2.0e-2),
        }]),
    );

    // Harmonic bond: k=100, r0=1.
    out.push(
        ConsistencyFixture::bond(
            "harmonic_bond",
            |reg| reg.register(Box::new(HarmonicBondBuilder)),
            vec![BondTypeConfig::harmonic("HH", 100.0, 1.0)],
            vec![0.85, 1.0, 1.15, 1.3],
        )
        .with_reference_points(vec![ReferencePoint {
            coordinate: 1.0,
            energy: Some(0.0),
            coord_force: Some(0.0),
            tol: Tolerance::new(2.0e-2, 5.0e-2),
        }]),
    );

    // Harmonic angle: k_theta=50, theta0=1.9 rad. rq-e4ad8b2c
    out.push(
        ConsistencyFixture::angle(
            "harmonic_angle",
            |reg| reg.register(Box::new(HarmonicAngleBuilder)),
            vec![AngleTypeConfig::harmonic("AAA", 50.0, 1.9)],
            vec![1.6, 1.9, 2.2],
        )
        .with_reference_points(vec![ReferencePoint {
            coordinate: 1.9,
            energy: Some(0.0),
            coord_force: None,
            tol: Tolerance::new(2.0e-2, 5.0e-2),
        }]),
    );

    // Periodic dihedral: k_phi=1, n=2, phi0=0. rq-9db48a6d
    out.push(
        ConsistencyFixture::dihedral(
            "periodic_dihedral",
            |reg| reg.register(Box::new(PeriodicDihedralBuilder)),
            vec![DihedralTypeConfig::periodic("DDDD", 1.0, 2, 0.0)],
            vec![0.5, 1.2, 2.0, 2.8],
        ),
    );

    out
}

fn lj_type(name: &str, charge: f64) -> ParticleTypeConfig {
    ParticleTypeConfig { name: name.to_string(), mass: 1.0, sigma: None, epsilon: None, charge }
}

/// The set of fast-class fragment-composed slot labels producible by the
/// built-in registry (every slot whose `jit_participant()` is `Some`).
pub fn builtin_fragment_labels(gpu: &GpuContext) -> HashSet<String> {
    let sim_box = SimulationBox::new(&gpu.device, HARNESS_BOX, HARNESS_BOX, HARNESS_BOX, 0.0, 0.0, 0.0).unwrap();
    // A system activating every fragment-composed built-in at once.
    let particle_types = vec![lj_type("P", 1.0), lj_type("M", -1.0)];
    let pair_interactions = vec![
        PairInteractionConfig::lennard_jones(("P", "P"), 1.0, 1.0, 4.0, Some(3.5)),
        PairInteractionConfig::lennard_jones(("P", "M"), 1.0, 1.0, 4.0, Some(3.5)),
        PairInteractionConfig::lennard_jones(("M", "M"), 1.0, 1.0, 4.0, Some(3.5)),
    ];
    let bond_types = vec![BondTypeConfig::morse("MO", 2.0, 1.5, 1.0), BondTypeConfig::harmonic("HA", 100.0, 1.0)];
    let angle_types = vec![AngleTypeConfig::harmonic("AN", 50.0, 1.9)];
    let dihedral_types = vec![DihedralTypeConfig::periodic("DI", 1.0, 2, 0.0)];
    let n = 6;
    let bonds = single_bond_list_typed(n, &[(0, 1, 0), (2, 3, 1)]);
    let angles = single_angle_list(n, 2, 3, 4);
    let dihedrals = single_dihedral_list(n, 2, 3, 4, 5);
    let charges = vec![1.0 as Real, -1.0, 1.0, -1.0, 1.0, -1.0];
    let ff = ForceField::new(
        &PotentialRegistry::with_builtins(),
        gpu,
        n,
        &sim_box,
        &particle_types,
        &pair_interactions,
        &bond_types,
        &angle_types,
        &dihedral_types,
        Some(&SpmeConfig { alpha: 0.6, r_cut_real: 5.0, grid: [16, 16, 16], spline_order: 5 }),
        &charges,
        &bonds,
        &angles,
        &dihedrals,
        &ExclusionList::empty(n),
        &NeighborListConfig::AllPairs,
    )
    .expect("build all-fragment ForceField");
    ff.slots
        .iter()
        .filter(|s| s.jit_participant().is_some())
        .map(|s| s.label().to_string())
        .collect()
}

fn single_bond_list_typed(n: usize, bonds_in: &[(u32, u32, u32)]) -> BondList {
    let bonds: Vec<Bond> = bonds_in
        .iter()
        .map(|&(i, j, t)| Bond { atom_i: i, atom_j: j, bond_type_index: t })
        .collect();
    let mut atom_bond_offsets = vec![0u32; n + 1];
    for b in &bonds {
        atom_bond_offsets[b.atom_i as usize + 1] += 1;
        atom_bond_offsets[b.atom_j as usize + 1] += 1;
    }
    for i in 1..=n {
        atom_bond_offsets[i] += atom_bond_offsets[i - 1];
    }
    let mut atom_bond_indices = vec![0u32; bonds.len() * 2];
    let mut cursor: Vec<u32> = atom_bond_offsets[..n].to_vec();
    for (k, b) in bonds.iter().enumerate() {
        atom_bond_indices[cursor[b.atom_i as usize] as usize] = (2 * k) as u32;
        cursor[b.atom_i as usize] += 1;
        atom_bond_indices[cursor[b.atom_j as usize] as usize] = (2 * k + 1) as u32;
        cursor[b.atom_j as usize] += 1;
    }
    BondList { bonds, atom_bond_offsets, atom_bond_indices, particle_count: n }
}

/// Panic if any built-in fragment label is not covered by `fixture_labels`.
/// rq-6dbbbccc
pub fn assert_fixture_coverage(fixture_labels: &HashSet<String>, gpu: &GpuContext) {
    for label in builtin_fragment_labels(gpu) {
        assert!(
            fixture_labels.contains(&label),
            "coverage: built-in fragment potential '{label}' has no consistency fixture"
        );
    }
}

/// Run every built-in fixture and assert coverage of every built-in fragment
/// potential. rq-2e64e2c1
pub fn assert_all_builtin_potentials_consistent(gpu: &GpuContext) {
    let fixtures = builtin_consistency_fixtures();
    let labels: HashSet<String> = fixtures.iter().map(|f| f.label.to_string()).collect();
    assert_fixture_coverage(&labels, gpu);
    for fixture in &fixtures {
        assert_potential_consistent(fixture, gpu);
    }
}
