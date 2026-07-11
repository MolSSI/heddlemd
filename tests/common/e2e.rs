//! End-to-end test harness. Constructs runnable input systems, drives
//! them through `run_simulation`, and asserts physical / compositional
//! invariants on the results. See `rqm/e2e-testing.md`.
//!
//! Included from `tests/common/mod.rs`; `#![allow(dead_code)]` on the
//! parent module covers helpers a given test binary does not use.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use heddle_md::runner::{run_simulation, PhaseSummary, RunSummary};

// A process-wide counter so concurrently-running cases never collide even
// within the same nanosecond.
static CASE_NONCE: AtomicU64 = AtomicU64::new(0);

// rq-a5e72a70 — an isolated working directory for one e2e test.
pub struct Case {
    dir: PathBuf,
}

impl Case {
    // rq-a5e72a70
    pub fn new(name: &str) -> Case {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let nonce = CASE_NONCE.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("heddle_e2e_{name}_{pid}_{nanos}_{nonce}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Case { dir }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config_path(&self) -> PathBuf {
        self.dir.join("sim.in.toml")
    }
}

impl Drop for Case {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[derive(Clone, Copy)]
enum Preset {
    // Simple-cubic argon lattice (ordered).
    ArgonLattice { side: usize, spacing: f64 },
    // Argon lattice with a deterministic per-particle displacement.
    DisorderedLj { side: usize, spacing: f64 },
    // Two SPC/E water molecules with rigid-geometry constraint metadata.
    SpceWater,
}

// rq-a5e72a70 — builder that writes a matched extended-XYZ initial-state
// file and TOML configuration into a `Case`.
#[derive(Clone)]
pub struct SystemBuilder {
    preset: Preset,
    integrator: String,
    lossless: bool,
    thermostat: Option<String>,
    barostat: Option<String>,
    barostat_is_per_step: bool,
    constraint_kind: Option<String>,
    dt: f64,
    n_steps: u64,
    log_every: u64,
    trajectory_every: u64,
    seed: u64,
    temperature: f64,
}

impl SystemBuilder {
    fn base(preset: Preset, dt: f64, temperature: f64) -> Self {
        SystemBuilder {
            preset,
            integrator: "velocity-verlet".to_string(),
            lossless: false,
            thermostat: None,
            barostat: None,
            barostat_is_per_step: false,
            constraint_kind: None,
            dt,
            n_steps: 100,
            log_every: 1,
            trajectory_every: 1,
            seed: 1,
            temperature,
        }
    }

    // rq-a5e72a70
    pub fn argon_lattice(side: usize) -> Self {
        // Near-equilibrium spacing, a small step, and a modest temperature
        // keep the lattice a stable solid so an NVE run conserves energy.
        Self::base(Preset::ArgonLattice { side, spacing: 4.3e-10 }, 1.0e-15, 40.0)
    }

    // rq-a5e72a70
    pub fn disordered_lj_liquid(side: usize, spacing: f64) -> Self {
        Self::base(Preset::DisorderedLj { side, spacing }, 2.0e-15, 120.0)
    }

    // rq-a5e72a70
    pub fn spce_water() -> Self {
        Self::base(Preset::SpceWater, 1.0e-15, 300.0)
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    pub fn n_steps(mut self, n: u64) -> Self {
        self.n_steps = n;
        self
    }

    pub fn log_every(mut self, k: u64) -> Self {
        self.log_every = k;
        self
    }

    pub fn trajectory_every(mut self, k: u64) -> Self {
        self.trajectory_every = k;
        self
    }

    // rq-a5e72a70 — install a thermostat slot at `target_temperature` (K).
    pub fn thermostat(mut self, kind: &str, target_temperature: f64) -> Self {
        let seed = self.seed;
        let block = match kind {
            "csvr" => format!(
                "[phase.thermostat]\nkind = \"csvr\"\ntemperature = {target_temperature}\ntau = 1.0e-13\nseed = {seed}\n"
            ),
            "andersen" => format!(
                "[phase.thermostat]\nkind = \"andersen\"\ntemperature = {target_temperature}\ncollision_rate = 1.0e12\nseed = {seed}\n"
            ),
            "berendsen" => format!(
                "[phase.thermostat]\nkind = \"berendsen\"\ntemperature = {target_temperature}\ntau = 1.0e-13\n"
            ),
            other => panic!("unsupported thermostat kind in harness: {other}"),
        };
        self.thermostat = Some(block);
        self
    }

    // rq-a5e72a70 — install a barostat slot at `target_pressure` (Pa).
    pub fn barostat(mut self, kind: &str, target_pressure: f64) -> Self {
        let seed = self.seed;
        let temperature = self.temperature;
        let (block, per_step) = match kind {
            "c-rescale" => (
                format!(
                    "[phase.barostat]\nkind = \"c-rescale\"\npressure = {target_pressure}\ntemperature = {temperature}\ntau = 1.0e-12\ncompressibility = 4.5e-10\nseed = {seed}\n"
                ),
                true,
            ),
            "berendsen" => (
                format!(
                    "[phase.barostat]\nkind = \"berendsen\"\npressure = {target_pressure}\ntau = 1.0e-12\ncompressibility = 4.5e-10\n"
                ),
                true,
            ),
            other => panic!("unsupported barostat kind in harness: {other}"),
        };
        self.barostat = Some(block);
        self.barostat_is_per_step = per_step;
        self
    }

    // rq-a5e72a70 — install a constraint slot over the preset's metadata.
    pub fn constraints(mut self, kind: &str) -> Self {
        self.constraint_kind = Some(kind.to_string());
        self
    }

    // rq-a5e72a70 — write `sim.in.xyz` (+ topology for water) and
    // `sim.in.toml` into the case, returning the config path.
    pub fn write(&self, case: &Case) -> PathBuf {
        match self.preset {
            Preset::ArgonLattice { side, spacing } => self.write_lj(case, side, spacing, false),
            Preset::DisorderedLj { side, spacing } => self.write_lj(case, side, spacing, true),
            Preset::SpceWater => self.write_water(case),
        }
    }

    fn phase_extra(&self) -> String {
        let mut s = String::new();
        if let Some(t) = &self.thermostat {
            s.push_str(t);
            s.push('\n');
        }
        if let Some(b) = &self.barostat {
            s.push_str(b);
            s.push('\n');
        }
        s
    }

    fn write_lj(&self, case: &Case, side: usize, spacing: f64, disordered: bool) -> PathBuf {
        let n = side * side * side;
        let l = side as f64 * spacing;
        let c = (side as f64 - 1.0) / 2.0;
        // Deterministic LCG jitter so the *input* is byte-identical every
        // call while the neighbour structure is irregular.
        let mut lcg: u64 = 0x1234_5678;
        let mut jitter = |on: bool| {
            if !on {
                return 0.0;
            }
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((lcg >> 33) as f64 / (1u64 << 31) as f64) - 0.5) * 0.6 * spacing
        };
        let mut body = format!("{n}\n");
        body.push_str(&format!(
            "Lattice=\"{l:.6e} 0 0 0 {l:.6e} 0 0 0 {l:.6e}\" Properties=species:S:1:pos:R:3\n"
        ));
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    let px = (i as f64 - c) * spacing + jitter(disordered);
                    let py = (j as f64 - c) * spacing + jitter(disordered);
                    let pz = (k as f64 - c) * spacing + jitter(disordered);
                    body.push_str(&format!("Ar {px:.9e} {py:.9e} {pz:.9e}\n"));
                }
            }
        }
        std::fs::write(case.dir().join("sim.in.xyz"), body).unwrap();

        let SystemBuilder { dt, n_steps, log_every, trajectory_every, seed, temperature, .. } = *self;
        let extra = self.phase_extra();
        let cfg = format!(
            r#"schema_version = 1
units = "si"
init = "sim.in.xyz"

[simulation]
cuda_graphs_disable = true
seed = {seed}
temperature = {temperature}

[[phase]]
name = "run"
n_steps = {n_steps}
dt = {dt:e}

[phase.integrator]
kind = "{integrator}"
lossless = {lossless}

[phase.output]
trajectory_every = {trajectory_every}
log_every = {log_every}

{extra}
[[particle_types]]
name = "Ar"
mass = 6.6335e-26

[[pair_interactions]]
between = ["Ar", "Ar"]
kind = "lennard-jones"
sigma = 3.40e-10
epsilon = 1.65e-21
cutoff = 7.0e-10
r_switch = 6.0e-10

[neighbor_list]
mode = "all-pairs"
"#,
            integrator = self.integrator,
            lossless = self.lossless,
        );
        let path = case.config_path();
        std::fs::write(&path, cfg).unwrap();
        path
    }

    fn write_water(&self, case: &Case) -> PathBuf {
        // Two SPC/E water molecules. Geometry lifted from the SETTLE e2e
        // fixture (O–H = 1.0 Å, H–H = 1.633 Å).
        let xyz = "6\n\
            Lattice=\"5.0e-9 0 0 0 5.0e-9 0 0 0 5.0e-9\" Properties=species:S:1:pos:R:3\n\
            O  -2.500000000e-10 0.000000000e0 0.000000000e0\n\
            H  -1.500000000e-10 0.000000000e0 0.000000000e0\n\
            H  -2.833800000e-10 9.426400000e-11 0.000000000e0\n\
            O   2.500000000e-10 0.000000000e0 0.000000000e0\n\
            H   3.500000000e-10 0.000000000e0 0.000000000e0\n\
            H   2.166200000e-10 9.426400000e-11 0.000000000e0\n";
        std::fs::write(case.dir().join("sim.in.xyz"), xyz).unwrap();
        std::fs::write(
            case.dir().join("sim.in.topology"),
            "[constraints]\n0 1 2 SPCE\n3 4 5 SPCE\n",
        )
        .unwrap();

        let constraint_kind = self.constraint_kind.as_deref().unwrap_or("settle");
        let constraint_block = format!(
            "[[constraint_types]]\nname = \"SPCE\"\nkind = \"{constraint_kind}\"\nd_OH = 1.0e-10\nd_HH = 1.633e-10\n"
        );
        let SystemBuilder { dt, n_steps, log_every, trajectory_every, seed, temperature, .. } = *self;
        let extra = self.phase_extra();
        let cfg = format!(
            r#"schema_version = 1
units = "si"
init = "sim.in.xyz"
topology = "sim.in.topology"

[simulation]
cuda_graphs_disable = true
seed = {seed}
temperature = {temperature}

[[phase]]
name = "run"
n_steps = {n_steps}
dt = {dt:e}

[phase.integrator]
kind = "{integrator}"
lossless = {lossless}

[phase.output]
trajectory_every = {trajectory_every}
log_every = {log_every}

{extra}
[[particle_types]]
name = "O"
mass = 2.6566e-26
charge = 0.0

[[particle_types]]
name = "H"
mass = 1.6735e-27
charge = 0.0

[[pair_interactions]]
between = ["O", "O"]
kind = "lennard-jones"
sigma = 3.166e-10
epsilon = 1.080e-21
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "H"]
kind = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

[[pair_interactions]]
between = ["H", "O"]
kind = "lennard-jones"
sigma = 1.0e-10
epsilon = 1.0e-30
cutoff = 1.0e-9
r_switch = 1.0e-9

{constraint_block}
[neighbor_list]
mode = "all-pairs"
"#,
            integrator = self.integrator,
            lossless = self.lossless,
        );
        let path = case.config_path();
        std::fs::write(&path, cfg).unwrap();
        path
    }

    fn traj_path(&self, case: &Case) -> PathBuf {
        case.dir().join("sim.out.run.xyz")
    }
}

// rq-e9ed7da2 — run one system to completion and return its summary.
pub fn run_case(config_path: &Path) -> RunSummary {
    run_simulation(config_path)
        .unwrap_or_else(|e| panic!("run_simulation({}) failed: {e:?}", config_path.display()))
}

// rq-e9ed7da2 — run the same system `n_runs` times into fresh cases and
// assert the trajectory and log output files are byte-identical.
pub fn assert_runs_reproducible(builder: &SystemBuilder, n_runs: usize) {
    assert!(n_runs >= 2, "need at least two runs to compare");
    let mut trajectories: Vec<Vec<u8>> = Vec::new();
    let mut logs: Vec<Vec<u8>> = Vec::new();
    // Hold every Case alive until all reads are done (Drop removes dirs).
    let mut cases = Vec::new();
    for r in 0..n_runs {
        let case = Case::new(&format!("repro{r}"));
        let cfg = builder.write(&case);
        run_case(&cfg);
        trajectories.push(std::fs::read(builder.traj_path(&case)).unwrap());
        let log = case.dir().join("sim.out.run.csv");
        if log.exists() {
            logs.push(std::fs::read(log).unwrap());
        }
        cases.push(case);
    }
    for r in 1..n_runs {
        assert!(
            trajectories[0] == trajectories[r],
            "run {r} trajectory differs from run 0 — pipeline is not run-to-run deterministic"
        );
    }
    for r in 1..logs.len() {
        assert!(logs[0] == logs[r], "run {r} log differs from run 0");
    }
}

// rq-e9ed7da2 — parse the final frame of a trajectory into
// (positions, velocities) in file units.
pub fn read_last_frame(trajectory_path: &Path) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let text = std::fs::read_to_string(trajectory_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let (mut pos, mut vel) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < lines.len() {
        let n: usize = lines[i].trim().parse().unwrap();
        let mut p = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        for a in 0..n {
            let cols: Vec<&str> = lines[i + 2 + a].split_whitespace().collect();
            p.push([
                cols[1].parse().unwrap(),
                cols[2].parse().unwrap(),
                cols[3].parse().unwrap(),
            ]);
            v.push([
                cols[4].parse().unwrap(),
                cols[5].parse().unwrap(),
                cols[6].parse().unwrap(),
            ]);
        }
        pos = p;
        vel = v;
        i += 2 + n;
    }
    (pos, vel)
}

// rq-e9ed7da2 — assert the SPC/E water frame lies on the position and
// velocity constraint manifolds (every molecule; two molecules of 3).
pub fn assert_water_on_manifold(pos: &[[f64; 3]], vel: &[[f64; 3]], rel_tol: f64) {
    const D_OH: f64 = 1.0e-10;
    const D_HH: f64 = 1.633e-10;
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| dot(a, a).sqrt();
    let vmag = vel
        .iter()
        .flat_map(|v| v.iter().map(|c| c.abs()))
        .fold(0.0_f64, f64::max);
    for base in [0usize, 3usize] {
        for (i, j, r0) in [
            (base, base + 1, D_OH),
            (base, base + 2, D_OH),
            (base + 1, base + 2, D_HH),
        ] {
            let dr = sub(pos[i], pos[j]);
            let len = norm(dr);
            assert!(
                (len - r0).abs() <= rel_tol * r0,
                "position manifold: bond {i}-{j} length {len:e} != r0 {r0:e}"
            );
            let d = dot(dr, sub(vel[i], vel[j]));
            let scale = (len * vmag).max(1e-30);
            assert!(
                d.abs() <= 1e-4 * scale,
                "velocity manifold: bond {i}-{j} (r·v) = {d:e}, scale {scale:e}"
            );
        }
    }
}

// rq-e9ed7da2 — assert the phase's total-energy drift slope is bounded.
pub fn assert_energy_drift_bounded(phase: &PhaseSummary, max_abs_slope: f64) {
    let s = &phase.physics;
    assert!(s.len() >= 3, "need several physics samples for a drift fit");
    let n = s.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for p in s {
        let (x, y) = (p.time, p.total_energy);
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    let slope = if denom.abs() > 0.0 { (n * sxy - sx * sy) / denom } else { 0.0 };
    assert!(
        slope.abs() <= max_abs_slope,
        "energy drift slope {slope:e} exceeds bound {max_abs_slope:e} \
         (E0={:e}, Elast={:e})",
        s.first().unwrap().total_energy,
        s.last().unwrap().total_energy
    );
}

// rq-e9ed7da2 — assert the mean of a field over post-equilibration
// samples is within `rel_tol` of `target`.
fn assert_mean_near(
    phase: &PhaseSummary,
    field: impl Fn(&heddle_md::runner::PhysicsSample) -> f64,
    target: f64,
    rel_tol: f64,
    what: &str,
) {
    let s = &phase.physics;
    assert!(!s.is_empty(), "no physics samples");
    let start = s.len() / 2; // drop the first half as equilibration
    let tail = &s[start..];
    let mean = tail.iter().map(|p| field(p)).sum::<f64>() / tail.len() as f64;
    assert!(
        (mean - target).abs() <= rel_tol * target.abs().max(f64::MIN_POSITIVE),
        "{what}: mean {mean:e} not within {rel_tol} of target {target:e}"
    );
}

pub fn assert_mean_temperature_near(phase: &PhaseSummary, target: f64, rel_tol: f64) {
    assert_mean_near(phase, |p| p.temperature, target, rel_tol, "temperature");
}

pub fn assert_mean_pressure_near(phase: &PhaseSummary, target: f64, rel_tol: f64) {
    assert_mean_near(phase, |p| p.pressure, target, rel_tol, "pressure");
}
