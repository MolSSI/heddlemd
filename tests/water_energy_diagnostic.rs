//! Energy-drift diagnostics for the SPC water-256 pipeline. These
//! tests are `#[ignore]` by default because they take several seconds
//! and write step-by-step totals to stdout; they're intended for
//! ad-hoc troubleshooting after any change to the bonded force
//! kernels or the SPME real-/reciprocal-space paths.
//!
//! Run with:
//!   cargo test --release --test water_energy_diagnostic -- --ignored --nocapture

use heddle_md::Registries;
use heddle_md::SimulationSetup;
use heddle_md::forces::AggregateLevel;
use heddle_md::gpu::{
    compute_kinetic_energy, compute_total_potential_energy, compute_total_virial,
};
use heddle_md::integrator::IntegratorStepExt;
use heddle_md::io::config::PhaseKind;
use heddle_md::precision::Real;
use heddle_md::timings::Timings;

struct Sample {
    t: f64,
    total: f64,
}

fn run_window(
    config: &str,
    dt_scale: f64,
    log_label: &str,
    target_time: f64,
    log_every_t: f64,
) -> Vec<Sample> {
    let config_path = std::path::PathBuf::from(config);
    let registries = Registries::with_builtins();
    let mut setup =
        SimulationSetup::new(&config_path, registries.clone()).expect("setup");
    let phase = match &setup.config.phases[0] {
        PhaseKind::Md(p) => p.clone(),
        _ => panic!("expected an MD phase"),
    };
    let dt = (phase.dt * dt_scale) as Real;
    let n_constraints = setup.n_constraints as usize;
    let mut integrator = registries
        .integrators
        .build(
            &phase.integrator,
            &setup.gpu,
            setup.buffers.particle_count(),
            n_constraints,
        )
        .expect("integrator");
    let mut timings = Timings::new(&setup.gpu).expect("timings");
    let mut pe_scratch = setup.gpu.device.alloc_zeros::<Real>(1).unwrap();
    let mut ke_scratch = setup.gpu.device.alloc_zeros::<Real>(1).unwrap();
    let mut vir_scratch = setup.gpu.device.alloc_zeros::<Real>(1).unwrap();

    setup
        .force_field
        .step(
            &mut setup.buffers,
            &setup.sim_box,
            &mut timings,
            AggregateLevel::ForcesAndScalars,
        )
        .expect("warm-up");

    let n_steps = (target_time / dt as f64).round() as usize;
    let log_every = ((log_every_t / dt as f64).round() as usize).max(1);
    let mut samples = Vec::new();

    println!(
        "# {}: dt = {:.4e} au_time ({:.3e} s), n_steps = {}, log every {} steps",
        log_label,
        dt,
        dt as f64 * 2.4188843265857195e-17,
        n_steps,
        log_every,
    );
    println!("# step,t_au,ke,pe,total,virial");
    for step in 0..=n_steps {
        if step % log_every == 0 {
            let ke = compute_kinetic_energy(&mut setup.buffers, &mut ke_scratch)
                .expect("ke") as f64;
            let pe = compute_total_potential_energy(&mut setup.buffers, &mut pe_scratch)
                .expect("pe") as f64;
            let vir = compute_total_virial(&mut setup.buffers, &mut vir_scratch)
                .expect("virial") as f64;
            let t = (step as f64) * (dt as f64);
            samples.push(Sample { t, total: ke + pe });
            println!("{},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}", step, t, ke, pe, ke + pe, vir);
        }
        if step == n_steps {
            break;
        }
        IntegratorStepExt::step(
            &mut *integrator,
            &mut setup.buffers,
            &mut setup.sim_box,
            &mut setup.force_field,
            dt,
            &mut timings,
        )
        .expect("integrator step");
    }
    samples
}

/// Linear-fit slope of total(t) over a samples slice, in Hartree / au_time.
fn linear_slope(samples: &[Sample]) -> f64 {
    let n = samples.len() as f64;
    let sx: f64 = samples.iter().map(|s| s.t).sum();
    let sy: f64 = samples.iter().map(|s| s.total).sum();
    let sxx: f64 = samples.iter().map(|s| s.t * s.t).sum();
    let sxy: f64 = samples.iter().map(|s| s.t * s.total).sum();
    let denom = n * sxx - sx * sx;
    (n * sxy - sx * sy) / denom
}

fn run_two_dts(config: &str, target_time: f64, log_every_t: f64) {
    println!("\n=== {}: dt = nominal ===", config);
    let a = run_window(config, 1.0, "dt_x1", target_time, log_every_t);
    println!("\n=== {}: dt halved ===", config);
    let b = run_window(config, 0.5, "dt_x0.5", target_time, log_every_t);
    let cutoff_a = (a.len() as f64 * 0.8) as usize;
    let cutoff_b = (b.len() as f64 * 0.8) as usize;
    let slope_a = linear_slope(&a[..cutoff_a]);
    let slope_b = linear_slope(&b[..cutoff_b]);
    let ratio = slope_a / slope_b;
    println!("\n=== {} drift summary ===", config);
    println!("dt = nominal:  slope = {:+.6e} H/au_time", slope_a);
    println!("dt = halved:   slope = {:+.6e} H/au_time", slope_b);
    println!("ratio (nominal / halved) = {:.3}", ratio);
    println!(
        "  truncation-error expectation: ~4.0 (drift ∝ dt²)\n  \
         kernel-bug expectation:        ~1.0"
    );
}

#[test]
#[ignore = "ad-hoc diagnostic; run with --ignored --nocapture"]
fn energy_drift_spc_water_256() {
    let config_path = std::path::PathBuf::from("examples/spc-water-256/spc.in.toml");
    if !config_path.exists() {
        eprintln!("skip: {} not present", config_path.display());
        return;
    }
    let target_time = 5000.0_f64;
    let log_every_t = 250.0_f64;
    run_two_dts("examples/spc-water-256/spc.in.toml", target_time, log_every_t);
}
