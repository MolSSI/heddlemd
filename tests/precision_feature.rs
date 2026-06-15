// Tests for the f64 compile-time precision feature flag described in
// rqm/precision.md. These tests exercise the build-time constants and
// the storage-layer round-trip in whichever feature mode the test
// binary was compiled with.

use heddle_md::precision::{
    CPU_REFERENCE_TOLERANCE, REAL_BYTES, REAL_FMT_DIGITS, REAL_IS_F64, REAL_NAME, Real,
};
use heddle_md::state::ParticleState;

// rq-default_resolves_real_to_f32 rq-f64_resolves_real_to_f64
#[test]
fn real_type_size_matches_feature_flag() {
    assert_eq!(std::mem::size_of::<Real>(), REAL_BYTES);
    if cfg!(feature = "f64") {
        assert_eq!(REAL_BYTES, 8);
    } else {
        assert_eq!(REAL_BYTES, 4);
    }
}

// rq-default_resolves_real_to_f32 rq-f64_resolves_real_to_f64
#[test]
fn real_is_f64_matches_feature_flag() {
    if cfg!(feature = "f64") {
        assert!(REAL_IS_F64);
    } else {
        assert!(!REAL_IS_F64);
    }
}

#[test]
fn real_name_matches_feature_flag() {
    if cfg!(feature = "f64") {
        assert_eq!(REAL_NAME, "f64");
    } else {
        assert_eq!(REAL_NAME, "f32");
    }
}

#[test]
fn real_fmt_digits_matches_feature_flag() {
    if cfg!(feature = "f64") {
        assert_eq!(REAL_FMT_DIGITS, 17);
    } else {
        assert_eq!(REAL_FMT_DIGITS, 9);
    }
}

#[test]
fn cpu_reference_tolerance_matches_feature_flag() {
    if cfg!(feature = "f64") {
        assert!(CPU_REFERENCE_TOLERANCE <= 1.0e-13);
    } else {
        assert!(CPU_REFERENCE_TOLERANCE <= 1.0e-5);
    }
}

// rq-particle_state_fields_are_vec_real
#[test]
fn particle_state_fields_are_vec_real() {
    let n = 1usize;
    let state = ParticleState::new(
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .expect("ParticleState::new");
    assert_eq!(state.particle_count(), n);
    // The storage layout is REAL_BYTES per element.
    assert_eq!(state.positions_x.capacity(), n);
    let layout = std::mem::size_of::<Real>();
    assert_eq!(layout, REAL_BYTES);
    // Images stay i32 regardless of feature.
    assert_eq!(std::mem::size_of_val(&state.images_x[0]), 4);
    // Particle IDs stay u32 regardless of feature.
    assert_eq!(std::mem::size_of_val(&state.particle_ids[0]), 4);
}

// rq-round_trip_default_build rq-round_trip_f64_build
#[test]
fn representative_real_round_trips_through_device() {
    // Skip when no GPU is available: init_device returns Err and we
    // treat the test as vacuously satisfied at the type level.
    let gpu = match heddle_md::gpu::init_device() {
        Ok(g) => g,
        Err(_) => return,
    };
    let pi: Real = if cfg!(feature = "f64") {
        std::f64::consts::PI as Real
    } else {
        std::f32::consts::PI as Real
    };
    let n = 1usize;
    let state = ParticleState::new(
        vec![pi; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0u32; n],
        None,
        None,
    )
    .expect("ParticleState::new");
    let buffers = heddle_md::gpu::ParticleBuffers::new(&gpu, &state).expect("buffers");
    let mut readback = state.clone();
    // Wipe the host copy so download must actually write back the bits.
    readback.positions_x[0] = 0.0;
    readback
        .download_from(&buffers)
        .expect("download_from failed");
    assert_eq!(readback.positions_x[0].to_bits(), pi.to_bits());
}
