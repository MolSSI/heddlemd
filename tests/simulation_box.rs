use heddle_md::pbc::{SimulationBox, SimulationBoxError};
use heddle_md::precision::Real;

fn default_box() -> SimulationBox {
    SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("default box")
}

// --- Construction: orthorhombic ---

#[test] // rq-27ffd3f4
fn construct_orthorhombic_box() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.lx(), 10.0);
    assert_eq!(b.ly(), 8.0);
    assert_eq!(b.lz(), 6.0);
    assert_eq!(b.xy(), 0.0);
    assert_eq!(b.xz(), 0.0);
    assert_eq!(b.yz(), 0.0);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-e1b51bd9
fn volume_returns_lx_ly_lz_regardless_of_tilts() {
    let b = SimulationBox::new(2.0, 3.0, 5.0, 7.0, -9.0, 11.0).expect("ok");
    assert_eq!(b.volume(), 30.0);
}

// --- Construction: triclinic ---

#[test] // rq-7a1c24be
fn construct_triclinic_box_with_non_zero_tilts() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.5, -2.0, 0.5).expect("ok");
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 1.5, -2.0, 0.5]);
}

#[test] // rq-67c5a863
fn tilts_may_be_negative() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, -3.0, -5.0, -1.5).expect("ok");
    assert_eq!(b.xy(), -3.0);
    assert_eq!(b.xz(), -5.0);
    assert_eq!(b.yz(), -1.5);
}

#[test] // rq-650875cc
fn tilts_may_exceed_the_corresponding_diagonals() {
    // No reduced-tilt enforcement; geometric infeasibility is the
    // neighbor list's problem (caught via min_perpendicular_width).
    let b = SimulationBox::new(2.0, 3.0, 4.0, 50.0, 50.0, 50.0).expect("ok");
    assert_eq!(b.xy(), 50.0);
}

// --- Construction: rejection ---

#[test] // rq-8259c9ca
fn reject_zero_lx() {
    let err = SimulationBox::new(0.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-05eb9fbb
fn reject_zero_ly() {
    let err = SimulationBox::new(10.0, 0.0, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "ly");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-74aa3a99
fn reject_zero_lz() {
    let err = SimulationBox::new(10.0, 8.0, 0.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lz");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-9b1f8a7c
fn reject_negative_diagonal() {
    let err = SimulationBox::new(-1.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lx");
            assert_eq!(value, -1.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-19fe4806
fn reject_nan_diagonal() {
    let err = SimulationBox::new(Real::NAN, 8.0, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-7f867e37
fn reject_infinite_diagonal() {
    let err = SimulationBox::new(10.0, Real::INFINITY, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "ly");
            assert_eq!(value, Real::INFINITY);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-0c9dc32b
fn reject_nan_tilt() {
    let err = SimulationBox::new(10.0, 8.0, 6.0, Real::NAN, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "xy");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-5318db55
fn reject_infinite_tilt() {
    let err = SimulationBox::new(10.0, 8.0, 6.0, 0.0, Real::INFINITY, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "xz");
            assert_eq!(value, Real::INFINITY);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-7541fd8a
fn validation_order_is_lx_ly_lz_xy_xz_yz() {
    let err = SimulationBox::new(0.0, -1.0, 0.0, Real::NAN, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-b9a4e3de
fn non_finite_check_precedes_non_positive_check_on_diagonal() {
    let err = SimulationBox::new(Real::NAN, 8.0, 6.0, 0.0, 0.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// --- minimum_image: orthorhombic special case ---

#[test] // rq-8c045718
fn minimum_image_of_zero_displacement_is_zero() {
    let b = default_box();
    assert_eq!(b.minimum_image([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
}

#[test] // rq-bfb3b9d8
fn minimum_image_leaves_displacement_strictly_inside_unchanged() {
    let b = default_box();
    assert_eq!(b.minimum_image([4.0, 3.0, 2.0]), [4.0, 3.0, 2.0]);
}

#[test] // rq-9a9523d9
fn minimum_image_at_plus_half_l_maps_to_minus_half_l() {
    let b = default_box();
    assert_eq!(b.minimum_image([5.0, 0.0, 0.0]), [-5.0, 0.0, 0.0]);
}

#[test] // rq-d19fc020
fn minimum_image_at_minus_half_l_stays_at_minus_half_l() {
    let b = default_box();
    assert_eq!(b.minimum_image([-5.0, 0.0, 0.0]), [-5.0, 0.0, 0.0]);
}

#[test] // rq-f7b922df
fn minimum_image_just_past_plus_half_l_wraps_one_period() {
    let b = default_box();
    assert_eq!(b.minimum_image([6.0, 0.0, 0.0]), [-4.0, 0.0, 0.0]);
}

#[test] // rq-a8df30ac
fn minimum_image_just_past_minus_half_l_wraps_one_period() {
    let b = default_box();
    assert_eq!(b.minimum_image([-6.0, 0.0, 0.0]), [4.0, 0.0, 0.0]);
}

#[test] // rq-0ae304bc
fn minimum_image_handles_many_period_displacements_orthorhombic() {
    let b = default_box();
    let result = b.minimum_image([24.0, 0.0, 0.0]);
    let lx = b.lx();
    assert!(result[0] >= -lx * 0.5);
    assert!(result[0] < lx * 0.5);
    // 24.0 wraps by 2 periods of 10.0 to 4.0
    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test] // rq-c9618bdd
fn minimum_image_is_per_axis_independent_for_orthorhombic() {
    let b = default_box();
    // x=6, lx=10  -> -4
    // y=-5, ly=8  -> 3
    // z=4, lz=6   -> -2
    let result = b.minimum_image([6.0, -5.0, 4.0]);
    assert_eq!(result, [-4.0, 3.0, -2.0]);
}

// --- minimum_image: triclinic ---

#[test] // rq-b4e4bdc7
fn minimum_image_of_c_aligned_displacement_subtracts_c_vector() {
    // box with xz=2.0, yz=3.0; the wrap of v_z = 4.0 picks k_c = 1
    // and propagates k_c * (xz, yz) = (2, 3) into x and y.
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 2.0, 3.0).expect("ok");
    let result = b.minimum_image([2.0, 3.0, 4.0]);
    // v_z: 4.0 - 6.0 = -2.0
    // v_y: 3.0 - 3.0 = 0.0; k_b wrap leaves it at 0.0
    // v_x: 2.0 - 2.0 = 0.0; k_a wrap leaves it at 0.0
    assert_eq!(result, [0.0, 0.0, -2.0]);
}

#[test] // rq-261fde88
fn minimum_image_requires_both_c_and_b_wrapping() {
    // box with xy=1.0 only.
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 0.0, 0.0).expect("ok");
    let result = b.minimum_image([0.0, 5.0, 0.0]);
    // v_z = 0 stays; k_c = 0
    // v_y = 5.0; k_b = floor((5.0 + 4.0) / 8.0) = 1; v_y -= 8.0 -> -3.0
    // x channel: subtract k_b * xy = 1.0; v_x: 0.0 - 1.0 = -1.0
    // k_a = floor((-1.0 + 5.0) / 10.0) = 0; v_x stays -1.0
    assert_eq!(result, [-1.0, -3.0, 0.0]);
}

#[test] // rq-a9ab33a8
fn wrap_result_lies_inside_primary_parallelepiped() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 4.0, -3.0, 1.5).expect("ok");
    for v in [
        [0.0, 0.0, 0.0],
        [12.0, -9.0, 7.5],
        [50.0, 0.0, 0.0],
        [-100.0, 200.0, -300.0],
    ] {
        let result = b.minimum_image(v);
        let s = b.fractional_coords(result);
        for d in 0..3 {
            assert!(
                s[d] >= -0.5 && s[d] < 0.5,
                "fractional coord {} = {} out of [-0.5, 0.5) for v = {v:?}",
                ["a", "b", "c"][d],
                s[d]
            );
        }
    }
}

// --- wrap_position ---

#[test] // rq-3e8324c2
fn wrap_position_inside_primary_image_unchanged() {
    let b = default_box();
    assert_eq!(b.wrap_position([4.0, 3.0, 2.0]), [4.0, 3.0, 2.0]);
}

#[test] // rq-4b9d059e
fn wrap_position_wraps_outside_primary_image_orthorhombic() {
    let b = default_box();
    let position = [12.0, -5.0, 7.0];
    let result = b.wrap_position(position);
    assert!(result[0] >= -b.lx() * 0.5 && result[0] < b.lx() * 0.5);
    assert!(result[1] >= -b.ly() * 0.5 && result[1] < b.ly() * 0.5);
    assert!(result[2] >= -b.lz() * 0.5 && result[2] < b.lz() * 0.5);
    assert_eq!(result, b.minimum_image(position));
}

#[test] // rq-941c4000
fn wrap_position_is_idempotent_orthorhombic() {
    let b = default_box();
    let position = [123.45, -67.89, 42.0];
    let once = b.wrap_position(position);
    let twice = b.wrap_position(once);
    assert_eq!(twice, once);
}

#[test] // rq-5269221c
fn wrap_position_is_idempotent_for_triclinic_box() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.5, -2.0, 0.5).expect("ok");
    let position = [200.0, -150.0, 75.5];
    let once = b.wrap_position(position);
    let twice = b.wrap_position(once);
    assert_eq!(twice, once);
}

#[test] // rq-a1fc0841
fn wrap_position_and_minimum_image_agree() {
    let b = default_box();
    let v = [17.0, -13.0, 9.5];
    assert_eq!(b.minimum_image(v), b.wrap_position(v));
}

// --- wrap_position_with_image_count ---

#[test] // rq-870ed681
fn wrap_position_with_image_count_returns_image_triple() {
    let b = default_box(); // orthorhombic, lx=10
    let (wrapped, image) = b.wrap_position_with_image_count([12.0, 0.0, 0.0]);
    assert_eq!(wrapped, [2.0, 0.0, 0.0]);
    assert_eq!(image, [1, 0, 0]);
}

#[test] // rq-5355f3f0
fn wrap_position_with_image_count_for_triclinic() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).expect("ok");
    // pos_z = 20.0 wraps via k_c = floor((20 + 3)/6) = 3
    let (wrapped, image) = b.wrap_position_with_image_count([0.0, 0.0, 20.0]);
    assert_eq!(image[2], 3);
    // After the wrap result must lie in the primary parallelepiped:
    let s = b.fractional_coords(wrapped);
    for d in 0..3 {
        assert!(s[d] >= -0.5 && s[d] < 0.5);
    }
}

#[test] // rq-6c52e57d
fn wrap_position_with_image_count_unwrap_invariant() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).expect("ok");
    let p = [37.5, -41.0, 22.5];
    let (wrapped, image) = b.wrap_position_with_image_count(p);
    // wrapped + image[0] * a + image[1] * b + image[2] * c == p
    let lat = b.lattice();
    let (lx, ly, lz) = (lat[0], lat[1], lat[2]);
    let (xy, xz, yz) = (lat[3], lat[4], lat[5]);
    let recovered_x = wrapped[0]
        + image[0] as Real * lx
        + image[1] as Real * xy
        + image[2] as Real * xz;
    let recovered_y =
        wrapped[1] + image[1] as Real * ly + image[2] as Real * yz;
    let recovered_z = wrapped[2] + image[2] as Real * lz;
    assert!((recovered_x - p[0]).abs() < 1.0e-3);
    assert!((recovered_y - p[1]).abs() < 1.0e-3);
    assert!((recovered_z - p[2]).abs() < 1.0e-3);
}

// --- Fractional coordinates ---

#[test] // rq-545c961a
fn fractional_coords_inverts_cartesian_coords() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).expect("ok");
    let s = [0.1, -0.2, 0.3];
    let v = b.cartesian_coords(s);
    let s_back = b.fractional_coords(v);
    for d in 0..3 {
        assert!(
            (s_back[d] - s[d]).abs() < 1.0e-6,
            "round-trip mismatch on axis {d}: {} vs {}",
            s_back[d],
            s[d]
        );
    }
}

#[test] // rq-7f018040
fn cartesian_coords_of_unit_fractional_triples_yields_lattice_vectors() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).expect("ok");
    let a = b.cartesian_coords([1.0, 0.0, 0.0]);
    assert_eq!(a, [10.0, 0.0, 0.0]);
    let bv = b.cartesian_coords([0.0, 1.0, 0.0]);
    assert_eq!(bv, [1.0, 8.0, 0.0]);
    let c = b.cartesian_coords([0.0, 0.0, 1.0]);
    assert_eq!(c, [2.0, 3.0, 6.0]);
}

// --- min_perpendicular_width ---

#[test] // rq-ef6ae25a
fn min_perpendicular_width_equals_min_lx_ly_lz_for_orthorhombic() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(b.min_perpendicular_width(), 6.0);
}

#[test] // rq-47e800e0
fn min_perpendicular_width_of_c_tilted_box() {
    // yz=10, so w_b = ly*lz / sqrt(lz² + yz²) = 100 / sqrt(200) ≈ 7.07
    let b = SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0).expect("ok");
    let w = b.min_perpendicular_width();
    let expected = 100.0 / (200.0 as Real).sqrt();
    assert!(
        (w - expected).abs() < 1.0e-5,
        "expected {expected}, got {w}"
    );
    assert!(w < 10.0);
}

#[test] // rq-9c3ecf3f
fn min_perpendicular_width_of_xy_tilted_box() {
    // xy=5, so w_a = (lx*ly*lz) / sqrt((ly*lz)² + (xy*lz)²)
    //              = 1000 / sqrt(10000 + 2500)
    //              = 1000 / sqrt(12500)
    let b = SimulationBox::new(10.0, 10.0, 10.0, 5.0, 0.0, 0.0).expect("ok");
    let w = b.min_perpendicular_width();
    let expected = 1000.0 / (12500.0 as Real).sqrt();
    assert!(
        (w - expected).abs() < 1.0e-4,
        "expected {expected}, got {w}"
    );
    assert!(w < 10.0);
}

// --- Numerical edge cases ---

#[test] // rq-4b63564b
fn nan_displacement_propagates_to_nan_output() {
    let b = default_box();
    let result = b.minimum_image([Real::NAN, 0.0, 0.0]);
    assert!(result[0].is_nan());
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test] // rq-74f48855
fn nan_z_displacement_propagates_through_tilt_coupling() {
    // For a triclinic box with non-zero xz and yz, a NaN z-displacement
    // propagates into the x and y channels via the tilt-subtraction.
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 2.0, 3.0).expect("ok");
    let result = b.minimum_image([0.0, 0.0, Real::NAN]);
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_nan());
}

// --- Generation counter ---

#[test] // rq-2cb82d44
fn newly_constructed_box_reports_generation_zero() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).expect("ok");
    assert_eq!(b.generation(), 0);
}

#[test] // rq-a3563587
fn successful_set_lattice_increments_generation_by_one() {
    let mut b = default_box();
    b.set_lattice(12.0, 9.0, 7.0, 1.0, 2.0, 3.0).expect("ok");
    assert_eq!(b.lattice(), [12.0, 9.0, 7.0, 1.0, 2.0, 3.0]);
    assert_eq!(b.generation(), 1);
}

#[test] // rq-9e09673b
fn successive_set_lattice_calls_increment_generation_monotonically() {
    let mut b = default_box();
    b.set_lattice(11.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    b.set_lattice(11.0, 9.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    b.set_lattice(11.0, 9.0, 7.0, 1.0, 2.0, 3.0).expect("ok");
    assert_eq!(b.lattice(), [11.0, 9.0, 7.0, 1.0, 2.0, 3.0]);
    assert_eq!(b.generation(), 3);
}

#[test] // rq-89c71321
fn set_lattice_rejects_non_positive_diagonal_without_mutating() {
    let mut b = default_box();
    let err = b
        .set_lattice(0.0, 9.0, 7.0, 0.0, 0.0, 0.0)
        .expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-d28774dc
fn set_lattice_rejects_non_finite_diagonal_without_mutating() {
    let mut b = default_box();
    let err = b
        .set_lattice(10.0, Real::NAN, 7.0, 0.0, 0.0, 0.0)
        .expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "ly");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-50fa922c
fn set_lattice_rejects_non_finite_tilt_without_mutating() {
    let mut b = default_box();
    let err = b
        .set_lattice(10.0, 8.0, 6.0, 1.0, Real::NAN, 0.0)
        .expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "xz");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-153dd875
fn set_lattice_validation_order_is_lx_ly_lz_xy_xz_yz() {
    let mut b = default_box();
    let err = b
        .set_lattice(0.0, -1.0, 0.0, Real::NAN, 0.0, 0.0)
        .expect_err("err");
    match err {
        SimulationBoxError::NonPositiveDiagonal { name, value } => {
            assert_eq!(name, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.generation(), 0);
}

#[test] // rq-7edab504
fn set_lattice_non_finite_precedes_non_positive_on_diagonal() {
    let mut b = default_box();
    let err = b
        .set_lattice(Real::NAN, 9.0, 7.0, 0.0, 0.0, 0.0)
        .expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLatticeValue { name, value } => {
            assert_eq!(name, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.generation(), 0);
}

#[test] // rq-d6e10419
fn minimum_image_after_set_lattice_uses_new_parameters() {
    let mut b = default_box();
    b.set_lattice(20.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    let result = b.minimum_image([12.0, 0.0, 0.0]);
    assert_eq!(result[0], -8.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test] // rq-491235c1
fn minimum_image_after_set_lattice_reflects_new_tilts() {
    let mut b = default_box();
    b.set_lattice(10.0, 8.0, 6.0, 0.0, 4.0, 0.0).expect("ok");
    let result = b.minimum_image([0.0, 0.0, 4.0]);
    // k_c = floor((4 + 3)/6) = 1
    // v_z = 4 - 6 = -2
    // v_x -= 1 * xz = 4 -> v_x = -4
    assert_eq!(result, [-4.0, 0.0, -2.0]);
}

#[test] // rq-fa98ca13
fn copy_of_simulation_box_carries_originals_generation() {
    let mut b = default_box();
    b.set_lattice(11.0, 8.0, 6.0, 1.0, 0.0, 0.0).expect("ok");
    let copy = b;
    assert_eq!(copy.generation(), 1);
    assert_eq!(copy.lattice(), [11.0, 8.0, 6.0, 1.0, 0.0, 0.0]);
}

#[test] // rq-22fb3b0e
fn mutating_a_copy_does_not_affect_the_original() {
    let b = default_box();
    let mut copy = b;
    copy.set_lattice(20.0, 8.0, 6.0, 0.0, 0.0, 0.0).expect("ok");
    assert_eq!(copy.lattice(), [20.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(copy.generation(), 1);
    assert_eq!(b.lattice(), [10.0, 8.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(b.generation(), 0);
}

// --- check_min_perpendicular_width ---

#[test] // rq-0fa3b49f
fn check_min_perpendicular_width_ok_when_every_width_meets_threshold() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).unwrap();
    assert!(b.check_min_perpendicular_width(5.0).is_ok());
}

#[test] // rq-0061906c
fn check_min_perpendicular_width_ok_at_exact_equality() {
    // Smallest width is lz = 6.0; threshold of 6.0 must still pass.
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).unwrap();
    assert!(b.check_min_perpendicular_width(6.0).is_ok());
}

#[test] // rq-394a4bb1
fn check_min_perpendicular_width_flags_direction_a_when_only_w_a_fails() {
    let b = SimulationBox::new(4.0, 10.0, 10.0, 0.0, 0.0, 0.0).unwrap();
    let err = b.check_min_perpendicular_width(5.0).unwrap_err();
    match err {
        SimulationBoxError::PerpendicularWidthTooSmall {
            direction,
            width,
            required,
        } => {
            assert_eq!(direction, "a");
            assert_eq!(width, 4.0);
            assert_eq!(required, 5.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-7600d28c
fn check_min_perpendicular_width_flags_direction_b_when_only_w_b_fails() {
    let b = SimulationBox::new(10.0, 4.0, 10.0, 0.0, 0.0, 0.0).unwrap();
    let err = b.check_min_perpendicular_width(5.0).unwrap_err();
    match err {
        SimulationBoxError::PerpendicularWidthTooSmall {
            direction,
            width,
            required,
        } => {
            assert_eq!(direction, "b");
            assert_eq!(width, 4.0);
            assert_eq!(required, 5.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-5ffa0551
fn check_min_perpendicular_width_flags_direction_c_when_only_w_c_fails() {
    let b = SimulationBox::new(10.0, 10.0, 4.0, 0.0, 0.0, 0.0).unwrap();
    let err = b.check_min_perpendicular_width(5.0).unwrap_err();
    match err {
        SimulationBoxError::PerpendicularWidthTooSmall {
            direction,
            width,
            required,
        } => {
            assert_eq!(direction, "c");
            assert_eq!(width, 4.0);
            assert_eq!(required, 5.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-743ae35c
fn check_min_perpendicular_width_reports_first_failing_direction_when_multiple_fail() {
    // All three widths fail; only direction "a" should be reported.
    let b = SimulationBox::new(4.0, 4.0, 4.0, 0.0, 0.0, 0.0).unwrap();
    let err = b.check_min_perpendicular_width(5.0).unwrap_err();
    match err {
        SimulationBoxError::PerpendicularWidthTooSmall {
            direction, width, ..
        } => {
            assert_eq!(direction, "a");
            assert_eq!(width, 4.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-8ac1a52f
fn check_min_perpendicular_width_on_triclinic_uses_perpendicular_width_not_edge() {
    // A box with yz tilt: w_b = (ly * lz) / sqrt(lz^2 + yz^2) =
    // 100 / sqrt(200) ≈ 7.071. Edge length ly = 10.0 would pass an 8.0
    // threshold; the perpendicular width fails.
    let b = SimulationBox::new(10.0, 10.0, 10.0, 0.0, 0.0, 10.0).unwrap();
    let err = b.check_min_perpendicular_width(8.0).unwrap_err();
    match err {
        SimulationBoxError::PerpendicularWidthTooSmall {
            direction,
            width,
            required,
        } => {
            assert_eq!(direction, "b");
            let expected = 100.0 / (200.0 as Real).sqrt();
            assert!((width - expected).abs() < 1.0e-6);
            assert_eq!(required, 8.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test] // rq-98ac1915
fn check_min_perpendicular_width_with_non_positive_threshold_always_ok() {
    let b = SimulationBox::new(10.0, 8.0, 6.0, 0.0, 0.0, 0.0).unwrap();
    assert!(b.check_min_perpendicular_width(-1.0).is_ok());
    assert!(b.check_min_perpendicular_width(0.0).is_ok());
}

#[test] // rq-3eaf65b6
fn check_min_perpendicular_width_is_deterministic() {
    let b1 = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).unwrap();
    let b2 = SimulationBox::new(10.0, 8.0, 6.0, 1.0, 2.0, 3.0).unwrap();
    let r1 = b1.check_min_perpendicular_width(7.0);
    let r2 = b2.check_min_perpendicular_width(7.0);
    match (r1, r2) {
        (Ok(()), Ok(())) => {}
        (
            Err(SimulationBoxError::PerpendicularWidthTooSmall {
                direction: d1,
                width: w1,
                required: r1,
            }),
            Err(SimulationBoxError::PerpendicularWidthTooSmall {
                direction: d2,
                width: w2,
                required: r2,
            }),
        ) => {
            assert_eq!(d1, d2);
            assert_eq!(w1.to_bits(), w2.to_bits());
            assert_eq!(r1.to_bits(), r2.to_bits());
        }
        other => panic!("non-deterministic outcomes: {other:?}"),
    }
}
