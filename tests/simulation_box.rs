use dynamics::pbc::{SimulationBox, SimulationBoxError};

fn default_box() -> SimulationBox {
    SimulationBox::new_orthorhombic(10.0, 8.0, 6.0).expect("default box")
}

// --- Construction ---

#[test] // rq-27ffd3f4
fn construct_with_positive_finite_edge_lengths() {
    let b = SimulationBox::new_orthorhombic(10.0, 8.0, 6.0).expect("ok");
    assert_eq!(b.lengths(), [10.0_f32, 8.0, 6.0]);
    assert_eq!(b.lx(), 10.0);
    assert_eq!(b.ly(), 8.0);
    assert_eq!(b.lz(), 6.0);
}

#[test] // rq-e1b51bd9
fn volume_returns_product_of_edge_lengths() {
    let b = SimulationBox::new_orthorhombic(2.0, 3.0, 5.0).expect("ok");
    assert_eq!(b.volume(), 30.0_f32);
}

#[test] // rq-8259c9ca
fn reject_zero_lx() {
    let err = SimulationBox::new_orthorhombic(0.0, 8.0, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-05eb9fbb
fn reject_zero_ly() {
    let err = SimulationBox::new_orthorhombic(10.0, 0.0, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "ly");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-74aa3a99
fn reject_zero_lz() {
    let err = SimulationBox::new_orthorhombic(10.0, 8.0, 0.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lz");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-9b1f8a7c
fn reject_negative_lx() {
    let err = SimulationBox::new_orthorhombic(-1.0, 8.0, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert_eq!(value, -1.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-19fe4806
fn reject_nan_lx() {
    let err = SimulationBox::new_orthorhombic(f32::NAN, 8.0, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-7f867e37
fn reject_infinite_ly() {
    let err = SimulationBox::new_orthorhombic(10.0, f32::INFINITY, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLength { axis, value } => {
            assert_eq!(axis, "ly");
            assert_eq!(value, f32::INFINITY);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-7541fd8a
fn validation_order_is_lx_then_ly_then_lz() {
    let err = SimulationBox::new_orthorhombic(0.0, -1.0, f32::NAN).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test] // rq-b9a4e3de
fn non_finite_check_precedes_non_positive_check_on_same_axis() {
    let err = SimulationBox::new_orthorhombic(f32::NAN, 8.0, 6.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

// --- minimum_image ---

#[test] // rq-8c045718
fn minimum_image_of_zero_displacement_is_zero() {
    let b = default_box();
    assert_eq!(b.minimum_image([0.0, 0.0, 0.0]), [0.0_f32, 0.0, 0.0]);
}

#[test] // rq-bfb3b9d8
fn minimum_image_leaves_displacement_strictly_inside_unchanged() {
    let b = default_box();
    assert_eq!(b.minimum_image([4.0, 3.0, 2.0]), [4.0_f32, 3.0, 2.0]);
}

#[test] // rq-9a9523d9
fn minimum_image_at_plus_half_l_maps_to_minus_half_l() {
    let b = default_box();
    assert_eq!(b.minimum_image([5.0, 0.0, 0.0]), [-5.0_f32, 0.0, 0.0]);
}

#[test] // rq-d19fc020
fn minimum_image_at_minus_half_l_stays_at_minus_half_l() {
    let b = default_box();
    assert_eq!(b.minimum_image([-5.0, 0.0, 0.0]), [-5.0_f32, 0.0, 0.0]);
}

#[test] // rq-f7b922df
fn minimum_image_just_past_plus_half_l_wraps_one_period() {
    let b = default_box();
    assert_eq!(b.minimum_image([6.0, 0.0, 0.0]), [-4.0_f32, 0.0, 0.0]);
}

#[test] // rq-a8df30ac
fn minimum_image_just_past_minus_half_l_wraps_one_period() {
    let b = default_box();
    assert_eq!(b.minimum_image([-6.0, 0.0, 0.0]), [4.0_f32, 0.0, 0.0]);
}

#[test] // rq-0ae304bc
fn minimum_image_handles_many_period_displacements() {
    let b = default_box();
    let result = b.minimum_image([24.0, 0.0, 0.0]);
    let lx = b.lx();
    assert!(result[0] >= -lx * 0.5);
    assert!(result[0] < lx * 0.5);
    // 24.0 wraps by 2 periods of 10.0 to 4.0
    assert_eq!(result[0], 4.0_f32);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test] // rq-c9618bdd
fn minimum_image_is_per_axis_independent() {
    let b = default_box();
    // x=6, lx=10  -> -4
    // y=-5, ly=8  -> 3
    // z=4, lz=6   -> -2
    let result = b.minimum_image([6.0, -5.0, 4.0]);
    assert_eq!(result, [-4.0_f32, 3.0, -2.0]);
}

// --- wrap_position ---

#[test] // rq-3e8324c2
fn wrap_position_inside_primary_image_unchanged() {
    let b = default_box();
    assert_eq!(b.wrap_position([4.0, 3.0, 2.0]), [4.0_f32, 3.0, 2.0]);
}

#[test] // rq-4b9d059e
fn wrap_position_wraps_outside_primary_image() {
    let b = default_box();
    let position = [12.0_f32, -5.0, 7.0];
    let result = b.wrap_position(position);
    assert!(result[0] >= -b.lx() * 0.5 && result[0] < b.lx() * 0.5);
    assert!(result[1] >= -b.ly() * 0.5 && result[1] < b.ly() * 0.5);
    assert!(result[2] >= -b.lz() * 0.5 && result[2] < b.lz() * 0.5);
    assert_eq!(result, b.minimum_image(position));
}

#[test] // rq-941c4000
fn wrap_position_is_idempotent() {
    let b = default_box();
    let position = [123.45_f32, -67.89, 42.0];
    let once = b.wrap_position(position);
    let twice = b.wrap_position(once);
    assert_eq!(twice, once);
}

#[test] // rq-a1fc0841
fn wrap_position_and_minimum_image_agree() {
    let b = default_box();
    let v = [17.0_f32, -13.0, 9.5];
    assert_eq!(b.minimum_image(v), b.wrap_position(v));
}

// --- Numerical edge cases ---

#[test] // rq-4b63564b
fn nan_displacement_propagates_to_nan_output() {
    let b = default_box();
    let result = b.minimum_image([f32::NAN, 0.0, 0.0]);
    assert!(result[0].is_nan());
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

// --- Generation counter ---

#[test] // rq-2cb82d44
fn newly_constructed_box_reports_generation_zero() {
    let b = default_box();
    assert_eq!(b.generation(), 0);
}

#[test] // rq-a3563587
fn successful_set_lengths_increments_generation_by_one() {
    let mut b = default_box();
    b.set_lengths(12.0, 9.0, 7.0).expect("ok");
    assert_eq!(b.lengths(), [12.0_f32, 9.0, 7.0]);
    assert_eq!(b.generation(), 1);
}

#[test] // rq-9e09673b
fn successive_set_lengths_calls_increment_generation_monotonically() {
    let mut b = default_box();
    b.set_lengths(11.0, 8.0, 6.0).expect("ok");
    b.set_lengths(11.0, 9.0, 6.0).expect("ok");
    b.set_lengths(11.0, 9.0, 7.0).expect("ok");
    assert_eq!(b.lengths(), [11.0_f32, 9.0, 7.0]);
    assert_eq!(b.generation(), 3);
}

#[test] // rq-89c71321
fn set_lengths_rejects_non_positive_length_without_mutating() {
    let mut b = default_box();
    let err = b.set_lengths(0.0, 9.0, 7.0).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.lengths(), [10.0_f32, 8.0, 6.0]);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-d28774dc
fn set_lengths_rejects_non_finite_length_without_mutating() {
    let mut b = default_box();
    let err = b.set_lengths(10.0, f32::NAN, 7.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLength { axis, value } => {
            assert_eq!(axis, "ly");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.lengths(), [10.0_f32, 8.0, 6.0]);
    assert_eq!(b.generation(), 0);
}

#[test] // rq-153dd875
fn set_lengths_validation_order_is_lx_then_ly_then_lz() {
    let mut b = default_box();
    let err = b.set_lengths(0.0, -1.0, f32::NAN).expect_err("err");
    match err {
        SimulationBoxError::NonPositiveLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert_eq!(value, 0.0);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.generation(), 0);
}

#[test] // rq-7edab504
fn set_lengths_non_finite_precedes_non_positive_on_same_axis() {
    let mut b = default_box();
    let err = b.set_lengths(f32::NAN, 9.0, 7.0).expect_err("err");
    match err {
        SimulationBoxError::NonFiniteLength { axis, value } => {
            assert_eq!(axis, "lx");
            assert!(value.is_nan());
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(b.generation(), 0);
}

#[test] // rq-d6e10419
fn minimum_image_after_set_lengths_uses_new_edge_lengths() {
    let mut b = default_box();
    b.set_lengths(20.0, 8.0, 6.0).expect("ok");
    let result = b.minimum_image([12.0, 0.0, 0.0]);
    assert_eq!(result[0], -8.0_f32);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.0);
}

#[test] // rq-fa98ca13
fn copy_of_simulation_box_carries_originals_generation() {
    let mut b = default_box();
    b.set_lengths(11.0, 8.0, 6.0).expect("ok");
    let copy = b;
    assert_eq!(copy.generation(), 1);
    assert_eq!(copy.lengths(), [11.0_f32, 8.0, 6.0]);
}

#[test] // rq-22fb3b0e
fn mutating_a_copy_does_not_affect_the_original() {
    let b = default_box();
    let mut copy = b;
    copy.set_lengths(20.0, 8.0, 6.0).expect("ok");
    assert_eq!(copy.lengths(), [20.0_f32, 8.0, 6.0]);
    assert_eq!(copy.generation(), 1);
    assert_eq!(b.lengths(), [10.0_f32, 8.0, 6.0]);
    assert_eq!(b.generation(), 0);
}
