// rq-3d7c8e53
//
// Host-side Philox-4×32-10 RNG. Byte-for-byte equivalent to the
// device-side `philox4x32_10` in `kernels/langevin.cu`. Reusable from
// any host-side stochastic integrator that needs reproducible random
// draws (CSVR, C-rescale barostat, future stochastic slots).

const PHILOX_M0: u32 = 0xD2511F53;
const PHILOX_M1: u32 = 0xCD9E8D57;
const PHILOX_W0: u32 = 0x9E3779B9;
const PHILOX_W1: u32 = 0xBB67AE85;

#[inline]
fn mulhi32(a: u32, b: u32) -> u32 {
    ((a as u64).wrapping_mul(b as u64) >> 32) as u32
}

/// Counter-based Philox-4×32-10. Inputs: 2-word key, 4-word counter.
/// Output: 4-word block. Pure function; matches the device-side helper
/// in `kernels/langevin.cu` byte-for-byte.
pub fn philox_4x32_10(
    key_lo: u32,
    key_hi: u32,
    ctr0: u32,
    ctr1: u32,
    ctr2: u32,
    ctr3: u32,
) -> [u32; 4] {
    let mut c0 = ctr0;
    let mut c1 = ctr1;
    let mut c2 = ctr2;
    let mut c3 = ctr3;
    let mut k0 = key_lo;
    let mut k1 = key_hi;
    for _ in 0..10 {
        let hi0 = mulhi32(c0, PHILOX_M0);
        let lo0 = c0.wrapping_mul(PHILOX_M0);
        let hi2 = mulhi32(c2, PHILOX_M1);
        let lo2 = c2.wrapping_mul(PHILOX_M1);
        let nc0 = hi2 ^ c1 ^ k0;
        let nc1 = lo2;
        let nc2 = hi0 ^ c3 ^ k1;
        let nc3 = lo0;
        c0 = nc0;
        c1 = nc1;
        c2 = nc2;
        c3 = nc3;
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
    }
    [c0, c1, c2, c3]
}

/// One standard-normal draw via Box-Muller (cos branch), matching the
/// device-side `philox_gaussian` formula exactly. Returns `f64` (the
/// device-side helper truncates to `f32` for its on-device use; CSVR
/// keeps the full `f64` because its chain math benefits from it).
pub fn philox_normal(
    key_lo: u32,
    key_hi: u32,
    ctr0: u32,
    ctr1: u32,
    ctr2: u32,
    ctr3: u32,
) -> f64 {
    let out = philox_4x32_10(key_lo, key_hi, ctr0, ctr1, ctr2, ctr3);
    let scale = 1.0_f64 / 4_294_967_296.0;
    let u1 = (out[0] as f64 + 0.5) * scale;
    let u2 = (out[1] as f64 + 0.5) * scale;
    let r = (-2.0_f64 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    r * theta.cos()
}
