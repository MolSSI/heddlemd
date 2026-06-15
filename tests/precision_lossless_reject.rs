// Lossless mode is rejected at config-load time under the f64 build.
//
// rq-f64_rejects_lossless rq-default_accepts_lossless

#[cfg(feature = "f64")]
#[test]
fn f64_build_rejects_lossless_true() {
    use heddle_md::io::config::ConfigError;
    use heddle_md::integrator::velocity_verlet::VelocityVerletBuilder;
    use heddle_md::integrator::IntegratorBuilder;

    let params: toml::Value = toml::from_str("lossless = true").expect("params");
    let result = VelocityVerletBuilder.validate_params(&params);
    match result {
        Err(ConfigError::LosslessUnsupportedInF64Build) => {}
        other => panic!(
            "expected ConfigError::LosslessUnsupportedInF64Build, got {:?}",
            other
        ),
    }
}

#[test]
fn lossless_false_accepted_in_any_build() {
    use heddle_md::integrator::velocity_verlet::VelocityVerletBuilder;
    use heddle_md::integrator::IntegratorBuilder;
    let params: toml::Value = toml::from_str("lossless = false").expect("params");
    VelocityVerletBuilder
        .validate_params(&params)
        .expect("lossless = false should always validate");
}

#[cfg(not(feature = "f64"))]
#[test]
fn default_build_accepts_lossless_true() {
    use heddle_md::integrator::velocity_verlet::VelocityVerletBuilder;
    use heddle_md::integrator::IntegratorBuilder;
    let params: toml::Value = toml::from_str("lossless = true").expect("params");
    VelocityVerletBuilder
        .validate_params(&params)
        .expect("lossless = true should validate in the default f32 build");
}
