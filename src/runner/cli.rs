use super::*;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE_LINE: &str = "\
usage: heddlemd run     <config-path>
       heddlemd lint    <config-path> [--with-gpu]
       heddlemd analyze <analysis-path>";

// rq-f7e279ee
pub fn cli_main(args: Vec<String>) -> ExitCode {
    ExitCode::from(cli_main_u8(args))
}

// Testable variant returning the raw exit code.
pub fn cli_main_u8(args: Vec<String>) -> u8 {
    // rq-82d0c34a rq-7e5cb9f8
    let mut iter = args.into_iter();
    let _exe = iter.next();
    let sub = match iter.next() {
        Some(s) => s,
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    match sub.as_str() {
        "run" => cli_main_run(iter.collect()),
        "lint" => cli_main_lint(iter.collect()),
        "analyze" => cli_main_analyze(iter.collect()),
        _ => {
            eprintln!("{USAGE_LINE}");
            1
        }
    }
}

fn cli_main_run(rest: Vec<String>) -> u8 {
    let mut iter = rest.into_iter();
    let config_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    if iter.next().is_some() {
        eprintln!("{USAGE_LINE}");
        return 1;
    }

    let registries = crate::Registries::with_builtins();
    match run_simulation_with_phase(&config_path, &registries) {
        Ok(summary) => {
            // rq-d29872e4 — one line per phase + one aggregate line.
            for ps in &summary.phases {
                let disp = if ps.elapsed_micros >= 10_000 {
                    format!("{} ms", ps.elapsed_micros / 1000)
                } else {
                    format!("{} \u{00b5}s", ps.elapsed_micros)
                };
                if ps.kind == "minimization" {
                    let conv = ps.convergence.unwrap_or("unknown");
                    // rq-6eb845c5 — StepFloor is a numerical-progress
                    // exhaustion, not a physical-tolerance convergence.
                    // Name the final F_max so the user can judge the
                    // residual gradient at hand-off; every other
                    // convergence reason implies `F_max ≤
                    // force_tolerance` or similar physical criterion
                    // and doesn't need the annotation.
                    if conv == "step_floor" {
                        let f_max = ps.min_final_max_force.unwrap_or(f64::NAN);
                        println!(
                            "[heddlemd] phase `{}`: {} iters in {} (converged: {}, F_max = {:.3e} N, frames: {}, log rows: {})",
                            ps.name, ps.n_steps, disp, conv, f_max,
                            ps.frames_written, ps.log_rows_written
                        );
                    } else {
                        println!(
                            "[heddlemd] phase `{}`: {} iters in {} (converged: {}, frames: {}, log rows: {})",
                            ps.name, ps.n_steps, disp, conv, ps.frames_written, ps.log_rows_written
                        );
                    }
                } else {
                    println!(
                        "[heddlemd] phase `{}`: {} steps in {} (frames: {}, log rows: {})",
                        ps.name, ps.n_steps, disp, ps.frames_written, ps.log_rows_written
                    );
                }
            }
            let total_disp = if summary.total_elapsed_micros >= 10_000 {
                format!("{} ms", summary.total_elapsed_micros / 1000)
            } else {
                format!("{} \u{00b5}s", summary.total_elapsed_micros)
            };
            println!(
                "[heddlemd] complete: {} phases, {} steps in {}",
                summary.phases.len(),
                summary.total_n_steps,
                total_disp,
            );
            0
        }
        Err((err, phase)) => {
            eprintln!("error: {err}");
            match phase {
                ExitPhase::Setup => 1,
                ExitPhase::Loop => 2,
            }
        }
    }
}

fn cli_main_lint(rest: Vec<String>) -> u8 {
    let mut config_path: Option<PathBuf> = None;
    let mut with_gpu = false;
    for arg in rest {
        match arg.as_str() {
            "--with-gpu" => with_gpu = true,
            a if a.starts_with("--") => {
                eprintln!("{USAGE_LINE}");
                return 1;
            }
            _ => {
                if config_path.is_some() {
                    eprintln!("{USAGE_LINE}");
                    return 1;
                }
                config_path = Some(PathBuf::from(arg));
            }
        }
    }
    let config_path = match config_path {
        Some(p) => p,
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };

    // Dispatch on extension: `.in.analysis` runs the analyze lint
    // pipeline; everything else falls through to the simulation lint
    // pipeline (whose filename-convention check rejects non-`.in.toml`
    // paths internally with `InvalidConfigFilename`).
    let is_analysis = config_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.ends_with(".in.analysis"))
        .unwrap_or(false);

    if is_analysis {
        if with_gpu {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
        let report = crate::analysis::lint_analyses(&config_path);
        let _ = report.write_to(&mut std::io::stdout());
        if let Some(err) = report.first_failure() {
            eprintln!("error: {err}");
        }
        return if report.ok() { 0 } else { 1 };
    }

    let report = lint_simulation(&config_path, with_gpu);
    let _ = report.write_to(&mut std::io::stdout());
    if let Some(err) = report.first_failure() {
        eprintln!("error: {err}");
    }
    if report.ok() { 0 } else { 1 }
}

fn cli_main_analyze(rest: Vec<String>) -> u8 {
    let mut iter = rest.into_iter();
    let config_path = match iter.next() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{USAGE_LINE}");
            return 1;
        }
    };
    if iter.next().is_some() {
        eprintln!("{USAGE_LINE}");
        return 1;
    }
    match crate::analysis::run_analyses(&config_path) {
        Ok(summary) => {
            let elapsed = summary.elapsed_micros;
            let elapsed_disp = if elapsed >= 10_000 {
                format!("{} ms", elapsed / 1000)
            } else {
                format!("{elapsed} \u{00b5}s")
            };
            println!(
                "[heddlemd] analyze complete: {} analyses over {} frames in {}",
                summary.analyses_written, summary.frames_consumed, elapsed_disp
            );
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            // Distinguish setup vs loop-time errors. The trajectory
            // pass and per-analysis writes correspond to the loop
            // phase; everything else is setup.
            match &e {
                crate::analysis::AnalyzeError::Trajectory(_)
                | crate::analysis::AnalyzeError::Analysis { .. } => 2,
                _ => 1,
            }
        }
    }
}
