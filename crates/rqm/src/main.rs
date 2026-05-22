use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

/// Requirements traceability tool — manages the .rqm/ content-addressed
/// store and materializes it into the working tree.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Path to the .rqm/ directory. Defaults to ./.rqm.
    #[arg(long, global = true, default_value = ".rqm")]
    rqm_dir: PathBuf,

    /// Path to the repo root (used to resolve relative paths in
    /// file-trees). Defaults to the current working directory.
    #[arg(long, global = true)]
    root: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Materialize all managed paths from .rqm/.
    Build,
    /// Verify the working tree matches .rqm/'s materialized output.
    Check,
    /// Initialize an empty .rqm/ directory.
    Init,
    /// Migrate an existing markdown or source file into .rqm/.
    Migrate {
        /// Path of the file to migrate. Markdown files (.md) are
        /// migrated as requirement files; other files are migrated as
        /// source files (and require the corresponding requirements
        /// file to already be migrated).
        path: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("rqm: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let root = cli
        .root
        .map(Ok)
        .unwrap_or_else(|| std::env::current_dir().context("current dir"))?;

    match cli.cmd {
        Cmd::Init => {
            rqm::store::Store::init(&cli.rqm_dir)?;
            println!("initialized {}", cli.rqm_dir.display());
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Build => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let report = rqm::materialize::build(&store, &root)?;
            for p in &report.written {
                println!("wrote {}", p.display());
            }
            if !report.unchanged.is_empty() {
                println!("{} unchanged", report.unchanged.len());
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Check => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let report = rqm::materialize::check(&store, &root)?;
            for p in &report.diffs {
                println!("diff: {}", p.display());
            }
            for p in &report.missing {
                println!("missing: {}", p.display());
            }
            for v in &report.integrity {
                println!("integrity: {v}");
            }
            if report.is_clean() {
                println!("ok ({} paths)", report.matches.len());
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::FAILURE)
            }
        }
        Cmd::Migrate { path } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            if path.is_absolute() {
                anyhow::bail!("migrate path must be relative to --root: {}", path.display());
            }
            if path.extension().and_then(|s| s.to_str()) == Some("md") {
                rqm::migrate::migrate_markdown(&store, &root, &path)?;
                println!("migrated markdown {}", path.display());
            } else {
                rqm::migrate::migrate_source(&store, &root, &path)?;
                println!("migrated source {}", path.display());
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}
