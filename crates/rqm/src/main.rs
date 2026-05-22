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
    /// Open a requirement's text in $EDITOR. On save, .rqm/ is
    /// updated and the working tree is rebuilt.
    ///
    /// The target can be either a canonical stable_id
    /// (`rq-XXXXXXXX`) or a file:line cursor (`rqm/foo.md:42`).
    Edit {
        /// Either `rq-XXXXXXXX` or `<path>:<line>` (1-based line).
        target: String,
    },
    /// Show the current state of a requirement: kind, ancestry,
    /// children, text-blob preview, source-blob locations, and any
    /// aliases that point at it.
    Log {
        /// Either `rq-XXXXXXXX` or `<path>:<line>`.
        target: String,
    },
    /// Move a blob to a different position, either within its current
    /// file or in another managed file.
    ///
    /// `src` is `<path>:<line>`; `dst` is `<path>:<line>`,
    /// `<path>:start`, or `<path>:end`. Default is to insert after the
    /// blob containing the destination line.
    Mv {
        /// Source: `<path>:<line>` of any line inside the blob to move.
        src: String,
        /// Destination: `<path>:<line>`, `<path>:start`, or `<path>:end`.
        dst: String,
        /// Insert before the destination blob instead of after.
        #[arg(long, conflicts_with = "split")]
        before: bool,
        /// Split the destination blob at the requested line; insert
        /// the moved blob between the two halves.
        #[arg(long, conflicts_with = "before")]
        split: bool,
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
        Cmd::Log { target } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let target = rqm::edit::EditTarget::parse(&target)?;
            rqm::log::run(&store, &target)?;
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Mv {
            src,
            dst,
            before,
            split,
        } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let modifier = if split {
                rqm::mv::Modifier::Split
            } else if before {
                rqm::mv::Modifier::Before
            } else {
                rqm::mv::Modifier::After
            };
            let src = rqm::mv::SourceSpec::parse(&src)?;
            let dst = rqm::mv::DestSpec::parse(&dst, modifier)?;
            let outcome = rqm::mv::do_move(&store, src, dst)?;
            println!("moved blob: {}", outcome.moved_blob);
            if let Some(info) = &outcome.split {
                println!("  split: {} -> {} + {}", info.original, info.left, info.right);
                for id in &info.metas_updated {
                    println!("  updated meta: {id}");
                }
            }
            for p in &outcome.paths_updated {
                println!("  rewrote file-tree: {}", p.display());
            }
            let report = rqm::materialize::build(&store, &root)?;
            for p in &report.written {
                println!("  wrote {}", p.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Edit { target } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let target = rqm::edit::EditTarget::parse(&target)?;
            let blob = rqm::edit::target_to_blob(&store, &target)?;
            match rqm::edit::edit_blob_interactive(&store, &blob)? {
                rqm::edit::EditOutcome::Unchanged => {
                    println!("no changes");
                }
                rqm::edit::EditOutcome::Canceled => {
                    println!("edit canceled (editor exited non-zero)");
                }
                rqm::edit::EditOutcome::Changed {
                    new_blob,
                    metas_updated,
                    paths_updated,
                } => {
                    println!("new blob: {new_blob}");
                    for id in &metas_updated {
                        println!("  updated meta: {id}");
                    }
                    for p in &paths_updated {
                        println!("  rewrote file-tree: {}", p.display());
                    }
                    let report = rqm::materialize::build(&store, &root)?;
                    for p in &report.written {
                        println!("  wrote {}", p.display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}
