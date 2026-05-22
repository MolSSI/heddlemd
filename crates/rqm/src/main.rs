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
    /// Dump a requirement and its full associated content: the
    /// text_blob, every source_blob (with file location), and every
    /// direct child's text_blob. Designed for reading the
    /// requirements-to-source mapping in full.
    View {
        /// Either `rq-XXXXXXXX` or `<path>:<line>`. A `<path>:<line>`
        /// pointing at a source blob resolves to its owning canonical
        /// requirement.
        target: String,
    },
    /// Insert a new blob at a specified position in a managed file.
    /// Auto-creates the file if it is not yet managed (use `:start` or
    /// `:end` in that case).
    ///
    /// Operates in one of two modes — exactly one must be selected:
    ///
    ///  * Attribute mode: `--owner <id|file:line>` (repeatable) —
    ///    attaches the new blob to one or more existing requirements.
    ///
    ///  * Create-new mode: `--parent <id|file:line>` or `--no-parent`
    ///    — generates a fresh stable_id, creates a new requirement
    ///    of the given `--kind` (default `behavior`) that owns the new
    ///    blob as its text_blob.
    ///
    /// Content is read from `$EDITOR` (default), `--from-file`, or
    /// `--from-stdin`.
    Insert {
        /// Position: `<path>:<line>`, `<path>:start`, or `<path>:end`.
        target: String,
        /// Attribute mode: existing requirement(s) that own the new
        /// blob. May repeat for joint ownership. Each is
        /// `rq-XXXXXXXX` or `<path>:<line>`. Aliases are rejected.
        #[arg(long, conflicts_with_all = ["parent", "no_parent"])]
        owner: Vec<String>,
        /// Create-new mode: parent stable_id (or file:line cursor) for
        /// the new requirement. Mutually exclusive with `--no-parent`
        /// and `--owner`.
        #[arg(long, conflicts_with_all = ["owner", "no_parent"])]
        parent: Option<String>,
        /// Create-new mode: create as a DAG root (no parent).
        /// Mutually exclusive with `--parent` and `--owner`.
        #[arg(long = "no-parent", conflicts_with_all = ["owner", "parent"])]
        no_parent: bool,
        /// Kind of the new requirement (only used with `--parent` or
        /// `--no-parent`). One of `behavior`, `design`, `pending`.
        #[arg(long, default_value = "behavior")]
        kind: String,
        /// Anchor before the destination blob instead of after.
        #[arg(long)]
        before: bool,
        /// Read content from this file instead of `$EDITOR`.
        #[arg(long, conflicts_with = "from_stdin")]
        from_file: Option<PathBuf>,
        /// Read content from stdin instead of `$EDITOR`.
        #[arg(long, conflicts_with = "from_file")]
        from_stdin: bool,
    },
    /// Remove a blob or a requirement.
    ///
    /// If `target` is `<path>:<line>`, removes the blob covering that
    /// line from its file-tree. Refuses to remove a requirement's
    /// text_blob unless `--force` is given.
    ///
    /// If `target` is `rq-XXXXXXXX`, removes the requirement entirely
    /// (delete the ref, remove every file-tree entry carrying its
    /// stable_id, auto-delete any aliases that point at it). Refuses
    /// if the requirement has source_blobs attached or children in the
    /// DAG — clear those first.
    Rm {
        /// `<path>:<line>` or `rq-XXXXXXXX`.
        target: String,
        /// (blob form only) Allow removing a blob that is a
        /// requirement's text_blob.
        #[arg(long)]
        force: bool,
    },
    /// Change a requirement's stable_id everywhere.
    ///
    /// Updates the ref, the meta, every child's parents, every
    /// file-tree entry, every alias's canonical pointer, and rewrites
    /// every reachable blob containing the old id as a word-bounded
    /// stamp.
    Rename {
        /// `rq-XXXXXXXX` or `<path>:<line>` (canonical only).
        old: String,
        /// New stable_id (must be `rq-XXXXXXXX`, not already in use).
        new: String,
    },
    /// Change a requirement's parents in the DAG.
    ///
    /// Replaces the target's `parents` list with the given parents
    /// (zero for a DAG root, or one or more for a multi-parent DAG).
    /// Rejects aliases, self-loops, and cycles.
    Reassign {
        /// `rq-XXXXXXXX` or `<path>:<line>`. Must resolve to a
        /// canonical requirement.
        target: String,
        /// New parents (canonical stable_ids or file:line cursors).
        /// May repeat. Mutually exclusive with `--no-parent`.
        #[arg(long, conflicts_with = "no_parent")]
        parent: Vec<String>,
        /// Make the requirement a DAG root (empty parents list).
        /// Mutually exclusive with `--parent`.
        #[arg(long = "no-parent", conflicts_with = "parent")]
        no_parent: bool,
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
        Cmd::View { target } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let target = rqm::edit::EditTarget::parse(&target)?;
            rqm::view::run(&store, &target)?;
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Insert {
            target,
            owner,
            parent,
            no_parent,
            kind,
            before,
            from_file,
            from_stdin,
        } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            // Pick the mode based on which group of options was used.
            let mode = if !owner.is_empty() {
                let mut owners = Vec::with_capacity(owner.len());
                for spec in &owner {
                    let t = rqm::edit::EditTarget::parse(spec)?;
                    owners.push(rqm::edit::target_to_canonical_id_strict(&store, &t)?);
                }
                rqm::insert::InsertMode::AttributeTo { owners }
            } else if parent.is_some() || no_parent {
                let parent_id = if let Some(p) = parent {
                    let t = rqm::edit::EditTarget::parse(&p)?;
                    Some(rqm::edit::target_to_canonical_id_strict(&store, &t)?)
                } else {
                    None
                };
                let kind = match kind.as_str() {
                    "behavior" => rqm::object::Kind::Behavior,
                    "design" => rqm::object::Kind::Design,
                    "pending" => rqm::object::Kind::Pending,
                    other => anyhow::bail!(
                        "unknown --kind {other:?}; expected behavior, design, or pending"
                    ),
                };
                rqm::insert::InsertMode::CreateNew {
                    parent: parent_id,
                    kind,
                }
            } else {
                anyhow::bail!(
                    "must specify one of --owner (attribute), --parent (new child), \
                     or --no-parent (new DAG root)"
                );
            };
            let content = if let Some(p) = from_file {
                rqm::insert::read_content(rqm::insert::ContentSource::File(&p))?
            } else if from_stdin {
                rqm::insert::read_content(rqm::insert::ContentSource::Stdin)?
            } else {
                rqm::insert::read_via_editor()?
            };
            let spec = rqm::insert::InsertSpec::parse(&target, before)?;
            let outcome = rqm::insert::do_insert(&store, spec, mode, content)?;
            if outcome.created_file {
                println!("created managed path: {}", outcome.path.display());
            }
            println!("new blob: {}", outcome.new_blob);
            match &outcome.change {
                rqm::insert::InsertChange::Attributed(ids) => {
                    for id in ids {
                        println!("  updated meta: {id}");
                    }
                }
                rqm::insert::InsertChange::Created(id) => {
                    println!("  created requirement: {id}");
                }
            }
            let report = rqm::materialize::build(&store, &root)?;
            for p in &report.written {
                println!("  wrote {}", p.display());
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Rm { target, force } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let parsed = rqm::edit::EditTarget::parse(&target)?;
            match parsed {
                rqm::edit::EditTarget::Id(id) => {
                    if force {
                        anyhow::bail!(
                            "--force is for blob removal only; requirement \
                             removal has explicit safety checks that --force \
                             does not override"
                        );
                    }
                    let outcome = rqm::rm::do_remove_requirement(&store, &id)?;
                    println!("removed requirement: {}", outcome.removed);
                    for alias in &outcome.aliases_removed {
                        println!("  auto-deleted alias: {alias}");
                    }
                    for p in &outcome.paths_updated {
                        println!("  rewrote file-tree: {}", p.display());
                    }
                    let report = rqm::materialize::build(&store, &root)?;
                    for p in &report.written {
                        println!("  wrote {}", p.display());
                    }
                }
                rqm::edit::EditTarget::FileLine { path, line } => {
                    let spec = rqm::rm::RemoveSpec { path, line };
                    let outcome = rqm::rm::do_remove(&store, spec, force)?;
                    println!("removed blob: {}", outcome.removed_blob);
                    if let Some(id) = &outcome.ghosted_requirement {
                        println!(
                            "  warning: {id} is now ghosted (text_blob no longer materialized)"
                        );
                    }
                    for id in &outcome.metas_updated {
                        println!("  updated meta: {id}");
                    }
                    let report = rqm::materialize::build(&store, &root)?;
                    for p in &report.written {
                        println!("  wrote {}", p.display());
                    }
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Rename { old, new } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let old_t = rqm::edit::EditTarget::parse(&old)?;
            let old_id = rqm::edit::target_to_canonical_id_strict(&store, &old_t)?;
            let new_id = rqm::object::StableId::new(new);
            let outcome = rqm::rename::do_rename(&store, &old_id, &new_id)?;
            println!("renamed {} -> {}", outcome.old, outcome.new);
            if outcome.blobs_rewritten > 0 {
                println!("  blobs rewritten: {}", outcome.blobs_rewritten);
            }
            for id in &outcome.metas_updated {
                println!("  updated meta: {id}");
            }
            for p in &outcome.paths_updated {
                println!("  rewrote file-tree: {}", p.display());
            }
            for alias in &outcome.aliases_updated {
                println!("  redirected alias: {alias}");
            }
            if !outcome.paths_updated.is_empty() || outcome.blobs_rewritten > 0 {
                let report = rqm::materialize::build(&store, &root)?;
                for p in &report.written {
                    println!("  wrote {}", p.display());
                }
            }
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Reassign {
            target,
            parent,
            no_parent,
        } => {
            let store = rqm::store::Store::open(&cli.rqm_dir)?;
            let target_t = rqm::edit::EditTarget::parse(&target)?;
            let target_id = rqm::edit::target_to_canonical_id_strict(&store, &target_t)?;
            if !no_parent && parent.is_empty() {
                anyhow::bail!(
                    "must specify --parent <id>... or --no-parent"
                );
            }
            let mut new_parents = Vec::with_capacity(parent.len());
            for p in &parent {
                let pt = rqm::edit::EditTarget::parse(p)?;
                new_parents.push(rqm::edit::target_to_canonical_id_strict(&store, &pt)?);
            }
            let outcome = rqm::reassign::do_reassign(&store, &target_id, new_parents)?;
            println!("reassigned {}", outcome.target);
            if outcome.old_parents.is_empty() {
                println!("  old parents: (DAG root)");
            } else {
                for p in &outcome.old_parents {
                    println!("  old parent: {p}");
                }
            }
            if outcome.new_parents.is_empty() {
                println!("  new parents: (DAG root)");
            } else {
                for p in &outcome.new_parents {
                    println!("  new parent: {p}");
                }
            }
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
