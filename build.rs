use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const NVCC_INSTALL_HINT: &str =
    "The CUDA Toolkit is required to build this project. Install it from \
     https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ and ensure \
     that `nvcc` is on PATH.";

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set by Cargo"));
    let manifest_dir =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let kernels_dir = manifest_dir.join("kernels");

    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-changed=build.rs");

    let mut kernels: Vec<(String, PathBuf)> = Vec::new();
    let entries = fs::read_dir(&kernels_dir).unwrap_or_else(|e| {
        panic!("failed to read kernels directory {}: {e}", kernels_dir.display())
    });
    for entry in entries {
        let entry = entry.expect("failed to read directory entry in kernels/");
        let path = entry.path();
        if path.extension() == Some(OsStr::new("cu")) {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| panic!("non-UTF-8 file name in kernels/: {}", path.display()))
                .to_owned();
            kernels.push((stem, path));
        }
    }
    kernels.sort_by(|a, b| a.0.cmp(&b.0));

    for (stem, source) in &kernels {
        compile_to_ptx(source, &out_dir.join(format!("{stem}.ptx")));
    }

    let generated = out_dir.join("kernels.rs");
    let mut f = fs::File::create(&generated)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", generated.display()));
    for (stem, _) in &kernels {
        let upper = stem.to_ascii_uppercase();
        writeln!(
            f,
            "pub const {upper}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{stem}.ptx\"));"
        )
        .expect("failed to write kernels.rs");
    }
}

fn compile_to_ptx(source: &Path, output: &Path) {
    let mut cmd = Command::new("nvcc");
    cmd.arg("--ptx").arg("-o").arg(output).arg(source);

    let output_result = cmd.output();
    match output_result {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let code = out
                .status
                .code()
                .map(|c| c.to_string())
                .unwrap_or_else(|| "<no exit code>".into());
            panic!(
                "nvcc failed compiling {src} (exit code {code}).\n\
                 command: nvcc --ptx -o {dst} {src}\n\
                 stdout:\n{stdout}\n\
                 stderr:\n{stderr}\n\n{hint}",
                src = source.display(),
                dst = output.display(),
                stdout = String::from_utf8_lossy(&out.stdout),
                stderr = String::from_utf8_lossy(&out.stderr),
                hint = NVCC_INSTALL_HINT,
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            panic!(
                "nvcc not found on PATH (failed to compile {src}).\n\n{hint}",
                src = source.display(),
                hint = NVCC_INSTALL_HINT,
            );
        }
        Err(e) => {
            panic!(
                "failed to invoke nvcc for {src}: {e}\n\n{hint}",
                src = source.display(),
                hint = NVCC_INSTALL_HINT,
            );
        }
    }
}
