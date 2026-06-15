use std::process::ExitCode;

use heddle_md::runner::cli_main;

fn main() -> ExitCode {
    cli_main(std::env::args().collect())
}
