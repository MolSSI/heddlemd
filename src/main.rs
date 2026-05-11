use std::process::ExitCode;

use dynamics::runner::cli_main;

fn main() -> ExitCode {
    cli_main(std::env::args().collect())
}
