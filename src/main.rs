use dynamics::gpu::init_device;

fn main() {
    match init_device() {
        Ok(_device) => println!("CUDA device initialized; fill module loaded."),
        Err(e) => {
            eprintln!("failed to initialize CUDA device: {e}");
            std::process::exit(1);
        }
    }
}
