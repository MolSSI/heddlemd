pub mod gpu;

pub mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}
