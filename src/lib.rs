pub mod gpu;
pub mod state;

pub mod kernels {
    include!(concat!(env!("OUT_DIR"), "/kernels.rs"));
}
