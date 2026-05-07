// rq-b75afb31
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationBox {
    lengths: [f32; 3],
}

// rq-aef9888b
#[derive(Debug)]
pub enum SimulationBoxError {
    NonFiniteLength { axis: &'static str, value: f32 },
    NonPositiveLength { axis: &'static str, value: f32 },
}

impl std::fmt::Display for SimulationBoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimulationBoxError::NonFiniteLength { axis, value } => {
                write!(f, "non-finite simulation-box length {axis} = {value}")
            }
            SimulationBoxError::NonPositiveLength { axis, value } => {
                write!(f, "non-positive simulation-box length {axis} = {value}")
            }
        }
    }
}

impl std::error::Error for SimulationBoxError {}

fn check_axis(axis: &'static str, value: f32) -> Result<(), SimulationBoxError> {
    if !value.is_finite() {
        return Err(SimulationBoxError::NonFiniteLength { axis, value });
    }
    if value <= 0.0 {
        return Err(SimulationBoxError::NonPositiveLength { axis, value });
    }
    Ok(())
}

fn wrap_axis(x: f32, length: f32) -> f32 {
    x - length * ((x + length * 0.5) / length).floor()
}

fn wrap_per_axis(v: [f32; 3], lengths: [f32; 3]) -> [f32; 3] {
    [
        wrap_axis(v[0], lengths[0]),
        wrap_axis(v[1], lengths[1]),
        wrap_axis(v[2], lengths[2]),
    ]
}

impl SimulationBox {
    // rq-f0da71ea
    pub fn new_orthorhombic(
        lx: f32,
        ly: f32,
        lz: f32,
    ) -> Result<Self, SimulationBoxError> {
        check_axis("lx", lx)?;
        check_axis("ly", ly)?;
        check_axis("lz", lz)?;
        Ok(SimulationBox {
            lengths: [lx, ly, lz],
        })
    }

    // rq-e8be1a1c
    pub fn lengths(&self) -> [f32; 3] {
        self.lengths
    }

    // rq-f73a0f99
    pub fn lx(&self) -> f32 {
        self.lengths[0]
    }

    // rq-f73a0f99
    pub fn ly(&self) -> f32 {
        self.lengths[1]
    }

    // rq-f73a0f99
    pub fn lz(&self) -> f32 {
        self.lengths[2]
    }

    // rq-3b9ed390
    pub fn volume(&self) -> f32 {
        self.lengths[0] * self.lengths[1] * self.lengths[2]
    }

    // rq-d49c9093
    pub fn minimum_image(&self, displacement: [f32; 3]) -> [f32; 3] {
        wrap_per_axis(displacement, self.lengths)
    }

    // rq-9b1c84c3
    pub fn wrap_position(&self, position: [f32; 3]) -> [f32; 3] {
        wrap_per_axis(position, self.lengths)
    }
}
