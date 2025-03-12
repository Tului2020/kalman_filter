use nalgebra::{Vector2, Vector3};

use crate::add_noise;

pub struct PolarCircular {
    radius: f64,
    angular_velocity: f64,
    noise: f64,
}

impl PolarCircular {
    /// Create a new PolarCircular instance
    pub fn new(radius: f64, rpm: f64, noise: f64) -> Self {
        Self {
            radius,
            angular_velocity: rpm * std::f64::consts::PI / 30.0,
            noise,
        }
    }

    /// Gets the initial state of the system
    pub fn initial_state(&self) -> Vector3<f64> {
        Vector3::new(self.radius, 0.0, 0.0)
    }

    /// Gets the state of the system at a given time
    pub fn state(&self, time: f64) -> Vector3<f64> {
        Vector3::new(
            self.radius,
            time * self.angular_velocity,
            self.angular_velocity,
        )
    }

    /// Gets the "measured" state of the system at a given time
    pub fn measurement(&self, time: f64) -> Vector2<f64> {
        let state = self.state(time);
        add_noise(Vector2::new(state[0], state[1]), self.noise)
    }

    /// Gets the initial "measured" state of the system
    pub fn initial_measurement(&self) -> Vector3<f64> {
        add_noise(self.initial_state(), self.noise)
    }
}
