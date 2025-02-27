use nalgebra::{SVector, Vector2};

pub fn circular_motion(time: f64) -> SVector<f64, 2> {
    Vector2::new(time.cos(), time.sin())
}

pub fn circular_motion_vel(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.sin(), time.cos())
}

pub fn add_noise(state: SVector<f64, 2>, noise: f64) -> SVector<f64, 2> {
    state + Vector2::new(rand::random::<f64>() * noise, rand::random::<f64>() * noise)
}
