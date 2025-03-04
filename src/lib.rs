use nalgebra::{SVector, Vector2, Vector4};

pub fn circular_motion(time: f64) -> SVector<f64, 2> {
    Vector2::new(time.cos(), time.sin())
}

pub fn circular_motion_vel(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.sin(), time.cos())
}

pub fn circular_motion_with_vel(time: f64) -> SVector<f64, 4> {
    let p_x = time.cos();
    let p_y = time.sin();
    let v_x = -time.sin();
    let v_y = time.cos();
    Vector4::new(p_x, p_y, v_x, v_y)
}

pub fn add_noise(state: SVector<f64, 2>, noise: f64) -> SVector<f64, 2> {
    state + Vector2::new(rand::random::<f64>() * noise, rand::random::<f64>() * noise)
}

pub fn add_noise4(state: SVector<f64, 4>, noise: f64) -> SVector<f64, 4> {
    state
        + Vector4::new(
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
        )
}
