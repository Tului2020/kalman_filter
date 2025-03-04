use nalgebra::{SVector, Vector2, Vector4, Vector6};

pub fn circular_motion(time: f64) -> SVector<f64, 2> {
    Vector2::new(time.cos(), time.sin())
}

pub fn circular_motion_vel(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.sin(), time.cos())
}

pub fn circular_motion_acc(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.cos(), -time.sin())
}

pub fn circular_motion_with_vel(time: f64) -> SVector<f64, 4> {
    let p_x = time.cos();
    let p_y = time.sin();
    let v_x = -time.sin();
    let v_y = time.cos();
    Vector4::new(p_x, p_y, v_x, v_y)
}

pub fn circular_motion_with_vel_acc(time: f64) -> SVector<f64, 6> {
    let p_x = time.cos();
    let p_y = time.sin();
    let v_x = -time.sin();
    let v_y = time.cos();
    let a_x = -time.cos();
    let a_y = -time.sin();
    Vector6::new(p_x, p_y, v_x, v_y, a_x, a_y)
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

pub fn add_noise6(state: SVector<f64, 6>, noise: f64) -> SVector<f64, 6> {
    state
        + Vector6::new(
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
        )
}
