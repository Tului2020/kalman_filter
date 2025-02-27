use nalgebra::{ArrayStorage, Const, Matrix};

pub fn circular_motion(time: f64) -> Matrix<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>> {
    Matrix::<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>>::new(time.cos(), time.sin())
}

pub fn circular_motion_vel(time: f64) -> Matrix<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>> {
    Matrix::<f64, Const<2>, Const<1>, ArrayStorage<f64, 2, 1>>::new(-time.sin(), time.cos())
}
