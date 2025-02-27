use kfilter::{Kalman1M, KalmanFilter, KalmanPredict};
use nalgebra::{Matrix2, Matrix2x1, SMatrix};

fn main() {
    let x_initial = SMatrix::repeat(1.0);
    let f = Matrix2::new(1.0, 0.1, 0.0, 1.0);
    let q = SMatrix::identity();
    let h = Matrix2::new(1.0, 0.0, 0.0, 1.0);
    let r = SMatrix::identity();

    println!("x_initial:    {:?}", x_initial);
    println!("f:            {:?}", f);
    println!("q:            {:?}", q);
    println!("h:            {:?}", h);
    println!("r:            {:?}", r);

    // Create a new 2 state kalman filter
    let mut k = Kalman1M::new(f, q, h, r, x_initial);

    println!("Covariance    {:?}", k.covariance());
    println!("State         {:?}\n", k.state());

    // Run 100 timesteps
    for i in 0..10 {
        // predict based on system model
        k.predict();

        let update_matrix = Matrix2x1::new((i * 10) as f64, i as f64);
        // let update_matrix = SMatrix::repeat(i);
        println!("iteration:        {:?}", i + 1);
        println!("update_matrix:    {:?}", update_matrix);

        // update based on new measurement
        let updated_state = k.update(update_matrix);
        println!("State:            {:?}", updated_state);
        println!("Covariance        {:?}\n", k.covariance());
    }
}
