use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise4, circular_motion_vel, circular_motion_with_vel};
use nalgebra::*;

const NOISE_SIGMA_SQUARED: f64 = 0.0001;
const DELTA_TIME: f64 = 0.0001;

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v) and velocity (sx, sy) as initial input
fn main() {
    let mut largest_error = 0.0;

    // 1. Observation matrix
    let h: Matrix<f64, Const<4>, Const<4>, ArrayStorage<f64, 4, 4>> = SMatrix::identity();
    println!("Observation matrix    (h):            {:?}", h);

    // 2. Observation noise COVARIANCE matrix
    let r = SMatrix::identity();
    println!("Observation noise COV (r):            {:?}", r);

    // 3. Initial state
    let x_initial = circular_motion_with_vel(0.0);
    println!("Initial state         (x_initial):    {:?}", x_initial);

    // 4. The initial state COVARIANCE matrix
    let p_initial = SMatrix::identity();
    println!("Initial state COV     (p_initial):    {:?}\n", p_initial);

    // Create a non-linear KF (EKF)
    let mut nonlinear_kalman = Kalman1M::new_ekf_with_input(step_fn, h, r, x_initial, p_initial);

    for i in 1..100 {
        println!("iteration:        {:?}", i);
        let time = (i as f64) * DELTA_TIME;

        let input: SVector<f64, 2> = circular_motion_vel(time);
        let input = Vector3::new(input.x, input.y, DELTA_TIME);

        let actual_state = circular_motion_with_vel(time);
        let noisy_state = add_noise4(actual_state, NOISE_SIGMA_SQUARED);
        println!("current state:    {:?}", nonlinear_kalman.state());
        println!("actual_state:     {:?}", actual_state);
        println!("noisy_state:      {:?}", noisy_state);

        // Update EKF with new sensor measurement data
        nonlinear_kalman.update(noisy_state);

        // Predict the next state by giving it velocity and delta time
        let predicted_state = nonlinear_kalman.predict(input);

        // Get the error
        let error = actual_state - predicted_state;
        error.iter().for_each(|e| {
            if e.abs() > largest_error {
                largest_error = e.abs();
            }
        });

        println!("predicted_state   {:?}", predicted_state);
        println!("error             {:?}\n", actual_state - predicted_state);
    }

    println!("Time Delta:       {:?}", DELTA_TIME);
    println!("Largest error:    {:?}", largest_error);
}

// Step function for circular motion
fn step_fn(state: SVector<f64, 4>, input: SVector<f64, 3>) -> StepReturn<f64, 4> {
    let dt = input.z;

    let px = state.x;
    let py = state.y;
    let vx = state.z;
    let vy = state.w;

    let ax = (input.x - vx) / dt;
    let ay = (input.y - vy) / dt;

    let jacobian_vec = Vector4::new(vx, vy, ax, ay);
    // let jacobian_vec = Vector4::new(0.0, 0.0, 0.0, 0.0);

    // Diagonal Jacobian matrix
    let jacobian = Matrix4::from_diagonal(&jacobian_vec);

    // // First Column Jacobian
    // let mut jacobian = Matrix4::zeros();
    // jacobian.column_mut(0).copy_from(&jacobian_vec);

    StepReturn {
        state: Vector4::new(px + vx * dt, py + vy * dt, vx, vy),
        jacobian,
        covariance: SMatrix::identity(),
    }
}
