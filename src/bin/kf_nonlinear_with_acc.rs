use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{
    add_noise6, circular_motion_acc, circular_motion_vel, circular_motion_with_vel_acc,
};
use nalgebra::*;

const NOISE_SIGMA_SQUARED: f64 = 0.0001;
const DELTA_TIME: f64 = 0.0001;

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v), velocity (sx, sy), and acceleration (ax, ay) as initial input
fn main() {
    let mut largest_error = 0.0;

    // 1. Observation matrix
    let h = Matrix6::identity();

    // 2. Observation noise COVARIANCE matrix
    let r = SMatrix::identity();

    // 3. Initial state
    let x_initial = circular_motion_with_vel_acc(0.0);

    // 4. The initial state COVARIANCE matrix
    let p_initial = SMatrix::identity();

    // Create a non-linear KF (EKF)
    let mut nonlinear_kalman = Kalman1M::new_ekf_with_input(step_fn, h, r, x_initial, p_initial);

    for i in 1..100 {
        println!("iteration:        {:?}", i);
        let time = (i as f64) * DELTA_TIME;

        let vel_input: SVector<f64, 2> = circular_motion_vel(time);
        let acc_input: SVector<f64, 2> = circular_motion_acc(time);
        let input = Vector5::new(
            vel_input.x,
            vel_input.y,
            acc_input.x,
            acc_input.y,
            DELTA_TIME,
        );

        let actual_state = circular_motion_with_vel_acc(time);
        let noisy_state = add_noise6(actual_state, NOISE_SIGMA_SQUARED);
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

    println!("h (Observation matrix):       {:?}", h);
    println!("r (Observation noise COV):    {:?}", r);
    println!("x_initial (Initial state):    {:?}", x_initial);
    println!("p_initial (Initial state COV):{:?}", p_initial);
    println!("Time Delta:                   {:?}", DELTA_TIME);
    println!("Largest error:                {:?}", largest_error);
}

// Step function for circular motion
fn step_fn(state: SVector<f64, 6>, input: SVector<f64, 5>) -> StepReturn<f64, 6> {
    let dt = input.a;
    let dt_squared = dt * dt;

    let px = state.x;
    let py = state.y;
    let vx = state.z;
    let vy = state.w;
    let ax = state.a;
    let ay = state.b;

    let ax_1 = input.z;
    let ay_1 = input.w;

    let jx = (ax_1 - ax) / dt;
    let jy = (ay_1 - ay) / dt;

    let jacobian_vec = Vector6::new(vx + ax * dt, vy + ay * dt, ax, ay, jx, jy);
    // let jacobian_vec = Vector4::new(0.0, 0.0, 0.0, 0.0);

    // Diagonal Jacobian matrix
    let jacobian = Matrix6::from_diagonal(&jacobian_vec);

    // // First Column Jacobian
    // let mut jacobian = Matrix4::zeros();
    // jacobian.column_mut(0).copy_from(&jacobian_vec);

    StepReturn {
        state: Vector6::new(
            px + vx * dt + ax * dt_squared,
            py + vy * dt + ay * dt_squared,
            vx + ax * dt,
            vy + ay * dt,
            ax_1,
            ay_1,
        ),
        jacobian,
        covariance: SMatrix::identity(),
    }
}
