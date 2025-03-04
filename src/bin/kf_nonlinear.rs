use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, circular_motion, circular_motion_vel, plot};
use nalgebra::*;

const NOISE_SIGMA_SQUARED: f64 = 0.001;
const DELTA_TIME: f64 = 0.1;

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v) as initial input
fn main() {
    let mut largest_error = 0.0;

    // 1. Observation matrix
    let h = SMatrix::identity();

    // 2. Observation noise COVARIANCE matrix
    let r = SMatrix::identity();

    // 3. Initial state
    let x_initial = circular_motion(0.0);

    // 4. The initial state COVARIANCE matrix
    let p_initial = SMatrix::identity();

    // Create a non-linear KF (EKF)
    let mut nonlinear_kalman = Kalman1M::new_ekf_with_input(step_fn, h, r, x_initial, p_initial);

    let mut t_history = Vec::new();
    let mut actual_state_history = Vec::new();
    let mut noisy_state_history = Vec::new();
    let mut predicted_state_history = Vec::new();

    for i in 0..100 {
        println!("iteration:        {:?}", i);
        let time = (i as f64) * DELTA_TIME;

        let input: SVector<f64, 2> = circular_motion_vel(time);
        let input = Vector3::new(input.x, input.y, DELTA_TIME);

        let actual_state = circular_motion(time);
        let noisy_state = add_noise(actual_state, NOISE_SIGMA_SQUARED);
        println!("current state:    {:?}", nonlinear_kalman.state());
        println!("actual_state:     {:?}", actual_state);
        println!("noisy_state:      {:?}", noisy_state);

        // Update EKF with new sensor measurement data
        nonlinear_kalman.update(noisy_state);

        // Predict the next state by giving it velocity and delta time
        let predicted_state = nonlinear_kalman.predict(input);

        // Save the history for plotting
        t_history.push(time);
        actual_state_history.push((actual_state.x, actual_state.y));
        noisy_state_history.push((noisy_state.x, noisy_state.y));
        predicted_state_history.push((predicted_state.x, predicted_state.y));

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

    plot(
        "Position",
        t_history,
        actual_state_history,
        predicted_state_history,
    )
    .unwrap();
}

// Step function for circular motion
fn step_fn(state: SVector<f64, 2>, input: SVector<f64, 3>) -> StepReturn<f64, 2> {
    let vx: f64 = input.x;
    let vy = input.y;
    let dt = input.z;

    StepReturn {
        state: Vector2::new(state.x + vx * dt, state.y + vy * dt),
        // jacobian: Matrix2::new(vx, 0.0, 0.0, vy),
        jacobian: Matrix2::new(vx, 0.0, vy, 0.0),
        covariance: SMatrix::identity(),
    }
}
