use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, circular_motion, circular_motion_vel, plot};
use nalgebra::{Matrix2, Vector2};

const NOISE_SIGMA_SQUARED: f64 = 0.1;
const DELTA_TIME: f64 = 0.1;
const NAME: &str = "Circular Motion Position";
// R matrix (Measurement covariance)
// Low R -> High Confidence in the sensor
// High R -> Low Confidence in the sensor
const OBSERVATION_COVARIANCE: f64 = 0.03;
// P matrix (Initial state covariance)
// Low P -> High Confidence in the state
// High P -> Low Confidence in the state
const STATE_COVARIANCE: f64 = 0.01;
// Q matrix
// Low Q -> High Confidence in predicted state
// High Q -> Low Confidence in predicted state
const PROCESS_COVARIANCE: f64 = 0.07;

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v) as initial input
fn main() {
    let mut largest_error = 0.0;

    // 1. Observation matrix
    let h = Matrix2::identity();

    // 2. Observation noise COVARIANCE matrix
    let r = Matrix2::identity() * OBSERVATION_COVARIANCE;

    // 3. Initial state
    let x_initial = circular_motion(0.0);

    // 4. The initial state COVARIANCE matrix
    let p_initial = Matrix2::identity() * STATE_COVARIANCE;

    // Create a non-linear KF (EKF)
    let mut nonlinear_kalman = Kalman1M::new_ekf_with_input(step_fn, h, r, x_initial, p_initial);

    let mut t_history = Vec::new();
    let mut actual_state_history = Vec::new();
    let mut noisy_state_history = Vec::new();
    let mut predicted_state_history = Vec::new();

    for i in 0..(10.0f64 / DELTA_TIME) as i32 {
        println!("iteration:        {:?}", i);
        let time = (i as f64) * DELTA_TIME;

        let input = circular_motion_vel(time);
        // let input = Vector3::new(input.x, input.y, time);

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

    // Print the results
    println!("h (Observation matrix):       {:?}", h);
    println!("OBSERVATION_COVARIANCE:       {:?}", OBSERVATION_COVARIANCE);
    println!("STATE_COVARIANCE:             {:?}", STATE_COVARIANCE);
    println!("PROCESS_COVARIANCE:           {:?}", PROCESS_COVARIANCE);
    println!("Time Delta:                   {:?}", DELTA_TIME);
    println!("Sensor noise:                 {:?}", NOISE_SIGMA_SQUARED);
    println!("Largest error:                {:?}", largest_error);

    plot(
        NAME,
        t_history,
        actual_state_history,
        noisy_state_history,
        predicted_state_history,
    )
    .unwrap();
}

// Step function for circular motion
fn step_fn(state: Vector2<f64>, input: Vector2<f64>) -> StepReturn<f64, 2> {
    let vx: f64 = input.x;
    let vy = input.y;

    // Process Covariance Matrix
    let q_covariance = Matrix2::identity() * PROCESS_COVARIANCE;
    // let q_covariance = Matrix2::new(time, 0.0, 0.0, time);

    // Jacbobian
    let jacobian_vec = Vector2::new(vx, vy);
    // let jacobian_vec = Vector4::new(0.0, 0.0, 0.0, 0.0);

    // Diagonal Jacobian matrix. NOTE: experiments show less error with this Diagonal Jacobian matrix
    let jacobian = Matrix2::from_diagonal(&jacobian_vec);

    // // First Column Jacobian
    // let mut jacobian = Matrix2::zeros();
    // jacobian.column_mut(0).copy_from(&jacobian_vec);

    StepReturn {
        state,
        jacobian,
        covariance: q_covariance,
    }
}
