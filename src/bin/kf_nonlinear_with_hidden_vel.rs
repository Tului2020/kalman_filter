use clap::{command, Parser};
use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, circular_motion, circular_motion_vel, plot};
use nalgebra::{Matrix2, Vector2, Vector3};
use statrs::statistics::Statistics;

const NOISE_SIGMA_SQUARED: f64 = 0.1;
const DELTA_TIME: f64 = 0.1;
const NAME: &str = "Circular Motion Position and Hidden Velocity";
// R matrix (Measurement covariance)
// Low R -> High Confidence in the sensor
// High R -> Low Confidence in the sensor
const OBSERVATION_COVARIANCE: f64 = 0.08;
// P matrix (Initial state covariance)
// Low P -> High Confidence in the state
// High P -> Low Confidence in the state
const STATE_COVARIANCE: f64 = 0.1;
// Q matrix
// Low Q -> High Confidence in predicted state
// High Q -> Low Confidence in predicted state
const PROCESS_COVARIANCE: f64 = 0.3;
// Show step results
const SHOW_STEP_RESULTS: bool = false;

/// A simple CLI for passing arguments
#[derive(Parser, Debug)]
#[command(name = "myapp")]
struct Args {
    #[arg(short, long, default_value_t = NOISE_SIGMA_SQUARED)]
    noise_sigma_squared: f64,

    #[arg(short, long, default_value_t = DELTA_TIME)]
    delta_time: f64,
}

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v) as initial input and velocity (sx, sy) as the hidden state
fn main() {
    let args = Args::parse();

    let noise_sigma_squared = args.noise_sigma_squared;
    let dt = args.delta_time;

    // Error tracking
    let mut error_by_per = 0.0;
    let mut raw_error_by_per = 0.0;
    let mut largest_raw_error = 0.0;
    let mut errors: Vec<f64> = Vec::new();

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

    // Initialize variables needed for plotting
    let mut t_history = Vec::new();
    let mut actual_state_history = Vec::new();
    let mut noisy_state_history = Vec::new();
    let mut predicted_state_history = Vec::new();

    for i in 0..(10.0f64 / dt) as i32 {
        // time is i + 1 here because that's the time we're predicting for
        let time_current = (i as f64) * dt;
        let time_next = time_current + dt;

        let input = circular_motion_vel(time_current);
        let input = Vector3::new(input.x, input.y, dt);

        let actual_next_state = circular_motion(time_next);
        let noisy_next_state = add_noise(actual_next_state, noise_sigma_squared);

        // Predict the next state by giving it velocity and delta time
        let predicted_state = nonlinear_kalman.predict(input).clone();

        // Update EKF with new sensor measurement data
        nonlinear_kalman.update(noisy_next_state);

        // Save the history for plotting
        t_history.push(time_current);
        actual_state_history.push((actual_next_state.x, actual_next_state.y));
        noisy_state_history.push((noisy_next_state.x, noisy_next_state.y));
        predicted_state_history.push((predicted_state.x, predicted_state.y));

        // Get the error
        let raw_error = actual_next_state - predicted_state;
        let error = raw_error.component_div(&actual_next_state) * 100.0;
        for j in 0..raw_error.len() {
            let raw_e = raw_error[j].abs();

            if raw_e > largest_raw_error {
                largest_raw_error = raw_e;

                if !SHOW_STEP_RESULTS {
                    println!("iteration:        {:?}", i);
                    println!("current state:    {:?}", nonlinear_kalman.state());
                    println!("actual_next_state:{:?}", actual_next_state);
                    println!("noisy_next_state: {:?}", noisy_next_state);
                    println!("predicted_state   {:?}", predicted_state);
                    println!("raw error         {:?}\n", raw_error);
                }
            }

            let e = error[j].abs();

            errors.push(e);

            if e > error_by_per {
                raw_error_by_per = raw_error[j].abs();

                error_by_per = e;

                if !SHOW_STEP_RESULTS {
                    println!("iteration:        {:?}", i);
                    println!("current state:    {:?}", nonlinear_kalman.state());
                    println!("actual_next_state:{:?}", actual_next_state);
                    println!("noisy_next_state: {:?}", noisy_next_state);
                    println!("predicted_state   {:?}", predicted_state);
                    println!("percent error     {:?}\n", error);
                }
            }
        }

        if SHOW_STEP_RESULTS {
            println!("iteration:        {:?}", i);
            println!("current state:    {:?}", nonlinear_kalman.state());
            println!("actual_next_state:{:?}", actual_next_state);
            println!("noisy_next_state: {:?}", noisy_next_state);
            println!("predicted_state   {:?}", predicted_state);
            println!("error             {:?}\n", error);
        }
    }

    // Print the results
    println!("------------------------- Input -------------------------");
    println!("h (Observation matrix):       {:?}", h);
    println!("OBSERVATION_COVARIANCE:       {:?}", OBSERVATION_COVARIANCE);
    println!("STATE_COVARIANCE:             {:?}", STATE_COVARIANCE);
    println!("PROCESS_COVARIANCE:           {:?}", PROCESS_COVARIANCE);
    println!("Time Delta:                   {:?}", dt);
    println!("Sensor noise:                 {:?}", noise_sigma_squared);
    println!("\n------------------------- Error -------------------------");
    let error_mean = errors.clone().mean();
    let error_std = errors.std_dev();
    println!("Mean:                         {:.2}%", error_mean);
    println!("Std:                          {:.2}%", error_std);
    println!("Largest (percent):            {:.2}%", error_by_per);
    println!("Largest RAW (percent):        {:?}", raw_error_by_per);
    println!("Largest RAW:                  {:?}\n", largest_raw_error);

    // Plot the results
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
fn step_fn(state: Vector2<f64>, input: Vector3<f64>) -> StepReturn<f64, 2> {
    let dt = input.z;

    let vx = input.x;
    let vy = input.y;

    // Process Covariance Matrix
    let q_covariance = Matrix2::identity() * PROCESS_COVARIANCE;

    // Jacobian Vector
    let jacobian_vec = Vector2::new(vx, vy);

    // // Diagonal Jacobian matrix
    // let jacobian = Matrix4::from_diagonal(&jacobian_vec);

    // First Column Jacobian
    let mut jacobian = Matrix2::zeros();
    jacobian.column_mut(0).copy_from(&jacobian_vec);

    StepReturn {
        state: state + jacobian_vec * dt,
        jacobian,
        covariance: q_covariance,
    }
}
