use clap::{command, Parser};
use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, plot, polar::PolarCircular};
use nalgebra::{Matrix2, SMatrix, SVector, Vector1, Vector2};
use statrs::statistics::Statistics;

/// How much noise to add to the system
const NOISE_SIGMA_SQUARED: f64 = 0.1;

/// Time step
const DELTA_TIME: f64 = 0.1;

/// Name of the plot
const NAME: &str = "Circular Motion Position and Hidden Velocity (Polar)";

// R matrix (Measurement covariance)
// Low R -> High Confidence in the sensor
// High R -> Low Confidence in the sensor
const OBSERVATION_COVARIANCE: f64 = 0.15;

// P matrix (Initial state covariance)
// Low P -> High Confidence in the state
// High P -> Low Confidence in the state
const STATE_COVARIANCE: f64 = 0.01;

// Q matrix
// Low Q -> High Confidence in predicted state
// High Q -> Low Confidence in predicted state
const PROCESS_COVARIANCE: f64 = 0.04;

// Show step results
const SHOW_STEP_RESULTS: bool = false;

// Polar Parameters
const RADIUS: f64 = 1.0;
const RPM: f64 = 2.0;

/// type alias for state vector
type StateVector = SVector<f64, 3>;
type Covariance = SMatrix<f64, 3, 3>;

/// A simple CLI for passing arguments
#[derive(Parser, Debug)]
#[command(name = "ekf")]
struct Args {
    #[arg(short, long, default_value_t = NOISE_SIGMA_SQUARED)]
    noise_sigma_squared: f64,

    #[arg(short, long, default_value_t = DELTA_TIME)]
    delta_time: f64,
}

// Example of creating an Kalman filter for a circular motion
// Takes position (u, v) as initial input and velocity (sx, sy) as the hidden state
fn main() {
    // Parse the arguments
    let args = Args::parse();
    let noise_sigma_squared = args.noise_sigma_squared;
    let dt = args.delta_time;

    // Error tracking
    let mut largest_error = 0.0;
    let mut errors: Vec<(f64, f64)> = Vec::new();

    // Initialize state
    let polar_state = PolarCircular::new(RADIUS, RPM, NOISE_SIGMA_SQUARED);
    let x_initial = polar_state.initial_measurement();
    let mut state_history = State::new(x_initial.x, x_initial.y, 0.0);

    // Setup EKF
    let mut ekf = {
        // Observation matrix
        // | 1 0 0 0 |
        // | 0 1 0 0 |
        let mut h = SMatrix::<f64, 2, 3>::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;

        // Observation noise COVARIANCE matrix
        let r = Matrix2::identity() * OBSERVATION_COVARIANCE;

        // The initial state COVARIANCE matrix
        // | X 0 0 0 |
        // | 0 X 0 0 |
        // | 0 0 X 0 |
        // | 0 0 0 X |
        let p_initial: Covariance = Covariance::identity() * STATE_COVARIANCE;

        // Create a non-linear KF (EKF)
        Kalman1M::new_ekf_with_input(step_fn, h, r, x_initial, p_initial)
    };

    // Initialize variables needed for plotting
    let mut t_history = Vec::new();
    let mut actual_state_history = Vec::new();
    let mut measured_state_history = Vec::new();
    let mut predicted_state_history = Vec::new();

    // Iterate over time
    for i in 0..(10.0f64 / dt) as i32 {
        // Define the currrent time
        let time_current = (i as f64) * dt;

        // Acquire target position
        let actual_current_state = polar_state.state(time_current);
        let measured_current_state = add_noise(actual_current_state, noise_sigma_squared);
        let measured_current_state =
            Vector2::new(measured_current_state.x, measured_current_state.y);

        if i > 0 {
            // If predicted data is available,
            // 1. find error between actual and predicted
            let mut new_large = false;

            let (predicted_x, predicted_y) = predicted_state_history[i as usize];
            let error_x: f64 = predicted_x - actual_current_state.x;
            let error_y: f64 = predicted_y - actual_current_state.y;
            if error_x.abs() > largest_error {
                new_large = true;
                largest_error = error_x.abs();
            }
            if error_y.abs() > largest_error {
                new_large = true;
                largest_error = error_y.abs();
            }
            errors.push((error_x.powi(2), error_y.powi(2)));
            if new_large && !SHOW_STEP_RESULTS {
                let (p_x, p_y) = predicted_state_history[i as usize];
                let (e_x, e_y) = errors[i as usize];
                println!("iteration:                {:.4?}", i);
                println!("current state:            {:.4?}", ekf.state());
                println!("actual_state:             {:.4?}", actual_current_state);
                println!("measured_state:           {:.4?}", measured_current_state);
                println!("predicted_state           [[{:.4?}, {:.4?}]]", p_x, p_y);
                println!("error                     [[{:.4?}, {:.4?}]]\n", e_x, e_y);
            }

            // 2. run correction/update
            let current_state = ekf.update(measured_current_state);

            if new_large && !SHOW_STEP_RESULTS {
                println!("updated current state:    {:?}\n", current_state);
            }

            // 3. Update "state_history" with new state
            state_history.update(current_state.x, current_state.y, time_current);
        } else {
            // Otherwise set intial predicted state to the measured state
            let current_state = ekf.update(measured_current_state);
            predicted_state_history.push((current_state.x, current_state.y));
            errors.push((0.0, 0.0));
        }

        if SHOW_STEP_RESULTS {
            let (p_x, p_y) = predicted_state_history[i as usize];
            let (e_x, e_y) = errors[i as usize];
            println!("iteration:                {:.4?}", i);
            println!("current state:            {:.4?}", ekf.state());
            println!("actual_state:             {:.4?}", actual_current_state);
            println!("measured_state:           {:.4?}", measured_current_state);
            println!("predicted_state           [[{:.4?}, {:.4?}]]", p_x, p_y);
            println!("error                     [[{:.4?}, {:.4?}]]\n", e_x, e_y);
        }

        // Predict next state
        let input = Vector1::new(dt);
        let predicted_state = ekf.predict(input).clone();
        predicted_state_history.push((predicted_state.x, predicted_state.y));

        // Save the history for plotting
        t_history.push(time_current);
        actual_state_history.push((actual_current_state.x, actual_current_state.y));
        measured_state_history.push((measured_current_state.x, measured_current_state.y));
    }

    // Print the results
    println!("------------------------- Input -------------------------");
    println!("OBSERVATION_COVARIANCE:       {:?}", OBSERVATION_COVARIANCE);
    println!("STATE_COVARIANCE:             {:?}", STATE_COVARIANCE);
    println!("PROCESS_COVARIANCE:           {:?}", PROCESS_COVARIANCE);
    println!("Time Delta:                   {:?}", dt);
    println!("Sensor noise:                 {:?}", noise_sigma_squared);
    println!("\n------------------------- Error -------------------------");
    let raw_mse = errors.into_iter().flat_map(|(a, b)| vec![a, b]).mean();
    println!("Largest:                      {:?}", largest_error);
    println!("MSE:                          {:?}", raw_mse);
    println!("RMSE:                         {:?}\n", raw_mse.sqrt());

    // Plot the results
    plot(
        NAME,
        t_history,
        actual_state_history,
        measured_state_history,
        predicted_state_history,
    )
    .unwrap();
}

// Step function for circular motion
fn step_fn(state: StateVector, input: Vector1<f64>) -> StepReturn<f64, 3> {
    // Process Covariance Matrix
    let q_covariance: Covariance = Covariance::identity() * PROCESS_COVARIANCE;

    // Prediction Jacobian Vector (F or A)
    // | 1 0 0  |
    // | 0 1 dt |
    // | 0 0 1  |
    let mut jacobian = SMatrix::<f64, 3, 3>::identity();
    jacobian[(1, 2)] = input.x;

    StepReturn {
        state: jacobian * state,
        jacobian,
        covariance: q_covariance,
    }
}

/// State struct to keep track of the current, previous, and previous previous state
#[derive(Clone)]
pub struct State {
    current_x: f64,
    current_y: f64,
    current_t: f64,
    previous_x: f64,
    previous_y: f64,
    previous_t: f64,
}

impl State {
    /// Create a new state
    pub fn new(x: f64, y: f64, timestamp: f64) -> Self {
        Self {
            current_x: x,
            current_y: y,
            current_t: timestamp,
            previous_x: x,
            previous_y: y,
            previous_t: timestamp - 1.0,
        }
    }

    /// Update the state
    pub fn update(&mut self, x: f64, y: f64, timestamp: f64) {
        self.previous_x = self.current_x;
        self.previous_y = self.current_y;
        self.previous_t = self.current_t;
        self.current_x = x;
        self.current_y = y;
        self.current_t = timestamp;
    }

    /// Get the current velocty
    pub fn velocity(&self) -> Vector2<f64> {
        let dt = self.current_t - self.previous_t;
        let vx = (self.current_x - self.previous_x) / dt;
        let vy = (self.current_y - self.previous_y) / dt;
        Vector2::new(vx, vy)
    }
}
