use clap::{command, Parser};
use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, circular_motion, plot};
use nalgebra::{Matrix2, Vector2, Vector3};
use statrs::statistics::Statistics;

/// How much noise to add to the system
const NOISE_SIGMA_SQUARED: f64 = 0.1;

/// Time step
const DELTA_TIME: f64 = 0.1;

/// Name of the plot
const NAME: &str = "Circular Motion Position and Hidden Velocity";

// R matrix (Measurement covariance)
// Low R -> High Confidence in the sensor
// High R -> Low Confidence in the sensor
const OBSERVATION_COVARIANCE: f64 = 0.1;

// P matrix (Initial state covariance)
// Low P -> High Confidence in the state
// High P -> Low Confidence in the state
const STATE_COVARIANCE: f64 = 0.04;

// Q matrix
// Low Q -> High Confidence in predicted state
// High Q -> Low Confidence in predicted state
const PROCESS_COVARIANCE: f64 = 0.04;

// Show step results
const SHOW_STEP_RESULTS: bool = false;

/// type alias for state vector
type StateVector = Vector2<f64>;
type Covariance = Matrix2<f64>;

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
    let x_initial = circular_motion(0.0);
    let mut state = State::new(x_initial.x, x_initial.y, 0.0);

    // Setup EKF
    let mut ekf = {
        // Observation matrix
        let h = Matrix2::identity();
        // Observation noise COVARIANCE matrix
        let r: Covariance = Matrix2::identity() * OBSERVATION_COVARIANCE;
        // The initial state COVARIANCE matrix
        let p_initial: Covariance = Matrix2::identity() * STATE_COVARIANCE;
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
        let actual_current_state = circular_motion(time_current);
        let measured_current_state = add_noise(actual_current_state, noise_sigma_squared);

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
                println!("iteration:                {:?}", i);
                println!("current state:            {:?}", ekf.state());
                println!("actual_state:             {:?}", actual_current_state);
                println!("measured_state:           {:?}", measured_current_state);
                println!("predicted_state           [[{p_x}, {p_y}]]");
                println!("error                     [[{e_x}, {e_y}]]\n");
            }

            // 2. run correction/update
            let current_state = ekf.update(measured_current_state);

            // 3. Update "state" with new state
            state.update(current_state.x, current_state.y, time_current);
        } else {
            // Otherwise set intial predicted state to the measured state
            predicted_state_history.push((measured_current_state.x, measured_current_state.y));
            errors.push((0.0, 0.0));
        }

        if SHOW_STEP_RESULTS {
            let (p_x, p_y) = predicted_state_history[i as usize];
            let (e_x, e_y) = errors[i as usize];
            println!("iteration:                {:?}", i);
            println!("current state:            {:?}", ekf.state());
            println!("actual__state:            {:?}", actual_current_state);
            println!("measured_state:           {:?}", measured_current_state);
            println!("predicted_state           [[{p_x}, {p_y}]]");
            println!("error                     [[{e_x}, {e_y}]]\n");
        }

        // Predict next state
        let velocity = state.velocity();
        let input = Vector3::new(velocity.x, velocity.y, dt);
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
fn step_fn(state: StateVector, input: Vector3<f64>) -> StepReturn<f64, 2> {
    let dt = input.z;

    let vx = input.x;
    let vy = input.y;

    // Process Covariance Matrix
    let q_covariance = Matrix2::identity() * PROCESS_COVARIANCE;

    // Jacobian Vector
    let jacobian_vec: StateVector = Vector2::new(vx, vy);

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

    /// Get the current state
    pub fn current(&self) -> StateVector {
        Vector2::new(self.current_x, self.current_y)
    }

    /// Get the previous state
    pub fn previous(&self) -> StateVector {
        Vector2::new(self.previous_x, self.previous_y)
    }
}

impl From<State> for StateVector {
    fn from(state: State) -> Self {
        Vector2::new(state.current_x, state.current_y)
    }
}
