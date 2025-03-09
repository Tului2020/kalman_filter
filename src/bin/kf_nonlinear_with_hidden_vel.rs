use std::collections::VecDeque;

use clap::{command, Parser};
use kfilter::{kalman::Kalman1M, system::StepReturn, KalmanFilter, KalmanPredictInput};
use kfilter2::{add_noise, circular_motion, plot};
use nalgebra::{Matrix2, SVector, Vector2};
use statrs::statistics::Statistics;

/// How much noise to add to the system
const NOISE_SIGMA_SQUARED: f64 = 0.10;

/// Time step
const DELTA_TIME: f64 = 0.1;

/// Name of the plot
const NAME: &str = "Circular Motion Position and Hidden Velocity";

/// R matrix (Measurement covariance)
/// Low R -> High Confidence in the sensor
/// High R -> Low Confidence in the sensor
const OBSERVATION_COVARIANCE_VALUE: f64 = 0.2;

/// P matrix (Initial state covariance)
/// Low P -> High Confidence in the state
/// High P -> Low Confidence in the state
const STATE_COVARIANCE_VALUE: f64 = 0.04;

/// Q matrix
/// Low Q -> High Confidence in predicted state
/// High Q -> Low Confidence in predicted state
const PROCESS_COVARIANCE_INITIAL_VALUE: f64 = 0.1;
const PROCESS_COVARIANCE_SIZE: usize = 5;
const PROCESS_COVARIANCE_ALPHA: f64 = 0.001;

/// Show step results
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

/// Example of creating an Kalman filter for a circular motion
/// Takes position (u, v) as initial input and velocity (sx, sy) as the hidden state
fn main() {
    // Parse arguments
    let args = Args::parse();
    let noise_sigma_squared = args.noise_sigma_squared;
    let dt = args.delta_time;

    // Error tracking
    let mut largest_error = 0.0;
    let mut errors = Vec::new();

    // Initialize variables needed for plotting
    let mut t_history = Vec::new();
    let mut actual_state_history = Vec::new();
    let mut measured_state_history = Vec::new();
    let mut predicted_state_history = Vec::new();

    // Extended Kalman Filter (EKF) setup
    // 1. Observation matrix
    let h = Matrix2::identity();

    // 2. Observation noise COVARIANCE matrix
    let r: Covariance = Matrix2::identity() * OBSERVATION_COVARIANCE_VALUE;

    // 3. Initial state
    let mut state: State = circular_motion(0.0).into();
    let initial_state = state.clone().into();

    // 4. The initial state COVARIANCE matrix
    let p_initial: Covariance = Matrix2::identity() * STATE_COVARIANCE_VALUE;

    // 5. Innovation vectors
    // They are used for calculating the process covariance (Q) at each step
    let mut q_covariance: Covariance = Matrix2::identity() * PROCESS_COVARIANCE_INITIAL_VALUE;
    println!("q_covariance:             {:?}\n", q_covariance);

    // Create an EKF instance
    let mut ekf = Kalman1M::new_ekf_with_input(step_fn, h, r, initial_state, p_initial);

    // Step loop starts
    for i in 0..(10.0f64 / dt) as i32 {
        // time is i + 1 here because that's the time we're predicting for
        let time_current = (i as f64) * dt;
        let time_next = time_current + dt;

        // Input contains the following:
        // 1. Velocity in x direction
        // 2. Velocity in y direction
        // 3. Delta time
        // 4-7. Current Process Covariance Matrix
        let input = state.velocity(dt);
        let empirical_innovation = state.empirical_innovation_covariance();
        let theoretical_innovation = ekf.covariance().clone() + r;
        q_covariance += PROCESS_COVARIANCE_ALPHA * (empirical_innovation - theoretical_innovation);
        println!("q_covariance:             {:?}\n", q_covariance);

        let input = [
            input.x,
            input.y,
            dt,
            q_covariance[(0, 0)],
            q_covariance[(0, 1)],
            q_covariance[(1, 0)],
            q_covariance[(1, 1)],
        ]
        .into();

        // Predict the next state by giving it velocity and delta time
        let predicted_state = ekf.predict(input).clone();
        state.update(predicted_state.x, predicted_state.y);

        // Measure next step
        let actual_next_state = circular_motion(time_next);
        let measured_next_state = add_noise(actual_next_state, noise_sigma_squared);

        // Get innovation vector
        let innovation_vector = measured_next_state - predicted_state;
        state.add_innovation(innovation_vector);

        // Update EKF with new sensor measurement data
        ekf.update(measured_next_state);

        // Save the history for plotting
        t_history.push(time_current);
        actual_state_history.push((actual_next_state.x, actual_next_state.y));
        measured_state_history.push((measured_next_state.x, measured_next_state.y));
        predicted_state_history.push((predicted_state.x, predicted_state.y));

        // Get the error
        let raw_error = actual_next_state - predicted_state;
        errors.push(raw_error.x.powi(2));
        errors.push(raw_error.y.powi(2));

        let error = raw_error.component_div(&actual_next_state) * 100.0;
        for j in 0..raw_error.len() {
            let raw_e = raw_error[j].abs();

            if raw_e > largest_error {
                largest_error = raw_e;

                if !SHOW_STEP_RESULTS {
                    println!("iteration:            {:?}", i);
                    println!("current state:        {:?}", ekf.state());
                    println!("actual_next_state:    {:?}", actual_next_state);
                    println!("measured_next_state:  {:?}", measured_next_state);
                    println!("predicted_state       {:?}", predicted_state);
                    println!("raw error             {:?}\n", raw_error);
                }
            }
        }

        if SHOW_STEP_RESULTS {
            println!("iteration:            {:?}", i);
            println!("current state:        {:?}", ekf.state());
            println!("actual_next_state:    {:?}", actual_next_state);
            println!("measured_next_state:  {:?}", measured_next_state);
            println!("predicted_state       {:?}", predicted_state);
            println!("error                 {:?}\n", error);
        }
    }

    // Print the results
    println!("------------------------- Input -------------------------");
    println!("h (Observation matrix):           {:?}", h);
    println!(
        "OBSERVATION_COVARIANCE_VALUE:     {:?}",
        OBSERVATION_COVARIANCE_VALUE
    );
    println!(
        "STATE_COVARIANCE_VALUE:           {:?}",
        STATE_COVARIANCE_VALUE
    );
    println!(
        "PROCESS_COVARIANCE_INITIAL_VALUE: {:?}",
        PROCESS_COVARIANCE_INITIAL_VALUE
    );
    println!("Time Delta:                       {:?}", dt);
    println!(
        "Sensor noise:                     {:?}",
        noise_sigma_squared
    );
    println!("\n------------------------- Error -------------------------");
    let raw_mse = errors.mean();
    println!("Largest:                          {:?}", largest_error);
    println!("MSE:                              {:?}", raw_mse);
    println!("RMSE:                             {:?}\n", raw_mse.sqrt());

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
fn step_fn(state: StateVector, input: SVector<f64, 7>) -> StepReturn<f64, 2> {
    let vx = input[0];
    let vy = input[1];
    let dt = input[2];

    // Process Covariance Matrix
    let q_covariance: Covariance = Matrix2::new(input[3], input[4], input[5], input[6]);

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

// /// Calculate the process covariance matrix
// fn calculate_process_covariance(
//     old_process_covariance: Covariance,
//     alpha: f64,
//     innovation_vectors: &mut StateVector,
//     observation_covariance: &Covariance,
// ) -> Covariance {
//     Covariance
// }

/// State struct to keep track of the current, previous, and previous previous state
#[derive(Clone)]
pub struct State {
    current_x: f64,
    current_y: f64,
    previous_x: f64,
    previous_y: f64,
    innovations: VecDeque<StateVector>,
}

impl State {
    /// Create a new state
    pub fn new(x: f64, y: f64) -> Self {
        Self {
            current_x: x,
            current_y: y,
            previous_x: x,
            previous_y: y,
            innovations: VecDeque::new(),
        }
    }

    /// Update the state
    pub fn update(&mut self, x: f64, y: f64) {
        self.previous_x = self.current_x;
        self.previous_y = self.current_y;
        self.current_x = x;
        self.current_y = y;
    }

    /// Get the current velocty
    pub fn velocity(&self, dt: f64) -> Vector2<f64> {
        let vx = (self.current_x - self.previous_x) / dt;
        let vy = (self.current_y - self.previous_y) / dt;
        Vector2::new(vx, vy)
    }

    /// Add innovation
    pub fn add_innovation(&mut self, innovation: StateVector) {
        if self.innovations.len() == PROCESS_COVARIANCE_SIZE {
            self.innovations.pop_front();
        }
        self.innovations.push_back(innovation);
    }

    /// Calculates empirical innovation covariance
    pub fn empirical_innovation_covariance(&self) -> Covariance {
        let mut sum = Matrix2::zeros();
        for innovation in &self.innovations {
            sum += innovation * innovation.transpose();
        }
        sum / self.innovations.len().max(1) as f64
    }
}

impl From<State> for StateVector {
    fn from(state: State) -> Self {
        Vector2::new(state.current_x, state.current_y)
    }
}

impl Into<State> for StateVector {
    fn into(self) -> State {
        State::new(self.x, self.y)
    }
}
