use kfilter::{
    measurement::LinearMeasurement,
    system::{NonLinearSystem, StepFunction},
    Kalman1M, KalmanPredictInput,
};
use nalgebra::{SMatrix, SVector};

pub struct Tracking<
    // Size of the State Vector
    const S: usize,
    // Size of the Measurement Vector
    const M: usize,
    // Size of the Input Vector
    const U: usize,
> {
    /// The Extended Kalman Filter
    ekf: Kalman1M<f64, S, U, M, NonLinearSystem<f64, S, U>, LinearMeasurement<f64, S, M>>,
    /// The initial state covariance (P)
    /// Low P -> High Confidence in the state
    /// High P -> Low Confidence in the state
    initial_state_covariance: f64,
    /// The measurement covariance (R)
    /// Low R -> High Confidence in the sensor
    /// High R -> Low Confidence in the sensor
    measurement_covariance: f64,
    /// The function that defines the system state transition, used when resetting the filter
    step_fn: StepFunction<f64, S, U>,
}

impl<const S: usize, const M: usize, const U: usize> Tracking<S, M, U> {
    /// Initializes a new Extended Kalman Filter
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The initial state of the system
    /// * `initial_state_covariance` - The initial state covariance (P Covariance)
    /// * `measurement_covariance` - The measurement/observation covariance (R Covariance)
    /// * `step_fn` - The function that defines the system state transition
    pub fn new(
        initial_state: SVector<f64, S>,
        initial_state_covariance: f64,
        measurement_covariance: f64,
        step_fn: StepFunction<f64, S, U>,
    ) -> Self {
        // Initial State Covariance Matrix
        let p_initial = SMatrix::<f64, S, S>::identity() * initial_state_covariance;
        // Observation/Measurement Covariance Matrix
        let r = SMatrix::<f64, M, M>::identity() * measurement_covariance;

        // Observation/Measurement Matrix
        let h = SMatrix::<f64, M, S>::identity();

        Self {
            ekf: Kalman1M::new_ekf_with_input(step_fn, h, r, initial_state, p_initial),
            initial_state_covariance,
            measurement_covariance,
            step_fn,
        }
    }

    /// Updates current state with observed state and predicts the next state of the system
    ///
    /// # Arguments
    ///
    /// * `observed_state` - The observed state of the system
    /// * `input` - The input to the system, NOTE: this is the same 'input' as step_fn
    pub fn predict(
        &mut self,
        observed_state: SVector<f64, M>,
        input: SVector<f64, U>,
    ) -> &SVector<f64, S> {
        self.ekf.update(observed_state);
        self.ekf.predict(input)
    }

    /// Resets the filter with a new initial state
    /// This can be used when we lose track of the object
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The new initial state of the system
    pub fn reset(&mut self, initial_state: SVector<f64, S>) {
        let p_initial = SMatrix::<f64, S, S>::identity() * self.initial_state_covariance;
        let r = SMatrix::<f64, M, M>::identity() * self.measurement_covariance;
        let h = SMatrix::<f64, M, S>::identity();
        self.ekf = Kalman1M::new_ekf_with_input(self.step_fn, h, r, initial_state, p_initial);
    }
}
