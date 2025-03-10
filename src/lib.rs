use nalgebra::{SVector, Vector2, Vector4, Vector6};
use plotters::prelude::*;

pub fn circular_motion(time: f64) -> SVector<f64, 2> {
    Vector2::new(time.cos(), time.sin())
}

pub fn circular_motion_vel(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.sin(), time.cos())
}

pub fn circular_motion_acc(time: f64) -> SVector<f64, 2> {
    Vector2::new(-time.cos(), -time.sin())
}

pub fn circular_motion_with_vel(time: f64) -> SVector<f64, 4> {
    let p_x = time.cos();
    let p_y = time.sin();
    let v_x = -time.sin();
    let v_y = time.cos();
    Vector4::new(p_x, p_y, v_x, v_y)
}

pub fn circular_motion_with_vel_acc(time: f64) -> SVector<f64, 6> {
    let p_x = time.cos();
    let p_y = time.sin();
    let v_x = -time.sin();
    let v_y = time.cos();
    let a_x = -time.cos();
    let a_y = -time.sin();
    Vector6::new(p_x, p_y, v_x, v_y, a_x, a_y)
}

fn get_random_noise(sigma_squared: f64) -> f64 {
    sigma_squared * (2.0 * rand::random::<f64>() - 1.0)
}

pub fn add_noise<const N: usize>(state: SVector<f64, N>, noise: f64) -> SVector<f64, N> {
    let mut noisy_vec: Vec<f64> = Vec::with_capacity(N + 1);
    for _ in 0..N {
        noisy_vec.push(get_random_noise(noise));
    }
    state + SVector::from_vec(noisy_vec)
}

pub fn plot(
    name: &str,
    t_history: Vec<f64>,
    actual_state_history: Vec<(f64, f64)>,
    measured_state_history: Vec<(f64, f64)>,
    predicted_state_history: Vec<(f64, f64)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path_name = format!("./graphs/{name}.png");

    let root = BitMapBackend::new(&path_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let t_min = t_history[0];
    let t_max = t_history[t_history.len() - 1];

    let mut chart = ChartBuilder::on(&root)
        .caption(&name, ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(t_min..t_max, -1.5..1.5)?;

    chart.configure_mesh().draw()?;

    // Draw multiple series on the same plot
    // ACTUAL state
    let x_history_actual: Vec<(f64, f64)> = t_history
        .iter()
        .zip(actual_state_history.iter())
        .map(&|(&t, &(x, _))| (t, x))
        .collect();
    let y_history_actual: Vec<(f64, f64)> = t_history
        .iter()
        .zip(actual_state_history.iter())
        .map(&|(&t, &(_, y))| (t, y))
        .collect();
    chart
        .draw_series(LineSeries::new(x_history_actual, &BLACK))?
        .label("actual_state")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart.draw_series(LineSeries::new(y_history_actual, &BLACK))?;

    // PREDICTED state
    let x_history_predicted: Vec<(f64, f64)> = t_history
        .iter()
        .zip(predicted_state_history.iter())
        .map(&|(&t, &(x, _))| (t, x))
        .collect();
    let y_history_predicted: Vec<(f64, f64)> = t_history
        .iter()
        .zip(predicted_state_history.iter())
        .map(&|(&t, &(_, y))| (t, y))
        .collect();
    chart
        .draw_series(LineSeries::new(x_history_predicted, &RED))?
        .label("predicted_state")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart.draw_series(LineSeries::new(y_history_predicted, &RED))?;

    // MEASURED state
    let x_history_measured: Vec<(f64, f64)> = t_history
        .iter()
        .zip(measured_state_history.iter())
        .map(&|(&t, &(x, _))| (t, x))
        .collect();
    let y_history_measured: Vec<(f64, f64)> = t_history
        .iter()
        .zip(measured_state_history.iter())
        .map(&|(&t, &(_, y))| (t, y))
        .collect();
    chart
        .draw_series(LineSeries::new(x_history_measured, &BLUE))?
        .label("measured_state")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart.draw_series(LineSeries::new(y_history_measured, &BLUE))?;

    // Configure the legend
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.8))
        .draw()?;

    root.present()?;
    println!("Graph saved as {path_name}");
    Ok(())
}
