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

pub fn add_noise(state: SVector<f64, 2>, noise: f64) -> SVector<f64, 2> {
    state + Vector2::new(rand::random::<f64>() * noise, rand::random::<f64>() * noise)
}

pub fn add_noise4(state: SVector<f64, 4>, noise: f64) -> SVector<f64, 4> {
    state
        + Vector4::new(
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
        )
}

pub fn add_noise6(state: SVector<f64, 6>, noise: f64) -> SVector<f64, 6> {
    state
        + Vector6::new(
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
            rand::random::<f64>() * noise,
        )
}

pub fn plot(
    name: &str,
    t_history: Vec<f64>,
    actual_state_history: Vec<(f64, f64)>,
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
        .draw_series(LineSeries::new(x_history_actual, &RED))?
        .label("actual_state_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(LineSeries::new(y_history_actual, &RED))?
        .label("actual_state_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(LineSeries::new(x_history_predicted, &BLACK))?
        .label("predicted_state_x")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
    chart
        .draw_series(LineSeries::new(y_history_predicted, &BLACK))?
        .label("predicted_state_y")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

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
