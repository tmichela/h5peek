use textplots::{Chart, Plot, Shape};

pub trait PlotBackend {
    fn render_1d(&self, series: &[f64]) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct TextplotsBackend {
    width: u32,
    height: u32,
    max_points: usize,
}

impl TextplotsBackend {
    pub fn new(width: u32, height: u32, max_points: usize) -> Self {
        Self {
            width: width.max(32),
            height: height.max(3),
            max_points: max_points.max(2),
        }
    }
}

impl Default for TextplotsBackend {
    fn default() -> Self {
        Self::new(120, 30, 1000)
    }
}

impl PlotBackend for TextplotsBackend {
    fn render_1d(&self, series: &[f64]) -> Option<String> {
        if series.len() < 2 {
            return None;
        }

        let points = downsample_points(series, self.max_points)?;
        if points.len() < 2 {
            return None;
        }

        let xmin = points.first()?.0;
        let xmax = points.last()?.0;
        let mut chart = Chart::new(self.width, self.height, xmin, xmax);
        let shape = Shape::Lines(points.as_slice());
        let chart_ref = chart.lineplot(&shape);
        chart_ref.axis();
        chart_ref.figures();
        Some(chart_ref.frame())
    }
}

pub fn default_backend() -> TextplotsBackend {
    TextplotsBackend::default()
}

fn downsample_points(series: &[f64], max_points: usize) -> Option<Vec<(f32, f32)>> {
    let len = series.len();
    if len == 0 {
        return None;
    }

    let stride = (len + max_points - 1) / max_points;
    let stride = stride.max(1);

    let mut samples: Vec<(usize, f64)> = Vec::new();
    for idx in (0..len).step_by(stride) {
        let value = series[idx];
        if value.is_finite() {
            samples.push((idx, value));
        }
    }

    let last_idx = len - 1;
    let needs_last = match samples.last() {
        Some((x, _)) => *x != last_idx,
        None => true,
    };
    if needs_last {
        let value = series[last_idx];
        if value.is_finite() {
            samples.push((last_idx, value));
        }
    }

    if samples.len() < 2 {
        None
    } else {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for (_, value) in &samples {
            if *value < min {
                min = *value;
            }
            if *value > max {
                max = *value;
            }
        }

        let mut f32_min = f32::INFINITY;
        let mut f32_max = f32::NEG_INFINITY;
        let mut f32_all_finite = true;
        for (_, value) in &samples {
            let value_f32 = *value as f32;
            if !value_f32.is_finite() {
                f32_all_finite = false;
                break;
            }
            if value_f32 < f32_min {
                f32_min = value_f32;
            }
            if value_f32 > f32_max {
                f32_max = value_f32;
            }
        }

        let use_normalized = !f32_all_finite || f32_min == f32_max;
        let range = max - min;

        let mut points = Vec::with_capacity(samples.len());
        if use_normalized {
            for (idx, value) in samples {
                let y = if range == 0.0 {
                    0.5
                } else {
                    ((value - min) / range) as f32
                };
                points.push((idx as f32, y));
            }
        } else {
            for (idx, value) in samples {
                points.push((idx as f32, value as f32));
            }
        }
        Some(points)
    }
}
