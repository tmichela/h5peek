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

    let mut samples: Vec<(usize, f64)> = Vec::new();
    for (idx, value) in series.iter().enumerate() {
        if value.is_finite() {
            samples.push((idx, *value));
        }
    }

    if samples.len() < 2 {
        None
    } else {
        let points = if samples.len() > max_points {
            let points_f64: Vec<(f64, f64)> = samples
                .iter()
                .map(|(idx, value)| (*idx as f64, *value))
                .collect();
            lttb(&points_f64, max_points)
        } else {
            samples
                .iter()
                .map(|(idx, value)| (*idx as f64, *value))
                .collect()
        };

        let mut f32_min = f32::INFINITY;
        let mut f32_max = f32::NEG_INFINITY;
        let mut f32_all_finite = true;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for (_, value) in &points {
            if *value < min {
                min = *value;
            }
            if *value > max {
                max = *value;
            }
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

        let mut points_out = Vec::with_capacity(points.len());
        if use_normalized {
            for (idx, value) in points {
                let y = if range == 0.0 {
                    0.5
                } else {
                    ((value - min) / range) as f32
                };
                points_out.push((idx as f32, y));
            }
        } else {
            for (idx, value) in points {
                points_out.push((idx as f32, value as f32));
            }
        }
        Some(points_out)
    }
}

fn lttb(points: &[(f64, f64)], threshold: usize) -> Vec<(f64, f64)> {
    let n = points.len();
    if threshold >= n || n <= 2 {
        return points.to_vec();
    }
    if threshold == 2 {
        return vec![points[0], points[n - 1]];
    }

    let every = (n - 2) as f64 / (threshold - 2) as f64;
    let mut sampled: Vec<(f64, f64)> = Vec::with_capacity(threshold);
    let mut a = 0usize;
    sampled.push(points[a]);

    for i in 0..(threshold - 2) {
        let mut avg_range_start = ((i as f64 + 1.0) * every).floor() as usize + 1;
        let mut avg_range_end = ((i as f64 + 2.0) * every).floor() as usize + 1;
        avg_range_end = avg_range_end.min(n);
        avg_range_start = avg_range_start.min(n - 1);

        let mut avg_x = 0.0;
        let mut avg_y = 0.0;
        let mut avg_count = 0usize;
        for idx in avg_range_start..avg_range_end {
            avg_x += points[idx].0;
            avg_y += points[idx].1;
            avg_count += 1;
        }
        if avg_count == 0 {
            avg_x = points[avg_range_start].0;
            avg_y = points[avg_range_start].1;
            avg_count = 1;
        }
        avg_x /= avg_count as f64;
        avg_y /= avg_count as f64;

        let mut range_start = ((i as f64) * every).floor() as usize + 1;
        let mut range_end = ((i as f64 + 1.0) * every).floor() as usize + 1;
        range_start = range_start.min(n - 2);
        range_end = range_end.min(n - 1);
        if range_end <= range_start {
            range_end = (range_start + 1).min(n - 1);
        }

        let mut max_area = -1.0;
        let mut next_a = range_start;
        for idx in range_start..range_end {
            let (ax, ay) = points[a];
            let (bx, by) = points[idx];
            let area = ((ax - avg_x) * (by - ay) - (ax - bx) * (avg_y - ay)).abs();
            if area > max_area {
                max_area = area;
                next_a = idx;
            }
        }

        sampled.push(points[next_a]);
        a = next_a;
    }

    sampled.push(points[n - 1]);
    sampled
}
