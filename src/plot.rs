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
            let ratio = points_f64.len().div_ceil(max_points);
            fpcs(&points_f64, ratio.max(1))
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

fn fpcs(points: &[(f64, f64)], ratio: usize) -> Vec<(f64, f64)> {
    let n = points.len();
    if n <= 2 || ratio <= 1 {
        return points.to_vec();
    }

    let mut output: Vec<(f64, f64)> = Vec::new();
    let mut potential: Option<(f64, f64)> = None;
    let mut previous_min_flag: i8 = -1;
    let mut counter: usize = 0;

    let first = points[0];
    let mut max_point = first;
    let mut min_point = first;
    output.push(first);
    counter += 1;

    for &p in points.iter().skip(1) {
        counter += 1;
        if p.1 >= max_point.1 {
            max_point = p;
        } else if p.1 < min_point.1 {
            min_point = p;
        }

        if counter >= ratio {
            if min_point.0 < max_point.0 {
                if previous_min_flag == 1 {
                    if let Some(pp) = potential {
                        if pp.0 != min_point.0 || pp.1 != min_point.1 {
                            output.push(pp);
                        }
                    }
                }
                output.push(min_point);
                potential = Some(max_point);
                min_point = max_point;
                previous_min_flag = 1;
            } else {
                if previous_min_flag == 0 {
                    if let Some(pp) = potential {
                        if pp.0 != max_point.0 || pp.1 != max_point.1 {
                            output.push(pp);
                        }
                    }
                }
                output.push(max_point);
                potential = Some(min_point);
                max_point = min_point;
                previous_min_flag = 0;
            }
            counter = 0;
        }
    }

    let last = points[n - 1];
    if output
        .last()
        .map(|p| p.0 != last.0 || p.1 != last.1)
        .unwrap_or(true)
    {
        output.push(last);
    }
    output
}
