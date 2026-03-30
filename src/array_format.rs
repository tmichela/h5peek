use ndarray::{ArrayD, ArrayViewD, Axis};

#[derive(Clone, Copy, Debug)]
pub struct EllipsisConfig {
    pub max_elems: usize,
    pub edge: usize,
}

impl EllipsisConfig {
    pub const fn new(max_elems: usize, edge: usize) -> Self {
        Self { max_elems, edge }
    }
}

pub fn format_debug_with_ellipsis<T: std::fmt::Debug>(
    arr: &ArrayD<T>,
    cfg: EllipsisConfig,
) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    if !needs_ellipsis(arr.len(), arr.shape(), cfg) {
        return format!("{:?}", arr);
    }
    format_array_view_with_ellipsis(arr.view(), cfg.edge)
}

pub fn format_string_array_with_ellipsis(
    arr: &ArrayD<String>,
    cfg: EllipsisConfig,
    quote_strings: bool,
) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    if !needs_ellipsis(arr.len(), arr.shape(), cfg) {
        return format_array_view_display(arr.view(), quote_strings);
    }
    format_array_view_with_ellipsis_display(arr.view(), cfg.edge, quote_strings)
}

pub fn format_string_array_full(arr: &ArrayD<String>, quote_strings: bool) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    format_array_view_display(arr.view(), quote_strings)
}

fn needs_ellipsis(len: usize, shape: &[usize], cfg: EllipsisConfig) -> bool {
    len > cfg.max_elems || shape.iter().any(|&d| d > cfg.edge * 2)
}

#[derive(Clone, Copy)]
enum AxisItem {
    Index(usize),
    Ellipsis,
}

fn axis_indices(len: usize, edge: usize) -> Vec<AxisItem> {
    if len <= edge * 2 {
        return (0..len).map(AxisItem::Index).collect();
    }
    let mut out = Vec::with_capacity(edge * 2 + 1);
    for i in 0..edge {
        out.push(AxisItem::Index(i));
    }
    out.push(AxisItem::Ellipsis);
    for i in (len - edge)..len {
        out.push(AxisItem::Index(i));
    }
    out
}

fn format_array_view_with_ellipsis<T: std::fmt::Debug>(view: ArrayViewD<T>, edge: usize) -> String {
    match view.ndim() {
        0 => format!("{:?}", view.first().unwrap()),
        1 => {
            let mut parts = Vec::new();
            for item in axis_indices(view.shape()[0], edge) {
                match item {
                    AxisItem::Index(i) => {
                        let v = view.index_axis(Axis(0), i);
                        parts.push(format!("{:?}", v.first().unwrap()));
                    }
                    AxisItem::Ellipsis => parts.push("...".to_string()),
                }
            }
            format!("[{}]", parts.join(", "))
        }
        _ => {
            let mut parts = Vec::new();
            for item in axis_indices(view.shape()[0], edge) {
                match item {
                    AxisItem::Index(i) => {
                        let v = view.index_axis(Axis(0), i);
                        parts.push(format_array_view_with_ellipsis(v.into_dyn(), edge));
                    }
                    AxisItem::Ellipsis => parts.push("...".to_string()),
                }
            }
            format!("[{}]", parts.join(", "))
        }
    }
}

fn format_array_view_with_ellipsis_display(
    view: ArrayViewD<String>,
    edge: usize,
    quote_strings: bool,
) -> String {
    match view.ndim() {
        0 => view
            .first()
            .map(|v| format_string_element(v, quote_strings))
            .unwrap_or_default(),
        1 => {
            let mut parts = Vec::new();
            for item in axis_indices(view.shape()[0], edge) {
                match item {
                    AxisItem::Index(i) => {
                        let v = view.index_axis(Axis(0), i);
                        parts.push(
                            v.first()
                                .map(|s| format_string_element(s, quote_strings))
                                .unwrap_or_default(),
                        );
                    }
                    AxisItem::Ellipsis => parts.push("...".to_string()),
                }
            }
            format!("[{}]", parts.join(", "))
        }
        _ => {
            let mut parts = Vec::new();
            for item in axis_indices(view.shape()[0], edge) {
                match item {
                    AxisItem::Index(i) => {
                        let v = view.index_axis(Axis(0), i);
                        parts.push(format_array_view_with_ellipsis_display(
                            v.into_dyn(),
                            edge,
                            quote_strings,
                        ));
                    }
                    AxisItem::Ellipsis => parts.push("...".to_string()),
                }
            }
            format_multiline_list(&parts)
        }
    }
}

fn format_array_view_display(view: ArrayViewD<String>, quote_strings: bool) -> String {
    match view.ndim() {
        0 => view
            .first()
            .map(|v| format_string_element(v, quote_strings))
            .unwrap_or_default(),
        1 => {
            let mut parts = Vec::new();
            for i in 0..view.shape()[0] {
                let v = view.index_axis(Axis(0), i);
                parts.push(
                    v.first()
                        .map(|s| format_string_element(s, quote_strings))
                        .unwrap_or_default(),
                );
            }
            format!("[{}]", parts.join(", "))
        }
        _ => {
            let mut parts = Vec::new();
            for i in 0..view.shape()[0] {
                let v = view.index_axis(Axis(0), i);
                parts.push(format_array_view_display(v.into_dyn(), quote_strings));
            }
            format_multiline_list(&parts)
        }
    }
}

fn format_string_element(value: &str, quote_strings: bool) -> String {
    if quote_strings {
        format!("{:?}", value)
    } else {
        value.to_string()
    }
}

fn format_multiline_list(parts: &[String]) -> String {
    if parts.is_empty() {
        return "[]".to_string();
    }
    let indented: Vec<String> = parts.iter().map(|p| indent_lines(p, 2)).collect();
    format!("[\n{}\n]", indented.join(",\n"))
}

fn indent_lines(value: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    value
        .lines()
        .map(|line| format!("{prefix}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
