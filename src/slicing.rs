use anyhow::{anyhow, Result};
use hdf5::{Hyperslab, Selection, SliceOrIndex};

pub fn parse_slice(s: &str, shape: &[usize]) -> Result<Selection> {
    let s = s.trim();
    if s.is_empty() || s == ":" {
        return Ok(Selection::new(..));
    }

    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    if parts.len() > shape.len() {
        return Err(anyhow!("Too many indices for {}D dataset", shape.len()));
    }

    let mut ranges: Vec<SliceOrIndex> = Vec::with_capacity(shape.len());
    for (i, dim_len) in shape.iter().enumerate() {
        let range = match parts.get(i) {
            Some(part) => parse_range(part, *dim_len)?,
            None => 0..*dim_len,
        };
        ranges.push(range.into());
    }

    Ok(Selection::new(Hyperslab::from(ranges)))
}

fn parse_range(s: &str, dim_len: usize) -> Result<std::ops::Range<usize>> {
    let s = s.trim();
    if s == ":" || s.is_empty() {
        return Ok(0..dim_len);
    }
    
    if !s.contains(':') {
        let idx = s.parse::<isize>().map_err(|_| anyhow!("Invalid index: {}", s))?;
        let start = if idx < 0 { dim_len as isize + idx } else { idx };
        if start < 0 || start >= dim_len as isize {
            return Err(anyhow!("Index {} out of bounds for dimension of length {}", s, dim_len));
        }
        let start = start as usize;
        let stop = start + 1;
        return Ok(start..stop);
    }

    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() > 2 {
        return Err(anyhow!("Slice steps are not supported: {}", s));
    }
    let start_s = parts[0].trim();
    let stop_s = if parts.len() > 1 { parts[1].trim() } else { "" };

    let start = if start_s.is_empty() { 0 } else { start_s.parse::<isize>()? };
    let start = if start < 0 { dim_len as isize + start } else { start };
    let start = start.clamp(0, dim_len as isize) as usize;

    let stop = if stop_s.is_empty() { dim_len as isize } else { stop_s.parse::<isize>()? };
    let stop = if stop < 0 { dim_len as isize + stop } else { stop };
    let stop = stop.clamp(0, dim_len as isize) as usize;

    Ok(start..stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_rejects_steps() {
        let err = parse_range("0:10:2", 10).unwrap_err();
        assert!(err.to_string().contains("steps"));
        let err = parse_range("::", 10).unwrap_err();
        assert!(err.to_string().contains("steps"));
    }

    #[test]
    fn parse_range_errors_on_out_of_bounds_single_index() {
        assert!(parse_range("5", 5).is_err());
        assert!(parse_range("-6", 5).is_err());
    }

    #[test]
    fn parse_range_accepts_valid_negative_index() {
        assert_eq!(parse_range("-1", 5).unwrap(), 4..5);
    }

    #[test]
    fn parse_slice_rejects_step_expression() {
        assert!(parse_slice("0:10:2", &[10]).is_err());
        assert!(parse_slice("::2", &[10]).is_err());
    }
}
