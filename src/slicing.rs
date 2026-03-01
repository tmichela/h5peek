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
            None => (0..*dim_len).into(),
        };
        ranges.push(range);
    }

    Ok(Selection::new(Hyperslab::from(ranges)))
}

fn parse_range(s: &str, dim_len: usize) -> Result<SliceOrIndex> {
    let s = s.trim();
    if s == ":" || s.is_empty() {
        return Ok((0..dim_len).into());
    }
    
    if !s.contains(':') {
        let idx = s.parse::<isize>().map_err(|_| anyhow!("Invalid index: {}", s))?;
        let start = if idx < 0 { dim_len as isize + idx } else { idx };
        if start < 0 || start >= dim_len as isize {
            return Err(anyhow!("Index {} out of bounds for dimension of length {}", s, dim_len));
        }
        return Ok(SliceOrIndex::Index(start as usize));
    }

    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() > 3 {
        return Err(anyhow!("Invalid slice: {}", s));
    }
    let start_s = parts[0].trim();
    let stop_s = if parts.len() > 1 { parts[1].trim() } else { "" };
    let step_s = if parts.len() > 2 { parts[2].trim() } else { "" };

    let mut start = if start_s.is_empty() { 0 } else { start_s.parse::<isize>().map_err(|_| anyhow!("Invalid slice start: {}", start_s))? };
    let mut stop = if stop_s.is_empty() { dim_len as isize } else { stop_s.parse::<isize>().map_err(|_| anyhow!("Invalid slice stop: {}", stop_s))? };
    let step = if step_s.is_empty() { 1 } else { step_s.parse::<isize>().map_err(|_| anyhow!("Invalid slice step: {}", step_s))? };

    if step == 0 {
        return Err(anyhow!("Slice step cannot be 0"));
    }
    if step < 0 {
        return Err(anyhow!("Negative slice steps are not supported: {}", s));
    }

    if start < 0 {
        start += dim_len as isize;
    }
    if stop < 0 {
        stop += dim_len as isize;
    }

    let start = start.clamp(0, dim_len as isize) as usize;
    let stop = stop.clamp(0, dim_len as isize) as usize;
    let step = step as usize;

    Ok(SliceOrIndex::SliceTo { start, step, end: stop, block: 1 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_supports_steps() {
        match parse_range("0:10:2", 10).unwrap() {
            SliceOrIndex::SliceTo { start, step, end, block } => {
                assert_eq!(start, 0);
                assert_eq!(step, 2);
                assert_eq!(end, 10);
                assert_eq!(block, 1);
            }
            _ => panic!("expected slice"),
        }
        match parse_range("::2", 10).unwrap() {
            SliceOrIndex::SliceTo { start, step, end, block } => {
                assert_eq!(start, 0);
                assert_eq!(step, 2);
                assert_eq!(end, 10);
                assert_eq!(block, 1);
            }
            _ => panic!("expected slice"),
        }
    }

    #[test]
    fn parse_range_errors_on_out_of_bounds_single_index() {
        assert!(parse_range("5", 5).is_err());
        assert!(parse_range("-6", 5).is_err());
    }

    #[test]
    fn parse_range_accepts_valid_negative_index() {
        assert_eq!(parse_range("-1", 5).unwrap(), SliceOrIndex::Index(4));
    }

    #[test]
    fn parse_slice_accepts_step_expression() {
        assert!(parse_slice("0:10:2", &[10]).is_ok());
        assert!(parse_slice("::2", &[10]).is_ok());
    }

    #[test]
    fn parse_range_rejects_negative_step() {
        let err = parse_range("10:0:-1", 10).unwrap_err();
        assert!(err.to_string().contains("Negative slice steps"));
    }
}
