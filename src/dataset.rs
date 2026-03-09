use hdf5::{Dataset, Dataspace, H5Type, Selection};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{CompoundType, TypeDescriptor, VarLenUnicode, IntSize, FloatSize};
use hdf5::types::dyn_value::DynCompound;
use crate::utils;
use crate::slicing;
use anyhow::{anyhow, Result};
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};
use hdf5_sys::h5d::H5Dread;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t::{H5Tget_class, H5T_INTEGER};

const MAX_ARRAY_ELEMS: usize = 200;
const ARRAY_EDGE: usize = 3;
const MAX_COMPOUND_ROWS: usize = 20;
const COMPOUND_EDGE: usize = 5;

pub fn print_dataset_info(ds: &Dataset, slice_expr: Option<&str>, array_fmt: &utils::NumFormat, scalar_fmt: &utils::NumFormat) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor().ok();
    let shape = ds.shape();
    println!("      dtype: {}", utils::fmt_dtype(&dtype));
    println!("      shape: {}", utils::fmt_shape(&shape));

    let elem_count = if shape.is_empty() {
        Some(1u64)
    } else {
        shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
    };
    match elem_count {
        Some(count) => println!("   elements: {}", utils::fmt_u64(count, scalar_fmt)),
        None => println!("   elements: (too large)"),
    }

    let storage_bytes = ds.storage_size();
    println!("    storage: {}", utils::fmt_bytes(storage_bytes));

    if let Some(desc) = desc {
        if !descriptor_has_vlen(&desc) {
            match elem_count.and_then(|count| count.checked_mul(dtype.size() as u64)) {
                Some(logical_bytes) => {
                    println!("logical size: {}", utils::fmt_bytes(logical_bytes));
                    if storage_bytes > 0 && logical_bytes > 0 {
                        let ratio = logical_bytes as f64 / storage_bytes as f64;
                        println!("compression ratio: {:.2}x", ratio);
                    }
                }
                None => println!("logical size: (too large)"),
            }
        } else {
            println!("logical size: (variable-length)");
        }
    }
    
    if let Ok(space) = ds.space() {
        let maxshape = space.maxdims();
        let current_shape = &shape;
        let different = maxshape.len() != current_shape.len() || 
                        maxshape.iter().zip(current_shape.iter()).any(|(m, s)| m.map_or(true, |mv| mv != *s));
        
        if different {
            println!("   maxshape: {}", utils::fmt_maxshape(&maxshape));
        }
    }

    let create_plist = ds.create_plist()?;
    let layout = create_plist.layout();
    println!("     layout: {:?}", layout);

    if layout == Layout::Chunked {
        if let Some(chunks) = create_plist.chunk() {
             println!("      chunk: {}", utils::fmt_shape(&chunks));
        }
        
        let filters = create_plist.filters();
        if !filters.is_empty() {
             let filter_strs: Vec<String> = filters.iter().map(|f| format!("{:?}", f)).collect();
             println!("compression: {}", filter_strs.join(", "));
        }
    }

    if let Some(expr) = slice_expr {
        println!("\nselected data [{}]:", expr);
        let selection = slicing::parse_slice(expr, &ds.shape())
            .map_err(|e| anyhow!("Error parsing slice: {}", e))?;
        print_selection_data(ds, selection, array_fmt)
            .map_err(|e| anyhow!("Error reading sliced data: {}", e))?;
    } else if ds.ndim() == 0 {
        print_scalar(ds, scalar_fmt)?;
    } else {
        println!("\nsample data:");
        if let Err(e) = print_sample_data(ds, array_fmt) {
            println!("(error reading sample data: {})", e);
        }
    }

    let attr_names = ds.attr_names()?;
    println!("\n{} attributes:", attr_names.len());
    for name in attr_names {
        let attr = ds.attr(&name)?;
        println!("* {}: {}", name, utils::format_attribute_value(&attr, scalar_fmt));
    }
    Ok(())
}

fn print_selection_data(ds: &Dataset, selection: Selection, fmt: &utils::NumFormat) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => {
            if dtype_is_integer(&dtype) {
                let size = dtype.size();
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            println!("(data type not supported for display)");
            return Ok(());
        }
    };

    match &desc {
        TypeDescriptor::Integer(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            print_selection_int(ds, selection, fmt)
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            print_selection_uint(ds, selection, fmt)
        }
        TypeDescriptor::Float(_) => print_selection_float(ds, selection, fmt),
        TypeDescriptor::Boolean => print_selection::<bool>(ds, selection),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => print_selection_string(ds, selection),
        TypeDescriptor::FixedAscii(len) => print_selection_fixed_string(ds, selection, *len, false),
        TypeDescriptor::FixedUnicode(len) => print_selection_fixed_string(ds, selection, *len, true),
        TypeDescriptor::Compound(compound) => print_selection_compound(ds, selection, compound),
        _ => {
            println!("(data type not supported for display)");
            Ok(())
        }
    }
}

fn print_selection<T>(ds: &Dataset, selection: Selection) -> Result<()> 
where T: H5Type + std::fmt::Debug
{
    // Use read_slice which handles multi-dim selections correctly in hdf5-metno
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    println!("{}", format_array_with_ellipsis(&arr));
    Ok(())
}

fn print_selection_int(ds: &Dataset, selection: Selection, fmt: &utils::NumFormat) -> Result<()> {
    let arr: ArrayD<i64> = ds.read_slice::<i64, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v| utils::fmt_i64(*v, fmt));
    println!("{}", format_array_with_ellipsis_display(&s_arr, false));
    Ok(())
}

fn print_selection_uint(ds: &Dataset, selection: Selection, fmt: &utils::NumFormat) -> Result<()> {
    let arr: ArrayD<u64> = ds.read_slice::<u64, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v| utils::fmt_u64(*v, fmt));
    println!("{}", format_array_with_ellipsis_display(&s_arr, false));
    Ok(())
}

fn print_selection_float(ds: &Dataset, selection: Selection, fmt: &utils::NumFormat) -> Result<()> {
    let arr: ArrayD<f64> = ds.read_slice::<f64, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v| utils::fmt_f64(*v, fmt));
    println!("{}", format_array_with_ellipsis_display(&s_arr, false));
    Ok(())
}

fn print_selection_string(ds: &Dataset, selection: Selection) -> Result<()> {
    let arr: ArrayD<VarLenUnicode> = ds.read_slice::<VarLenUnicode, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v: &VarLenUnicode| v.as_str().to_string());
    println!("{}", format_array_with_ellipsis_display(&s_arr, true));
    Ok(())
}

fn print_selection_fixed_string(ds: &Dataset, selection: Selection, len: usize, is_unicode: bool) -> Result<()> {
    let arr = read_fixed_string_selection(ds, selection, len, is_unicode)?;
    println!("{}", format_array_with_ellipsis_display(&arr, true));
    Ok(())
}

fn print_selection_compound(ds: &Dataset, selection: Selection, compound: &CompoundType) -> Result<()> {
    if compound_has_vlen(compound) {
        println!("(compound data with variable-length fields is not supported for display)");
        return Ok(());
    }
    if !compound_alignment_safe(compound) {
        println!("(compound data display skipped: unaligned or unsupported layout)");
        return Ok(());
    }

    let dtype = ds.dtype()?;
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        println!("[]");
        return Ok(());
    }

    let fspace = obj_space.select(selection)?;
    let mspace = Dataspace::try_new(&out_shape)?;
    let elem_size = dtype.size();
    let mut buf = vec![0u8; out_size * elem_size];

    let status = unsafe {
        H5Dread(
            ds.id(),
            dtype.id(),
            mspace.id(),
            fspace.id(),
            H5P_DEFAULT,
            buf.as_mut_ptr().cast(),
        )
    };
    if status < 0 {
        return Err(anyhow!("Error reading compound data"));
    }

    let include_index = out_shape.len() > 1;
    let mut headers: Vec<String> = Vec::new();
    if include_index {
        headers.push("idx".to_string());
    }
    headers.extend(compound.fields.iter().map(|f| f.name.clone()));

    let mut rows: Vec<Vec<String>> = Vec::with_capacity(out_size);
    for elem_idx in 0..out_size {
        let offset = elem_idx * elem_size;
        let elem_buf = &buf[offset..offset + elem_size];
        let value = DynCompound::new(compound, elem_buf);
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        if include_index {
            row.push(format_index(elem_idx, &out_shape));
        }
        for (_, field_val) in value.iter() {
            row.push(format!("{:?}", field_val));
        }
        rows.push(row);
    }

    let rows = truncate_rows(&rows, COMPOUND_EDGE, MAX_COMPOUND_ROWS);
    let table = format_aligned_table(&headers, &rows);
    println!("{}", table);
    Ok(())
}

fn print_scalar(ds: &Dataset, fmt: &utils::NumFormat) -> Result<()> {
    println!("\ndata:");
    let dtype = ds.dtype()?;
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => {
            if dtype_is_integer(&dtype) {
                let size = dtype.size();
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            println!("(data type not supported for display)");
            return Ok(());
        }
    };
    
    match desc {
        TypeDescriptor::Integer(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            match ds.read_scalar::<i64>() {
                Ok(v) => println!("{}", utils::fmt_i64(v, fmt)),
                Err(e) => println!("(failed to read scalar value: {e})"),
            }
        },
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            match ds.read_scalar::<u64>() {
                Ok(v) => println!("{}", utils::fmt_u64(v, fmt)),
                Err(e) => println!("(failed to read scalar value: {e})"),
            }
        },
        TypeDescriptor::Float(_) => {
             match ds.read_scalar::<f64>() {
                 Ok(v) => println!("{}", utils::fmt_f64(v, fmt)),
                 Err(e) => println!("(failed to read scalar value: {e})"),
             }
        },
        TypeDescriptor::Boolean => {
             match ds.read_scalar::<bool>() {
                 Ok(v) => println!("{}", v),
                 Err(e) => println!("(failed to read scalar value: {e})"),
             }
        },
        TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
             match ds.read_scalar::<VarLenUnicode>() {
                 Ok(v) => println!("{}", v.as_str()),
                 Err(e) => println!("(failed to read scalar value: {e})"),
             }
        },
        TypeDescriptor::FixedAscii(len) => {
            match read_fixed_string_scalar(ds, len, false) {
                Ok(value) => println!("{}", value),
                Err(e) => println!("(failed to read scalar value: {e})"),
            }
        }
        TypeDescriptor::FixedUnicode(len) => {
            match read_fixed_string_scalar(ds, len, true) {
                Ok(value) => println!("{}", value),
                Err(e) => println!("(failed to read scalar value: {e})"),
            }
        }
        _ => println!("(data type not supported for display)"),
    }
    Ok(())
}

fn print_sample_data(ds: &Dataset, fmt: &utils::NumFormat) -> Result<()> {
    let shape = ds.shape();
    if shape.is_empty() { return Ok(()); }

    // If dataset is very large, take a slice
    let total_size = total_size_checked(&shape);
    if total_size.map_or(true, |size| size > 100) {
        // Construct a sample slice string like "0:10, 0:10, ..."
        let mut sample_parts = Vec::new();
        for dim_len in &shape {
            sample_parts.push(format!("0:{}", dim_len.min(&10)));
        }
        let sample_expr = sample_parts.join(",");
        if let Ok(selection) = slicing::parse_slice(&sample_expr, &shape) {
            return print_selection_data(ds, selection, fmt);
        }
    }

    print_selection_data(ds, Selection::new(..), fmt)
}

fn is_standard_int_size(size: usize) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

fn total_size_checked(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

fn dtype_is_integer(dtype: &hdf5::Datatype) -> bool {
    unsafe { H5Tget_class(dtype.id()) == H5T_INTEGER }
}

fn alignment_for_descriptor(desc: &TypeDescriptor) -> Option<usize> {
    match desc {
        TypeDescriptor::Integer(size) => int_size_to_bytes(*size),
        TypeDescriptor::Unsigned(size) => int_size_to_bytes(*size),
        TypeDescriptor::Float(size) => float_size_to_bytes(*size),
        TypeDescriptor::Boolean => Some(1),
        TypeDescriptor::Enum(e) => int_size_to_bytes(e.size),
        TypeDescriptor::FixedAscii(_) | TypeDescriptor::FixedUnicode(_) => Some(1),
        TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenArray(_) => None,
        TypeDescriptor::FixedArray(_, _) => None,
        TypeDescriptor::Compound(_) => None,
        TypeDescriptor::Reference(_) => None,
    }
}

fn int_size_to_bytes(size: IntSize) -> Option<usize> {
    match size {
        IntSize::U1 => Some(1),
        IntSize::U2 => Some(2),
        IntSize::U4 => Some(4),
        IntSize::U8 => Some(8),
    }
}

fn float_size_to_bytes(size: FloatSize) -> Option<usize> {
    match size {
        FloatSize::U4 => Some(4),
        FloatSize::U8 => Some(8),
    }
}

fn compound_alignment_safe(compound: &CompoundType) -> bool {
    for field in &compound.fields {
        let align = match alignment_for_descriptor(&field.ty) {
            Some(a) if a > 0 => a,
            _ => return false,
        };
        // We read into a `Vec<u8>` (alignment 1). To avoid misaligned
        // dereferences in downstream dynamic parsing, only allow
        // compounds whose fields have alignment 1.
        if align != 1 {
            return false;
        }
        if field.offset % align != 0 {
            return false;
        }
    }
    true
}

fn read_fixed_string_scalar(ds: &Dataset, len: usize, is_unicode: bool) -> Result<String> {
    let arr = read_fixed_string_selection(ds, Selection::new(..), len, is_unicode)?;
    Ok(arr.first().cloned().unwrap_or_default())
}

fn read_fixed_string_selection(ds: &Dataset, selection: Selection, len: usize, is_unicode: bool) -> Result<ArrayD<String>> {
    let dtype = ds.dtype()?;
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
    if out_size == 0 {
        return Ok(ArrayD::from_shape_vec(IxDyn(&out_shape), Vec::new()).map_err(|e| anyhow!(e))?);
    }

    let fspace = obj_space.select(selection)?;
    let mspace = Dataspace::try_new(&out_shape)?;
    let mut buf = vec![0u8; out_size * len];

    let status = unsafe {
        H5Dread(
            ds.id(),
            dtype.id(),
            mspace.id(),
            fspace.id(),
            H5P_DEFAULT,
            buf.as_mut_ptr().cast(),
        )
    };
    if status < 0 {
        return Err(anyhow!("Error reading fixed-length string data"));
    }

    let mut out: Vec<String> = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let start = i * len;
        let end = start + len;
        let slice = &buf[start..end];
        out.push(fixed_bytes_to_string(slice, is_unicode));
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), out).map_err(|e| anyhow!(e))
}

fn fixed_bytes_to_string(bytes: &[u8], _is_unicode: bool) -> String {
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn format_array_with_ellipsis<T: std::fmt::Debug>(arr: &ArrayD<T>) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    let needs_ellipsis = arr.len() > MAX_ARRAY_ELEMS
        || arr.shape().iter().any(|&d| d > ARRAY_EDGE * 2);
    if !needs_ellipsis {
        return format!("{:?}", arr);
    }
    format_array_view_with_ellipsis(arr.view(), ARRAY_EDGE)
}

fn format_array_with_ellipsis_display(arr: &ArrayD<String>, quote_strings: bool) -> String {
    if arr.is_empty() {
        return "[]".to_string();
    }
    let needs_ellipsis = arr.len() > MAX_ARRAY_ELEMS
        || arr.shape().iter().any(|&d| d > ARRAY_EDGE * 2);
    if !needs_ellipsis {
        return format_array_view_display(arr.view(), quote_strings);
    }
    format_array_view_with_ellipsis_display(arr.view(), ARRAY_EDGE, quote_strings)
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

fn format_array_view_with_ellipsis_display(view: ArrayViewD<String>, edge: usize, quote_strings: bool) -> String {
    match view.ndim() {
        0 => view.first().map(|v| format_string_element(v, quote_strings)).unwrap_or_default(),
        1 => {
            let mut parts = Vec::new();
            for item in axis_indices(view.shape()[0], edge) {
                match item {
                    AxisItem::Index(i) => {
                        let v = view.index_axis(Axis(0), i);
                        parts.push(v.first().map(|s| format_string_element(s, quote_strings)).unwrap_or_default());
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
                        parts.push(format_array_view_with_ellipsis_display(v.into_dyn(), edge, quote_strings));
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
        0 => view.first().map(|v| format_string_element(v, quote_strings)).unwrap_or_default(),
        1 => {
            let mut parts = Vec::new();
            for i in 0..view.shape()[0] {
                let v = view.index_axis(Axis(0), i);
                parts.push(v.first().map(|s| format_string_element(s, quote_strings)).unwrap_or_default());
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

fn compound_has_vlen(compound: &CompoundType) -> bool {
    compound.fields.iter().any(|field| descriptor_has_vlen(&field.ty))
}

fn descriptor_has_vlen(desc: &TypeDescriptor) -> bool {
    match desc {
        TypeDescriptor::VarLenArray(_) | TypeDescriptor::VarLenAscii | TypeDescriptor::VarLenUnicode => true,
        TypeDescriptor::FixedArray(inner, _) => descriptor_has_vlen(inner),
        TypeDescriptor::Compound(compound) => compound.fields.iter().any(|field| descriptor_has_vlen(&field.ty)),
        _ => false,
    }
}

fn format_index(flat: usize, shape: &[usize]) -> String {
    if shape.is_empty() {
        return "()".to_string();
    }
    let mut idx = vec![0usize; shape.len()];
    let mut rem = flat;
    for (pos, dim) in shape.iter().rev().enumerate() {
        let i = shape.len() - 1 - pos;
        let d = *dim;
        if d == 0 {
            idx[i] = 0;
        } else {
            idx[i] = rem % d;
            rem /= d;
        }
    }
    format!("({})", idx.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))
}

fn truncate_rows(rows: &[Vec<String>], edge: usize, max_rows: usize) -> Vec<Vec<String>> {
    if rows.len() <= max_rows || rows.len() <= edge * 2 {
        return rows.to_vec();
    }
    let mut out = Vec::new();
    out.extend_from_slice(&rows[..edge]);
    out.push(vec!["...".to_string(); rows[0].len()]);
    out.extend_from_slice(&rows[rows.len() - edge..]);
    out
}

fn format_aligned_table(headers: &[String], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i >= widths.len() {
                widths.push(cell.len());
            } else if cell.len() > widths[i] {
                widths[i] = cell.len();
            }
        }
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push(format_row(headers, &widths));
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    lines.push(format_row(&sep, &widths));
    for row in rows {
        lines.push(format_row(row, &widths));
    }
    lines.join("\n")
}

fn format_row(cells: &[String], widths: &[usize]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(cells.len());
    for (cell, width) in cells.iter().zip(widths.iter()) {
        parts.push(format!("{:<width$}", cell, width = *width));
    }
    parts.join("  ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn format_array_with_ellipsis_small_matches_debug() {
        let arr = array![[1, 2], [3, 4]].into_dyn();
        assert_eq!(format_array_with_ellipsis(&arr), format!("{:?}", arr));
    }

    #[test]
    fn format_array_with_ellipsis_large_includes_ellipsis() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[1, 10]), (0..10).collect()).unwrap();
        let formatted = format_array_with_ellipsis(&arr);
        assert!(formatted.contains("..."));
    }

    #[test]
    fn format_aligned_table_pads_columns() {
        let headers = vec!["a".to_string(), "bb".to_string()];
        let rows = vec![
            vec!["1".to_string(), "22".to_string()],
            vec!["333".to_string(), "4".to_string()],
        ];
        let table = format_aligned_table(&headers, &rows);
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines[0].contains("a"));
        assert!(lines[0].contains("bb"));
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn format_array_display_does_not_quote_strings() {
        let arr = array![["alpha".to_string(), "beta".to_string()]].into_dyn();
        let formatted = format_array_with_ellipsis_display(&arr, false);
        assert!(formatted.contains("alpha"));
        assert!(!formatted.contains("\"alpha\""));
    }

    #[test]
    fn format_array_display_quotes_strings() {
        let arr = array![["alpha".to_string(), "beta".to_string()]].into_dyn();
        let formatted = format_array_with_ellipsis_display(&arr, true);
        assert!(formatted.contains("\"alpha\""));
    }

    #[test]
    fn total_size_checked_overflow_returns_none() {
        let shape = vec![usize::MAX, 2];
        assert!(total_size_checked(&shape).is_none());
    }
}
