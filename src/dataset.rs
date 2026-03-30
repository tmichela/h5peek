use hdf5::{Dataset, Dataspace, H5Type, Selection};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{CompoundType, TypeDescriptor, VarLenUnicode, VarLenAscii, IntSize, FloatSize};
use hdf5::types::dyn_value::DynCompound;
use crate::utils;
use crate::slicing;
use crate::plot;
use crate::plot::PlotBackend;
use crate::array_format::{self, EllipsisConfig};
use anyhow::{anyhow, Result};
use ndarray::{ArrayD, IxDyn};
use hdf5_sys::h5d::H5Dread;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t::{H5Tget_class, H5T_INTEGER};

const MAX_ARRAY_ELEMS: usize = 200;
const ARRAY_EDGE: usize = 3;
const MAX_COMPOUND_ROWS: usize = 20;
const COMPOUND_EDGE: usize = 5;
const ARRAY_FORMAT: EllipsisConfig = EllipsisConfig { max_elems: MAX_ARRAY_ELEMS, edge: ARRAY_EDGE };

pub fn print_dataset_info(ds: &Dataset, slice_expr: Option<&str>, array_fmt: &utils::NumFormat, scalar_fmt: &utils::NumFormat, truncate_attr_strings: bool) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor().ok();
    let shape = ds.shape();
    let storage_bytes = ds.storage_size();
    print_dataset_summary(&dtype, desc.as_ref(), &shape, storage_bytes, scalar_fmt);
    print_dataset_maxshape(ds, &shape)?;
    print_dataset_layout(ds)?;
    print_dataset_preview(ds, slice_expr, array_fmt, scalar_fmt)?;
    print_dataset_attrs(ds, scalar_fmt, truncate_attr_strings)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlotMode {
    Selection,
    Disabled,
}

fn print_dataset_summary(
    dtype: &hdf5::Datatype,
    desc: Option<&TypeDescriptor>,
    shape: &[usize],
    storage_bytes: u64,
    scalar_fmt: &utils::NumFormat,
) {
    println!("      dtype: {}", utils::fmt_dtype(dtype));
    println!("      shape: {}", utils::fmt_shape(shape));

    let elem_count = elem_count_u64(shape);
    match elem_count {
        Some(count) => println!("   elements: {}", utils::fmt_u64(count, scalar_fmt)),
        None => println!("   elements: (too large)"),
    }

    println!("    storage: {}", utils::fmt_bytes(storage_bytes));

    if let Some(desc) = desc {
        if !descriptor_has_vlen(desc) {
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
}

fn print_dataset_maxshape(ds: &Dataset, shape: &[usize]) -> Result<()> {
    if let Ok(space) = ds.space() {
        let maxshape = space.maxdims();
        let current_shape = shape;
        let different = maxshape.len() != current_shape.len()
            || maxshape
                .iter()
                .zip(current_shape.iter())
                .any(|(m, s)| m.is_none_or(|mv| mv != *s));

        if different {
            println!("   maxshape: {}", utils::fmt_maxshape(&maxshape));
        }
    }
    Ok(())
}

fn print_dataset_layout(ds: &Dataset) -> Result<()> {
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
    Ok(())
}

fn print_dataset_preview(
    ds: &Dataset,
    slice_expr: Option<&str>,
    array_fmt: &utils::NumFormat,
    scalar_fmt: &utils::NumFormat,
) -> Result<()> {
    if let Some(expr) = slice_expr {
        println!("\nselected data [{}]:", expr);
        let selection = slicing::parse_slice(expr, &ds.shape())
            .map_err(|e| anyhow!("Error parsing slice: {}", e))?;
        print_selection_data(ds, selection, array_fmt, PlotMode::Selection)
            .map_err(|e| anyhow!("Error reading sliced data: {}", e))?;
    } else if ds.ndim() == 0 {
        print_scalar(ds, scalar_fmt)?;
    } else {
        println!("\nsample data:");
        if let Err(e) = print_sample_data(ds, array_fmt) {
            println!("(error reading sample data: {})", e);
        }
    }
    Ok(())
}

fn print_dataset_attrs(
    ds: &Dataset,
    scalar_fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
) -> Result<()> {
    let attr_names = ds.attr_names()?;
    println!("\n{} attributes:", attr_names.len());
    for name in attr_names {
        let attr = ds.attr(&name)?;
        println!(
            "* {}: {}",
            name,
            utils::format_attribute_value(&attr, scalar_fmt, truncate_attr_strings)
        );
    }
    Ok(())
}

fn elem_count_u64(shape: &[usize]) -> Option<u64> {
    if shape.is_empty() {
        Some(1u64)
    } else {
        shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
    }
}

fn print_selection_data(ds: &Dataset, selection: Selection, fmt: &utils::NumFormat, plot_mode: PlotMode) -> Result<()> {
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
            print_selection_numeric::<i64>(ds, selection, fmt, plot_mode)
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            print_selection_numeric::<u64>(ds, selection, fmt, plot_mode)
        }
        TypeDescriptor::Float(_) => {
            print_selection_numeric::<f64>(ds, selection, fmt, plot_mode)
        }
        TypeDescriptor::Boolean => print_selection::<bool>(ds, selection),
        TypeDescriptor::VarLenUnicode => print_selection_varlen_string::<VarLenUnicode>(ds, selection),
        TypeDescriptor::VarLenAscii => print_selection_varlen_string::<VarLenAscii>(ds, selection),
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            print_selection_fixed_string(ds, selection, *len)
        }
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

trait NumericFormat: H5Type + Copy {
    fn format_value(self, fmt: &utils::NumFormat) -> String;
    fn to_f64(self) -> f64;
}

impl NumericFormat for i64 {
    fn format_value(self, fmt: &utils::NumFormat) -> String {
        utils::fmt_i64(self, fmt)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl NumericFormat for u64 {
    fn format_value(self, fmt: &utils::NumFormat) -> String {
        utils::fmt_u64(self, fmt)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl NumericFormat for f64 {
    fn format_value(self, fmt: &utils::NumFormat) -> String {
        utils::fmt_f64(self, fmt)
    }

    fn to_f64(self) -> f64 {
        self
    }
}

fn print_selection_numeric<T>(
    ds: &Dataset,
    selection: Selection,
    fmt: &utils::NumFormat,
    plot_mode: PlotMode,
) -> Result<()>
where
    T: NumericFormat,
{
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v| v.format_value(fmt));
    println!("{}", format_array_with_ellipsis_display(&s_arr, false));
    if plot_mode == PlotMode::Selection {
        maybe_print_plot_from_array(&arr);
    }
    Ok(())
}

fn print_selection_varlen_string<T>(ds: &Dataset, selection: Selection) -> Result<()>
where
    T: H5Type + AsRef<str>,
{
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v: &T| v.as_ref().to_string());
    println!("{}", format_array_with_ellipsis_display(&s_arr, true));
    Ok(())
}

fn print_selection_fixed_string(ds: &Dataset, selection: Selection, len: usize) -> Result<()> {
    let arr = read_fixed_string_selection(ds, selection, len)?;
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
            print_scalar_numeric::<i64>(ds, fmt);
        },
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!("(data display not supported for integer size {} bytes)", size);
                return Ok(());
            }
            print_scalar_numeric::<u64>(ds, fmt);
        },
        TypeDescriptor::Float(_) => {
             print_scalar_numeric::<f64>(ds, fmt);
        },
        TypeDescriptor::Boolean => {
             match ds.read_scalar::<bool>() {
                 Ok(v) => println!("{}", v),
                 Err(e) => println!("(failed to read scalar value: {e})"),
             }
        },
        TypeDescriptor::VarLenUnicode => {
             print_scalar_varlen_string::<VarLenUnicode>(ds);
        },
        TypeDescriptor::VarLenAscii => {
             print_scalar_varlen_string::<VarLenAscii>(ds);
        },
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            match read_fixed_string_scalar(ds, len) {
                Ok(value) => println!("{}", value),
                Err(e) => println!("(failed to read scalar value: {e})"),
            }
        }
        _ => println!("(data type not supported for display)"),
    }
    Ok(())
}

fn print_scalar_numeric<T>(ds: &Dataset, fmt: &utils::NumFormat)
where
    T: NumericFormat,
{
    match ds.read_scalar::<T>() {
        Ok(v) => println!("{}", v.format_value(fmt)),
        Err(e) => println!("(failed to read scalar value: {e})"),
    }
}

fn print_scalar_varlen_string<T>(ds: &Dataset)
where
    T: H5Type + AsRef<str>,
{
    match ds.read_scalar::<T>() {
        Ok(v) => println!("{}", v.as_ref()),
        Err(e) => println!("(failed to read scalar value: {e})"),
    }
}

fn print_sample_data(ds: &Dataset, fmt: &utils::NumFormat) -> Result<()> {
    let shape = ds.shape();
    if shape.is_empty() { return Ok(()); }

    // If dataset is very large, take a slice
    let total_size = total_size_checked(&shape);
    if total_size.is_none_or(|size| size > 100) {
        // Construct a sample slice string like "0:10, 0:10, ..."
        let mut sample_parts = Vec::new();
        for dim_len in &shape {
            sample_parts.push(format!("0:{}", dim_len.min(&10)));
        }
        let sample_expr = sample_parts.join(",");
        if let Ok(selection) = slicing::parse_slice(&sample_expr, &shape) {
            print_selection_data(ds, selection, fmt, PlotMode::Disabled)?;
            plot_full_dataset_1d(ds)?;
            return Ok(());
        }
    }

    print_selection_data(ds, Selection::new(..), fmt, PlotMode::Disabled)?;
    plot_full_dataset_1d(ds)?;
    Ok(())
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

fn read_fixed_string_scalar(ds: &Dataset, len: usize) -> Result<String> {
    let arr = read_fixed_string_selection(ds, Selection::new(..), len)?;
    Ok(arr.first().cloned().unwrap_or_default())
}

fn read_fixed_string_selection(ds: &Dataset, selection: Selection, len: usize) -> Result<ArrayD<String>> {
    let dtype = ds.dtype()?;
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = if out_shape.is_empty() { 1 } else { out_shape.iter().product() };
    if out_size == 0 {
        return ArrayD::from_shape_vec(IxDyn(&out_shape), Vec::new()).map_err(|e| anyhow!(e));
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
        out.push(utils::decode_fixed_bytes(slice, false));
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), out).map_err(|e| anyhow!(e))
}

fn plot_full_dataset_1d(ds: &Dataset) -> Result<()> {
    let shape = ds.shape();
    if shape.len() != 1 {
        return Ok(());
    }
    let len = shape[0];
    if len < 2 {
        return Ok(());
    }

    let dtype = ds.dtype()?;
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => {
            return Ok(());
        }
    };

    match desc {
        TypeDescriptor::Integer(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                return Ok(());
            }
            let arr: ArrayD<i64> = ds.read_slice::<i64, _, IxDyn>(Selection::new(..))?;
            maybe_print_plot_from_array(&arr);
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                return Ok(());
            }
            let arr: ArrayD<u64> = ds.read_slice::<u64, _, IxDyn>(Selection::new(..))?;
            maybe_print_plot_from_array(&arr);
        }
        TypeDescriptor::Float(_) => {
            let arr: ArrayD<f64> = ds.read_slice::<f64, _, IxDyn>(Selection::new(..))?;
            maybe_print_plot_from_array(&arr);
        }
        _ => {}
    }
    Ok(())
}

fn maybe_print_plot_from_array<T>(arr: &ArrayD<T>)
where
    T: NumericFormat,
{
    if arr.ndim() != 1 || arr.len() < 2 {
        return;
    }
    let values: Vec<f64> = arr.iter().map(|v| v.to_f64()).collect();
    maybe_print_plot(&values);
}

fn maybe_print_plot(values: &[f64]) {
    if let Some(frame) = plot::default_backend().render_1d(values) {
        println!("\nplot:");
        println!("{frame}");
    }
}

fn format_array_with_ellipsis<T: std::fmt::Debug>(arr: &ArrayD<T>) -> String {
    array_format::format_debug_with_ellipsis(arr, ARRAY_FORMAT)
}

fn format_array_with_ellipsis_display(arr: &ArrayD<String>, quote_strings: bool) -> String {
    array_format::format_string_array_with_ellipsis(arr, ARRAY_FORMAT, quote_strings)
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
