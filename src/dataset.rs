use crate::array_format::{self, EllipsisConfig};
use crate::plot;
use crate::plot::PlotBackend;
use crate::slicing;
use crate::utils;
use anyhow::{anyhow, Result};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{CompoundType, FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Dataset, Dataspace, Datatype, H5Type, Selection};
use hdf5_sys::h5d::H5Dread;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t::{H5Tget_class, H5T_INTEGER};
use ndarray::{ArrayD, IxDyn};

const MAX_ARRAY_ELEMS: usize = 200;
const ARRAY_EDGE: usize = 3;
const MAX_COMPOUND_ROWS: usize = 20;
const COMPOUND_EDGE: usize = 5;
const ARRAY_FORMAT: EllipsisConfig = EllipsisConfig {
    max_elems: MAX_ARRAY_ELEMS,
    edge: ARRAY_EDGE,
};

pub fn print_dataset_info(
    ds: &Dataset,
    slice_expr: Option<&str>,
    array_fmt: &utils::NumFormat,
    scalar_fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
) -> Result<()> {
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

#[derive(Clone, Copy)]
enum DisplayMode {
    Preview,
    Full,
}

impl DisplayMode {
    fn format_debug<T: std::fmt::Debug>(self, arr: &ArrayD<T>) -> String {
        match self {
            DisplayMode::Preview => format_array_with_ellipsis(arr),
            DisplayMode::Full => format_array_full(arr),
        }
    }

    fn format_strings(self, arr: &ArrayD<String>, quote_strings: bool) -> String {
        match self {
            DisplayMode::Preview => format_array_with_ellipsis_display(arr, quote_strings),
            DisplayMode::Full => format_array_full_display(arr, quote_strings),
        }
    }
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
        print_selection_data(
            ds,
            selection,
            array_fmt,
            PlotMode::Selection,
            DisplayMode::Full,
        )
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

fn print_selection_data(
    ds: &Dataset,
    selection: Selection,
    fmt: &utils::NumFormat,
    plot_mode: PlotMode,
    display_mode: DisplayMode,
) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => {
            if dtype_is_integer(&dtype) {
                let size = dtype.size();
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
                return Ok(());
            }
            println!("(data type not supported for display)");
            return Ok(());
        }
    };

    let plot_values = match &desc {
        TypeDescriptor::Integer(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
                return Ok(());
            }
            print_selection_numeric::<i64>(ds, selection, fmt, plot_mode, display_mode)?
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
                return Ok(());
            }
            print_selection_numeric::<u64>(ds, selection, fmt, plot_mode, display_mode)?
        }
        TypeDescriptor::Float(_) => {
            print_selection_numeric::<f64>(ds, selection, fmt, plot_mode, display_mode)?
        }
        TypeDescriptor::Boolean => {
            print_selection::<bool>(ds, selection, display_mode)?;
            None
        }
        TypeDescriptor::Enum(enum_type) => {
            print_selection_enum(ds, selection, &enum_type, display_mode)?;
            None
        }
        TypeDescriptor::VarLenUnicode => {
            print_selection_varlen_string::<VarLenUnicode>(ds, selection, display_mode)?;
            None
        }
        TypeDescriptor::VarLenAscii => {
            print_selection_varlen_string::<VarLenAscii>(ds, selection, display_mode)?;
            None
        }
        TypeDescriptor::FixedAscii(len) | TypeDescriptor::FixedUnicode(len) => {
            print_selection_fixed_string(ds, selection, *len, display_mode)?;
            None
        }
        TypeDescriptor::Compound(compound) => {
            print_selection_compound(ds, selection, compound)?;
            None
        }
        _ => {
            println!("(data type not supported for display)");
            None
        }
    };

    if plot_mode == PlotMode::Selection {
        if let Some(values) = plot_values {
            maybe_print_plot(&values);
        }
    }
    Ok(())
}

fn print_selection<T>(ds: &Dataset, selection: Selection, display_mode: DisplayMode) -> Result<()>
where
    T: H5Type + std::fmt::Debug,
{
    // Use read_slice which handles multi-dim selections correctly in hdf5-metno
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    println!("{}", display_mode.format_debug(&arr));
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
    display_mode: DisplayMode,
) -> Result<Option<Vec<f64>>>
where
    T: NumericFormat,
{
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v| v.format_value(fmt));
    println!("{}", display_mode.format_strings(&s_arr, false));
    if plot_mode == PlotMode::Selection {
        Ok(plot_values_from_array(&arr))
    } else {
        Ok(None)
    }
}

fn print_selection_varlen_string<T>(
    ds: &Dataset,
    selection: Selection,
    display_mode: DisplayMode,
) -> Result<()>
where
    T: H5Type + AsRef<str>,
{
    let arr: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v: &T| v.as_ref().to_string());
    println!("{}", display_mode.format_strings(&s_arr, true));
    Ok(())
}

fn print_selection_fixed_string(
    ds: &Dataset,
    selection: Selection,
    len: usize,
    display_mode: DisplayMode,
) -> Result<()> {
    let arr = read_fixed_string_selection(ds, selection, len)?;
    println!("{}", display_mode.format_strings(&arr, true));
    Ok(())
}

fn print_selection_enum(
    ds: &Dataset,
    selection: Selection,
    enum_type: &hdf5::types::EnumType,
    display_mode: DisplayMode,
) -> Result<()> {
    let arr = read_enum_selection(ds, selection, enum_type)?;
    println!("{}", display_mode.format_strings(&arr, false));
    Ok(())
}

fn print_selection_compound(
    ds: &Dataset,
    selection: Selection,
    compound: &CompoundType,
) -> Result<()> {
    if compound_has_vlen(compound) {
        println!("(compound data with variable-length fields is not supported for display)");
        return Ok(());
    }
    let packed = compound.to_packed_repr();
    if !compound_descriptor_supported(&TypeDescriptor::Compound(packed.clone())) {
        println!("(compound data contains unsupported field types)");
        return Ok(());
    }

    let mem_desc = TypeDescriptor::Compound(packed.clone());
    let mem_dtype = Datatype::from_descriptor(&mem_desc)?;
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = out_shape.iter().product();
    if out_size == 0 {
        println!("[]");
        return Ok(());
    }

    let fspace = obj_space.select(selection)?;
    let mspace = Dataspace::try_new(&out_shape)?;
    let elem_size = packed.size;
    let mut buf = vec![0u8; out_size * elem_size];

    let status = unsafe {
        H5Dread(
            ds.id(),
            mem_dtype.id(),
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
    headers.extend(packed.fields.iter().map(|f| f.name.clone()));

    let mut rows: Vec<Vec<String>> = Vec::with_capacity(out_size);
    for elem_idx in 0..out_size {
        let offset = elem_idx * elem_size;
        let elem_buf = &buf[offset..offset + elem_size];
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        if include_index {
            row.push(format_index(elem_idx, &out_shape));
        }
        for field in &packed.fields {
            let start = field.offset;
            let end = start + field.ty.size();
            if end > elem_buf.len() {
                row.push("(out of bounds)".to_string());
                continue;
            }
            row.push(format_compound_value(&field.ty, &elem_buf[start..end]));
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
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
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
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
                return Ok(());
            }
            print_scalar_numeric::<i64>(ds, fmt);
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                println!(
                    "(data display not supported for integer size {} bytes)",
                    size
                );
                return Ok(());
            }
            print_scalar_numeric::<u64>(ds, fmt);
        }
        TypeDescriptor::Float(_) => {
            print_scalar_numeric::<f64>(ds, fmt);
        }
        TypeDescriptor::Boolean => match ds.read_scalar::<bool>() {
            Ok(v) => println!("{}", v),
            Err(e) => println!("(failed to read scalar value: {e})"),
        },
        TypeDescriptor::Enum(enum_type) => {
            print_scalar_enum(ds, &enum_type);
        }
        TypeDescriptor::VarLenUnicode => {
            print_scalar_varlen_string::<VarLenUnicode>(ds);
        }
        TypeDescriptor::VarLenAscii => {
            print_scalar_varlen_string::<VarLenAscii>(ds);
        }
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

fn print_scalar_enum(ds: &Dataset, enum_type: &hdf5::types::EnumType) {
    match read_enum_selection(ds, Selection::new(..), enum_type) {
        Ok(arr) => match arr.first() {
            Some(value) => println!("{}", value),
            None => println!("[]"),
        },
        Err(e) => println!("(failed to read scalar value: {e})"),
    }
}

fn print_sample_data(ds: &Dataset, fmt: &utils::NumFormat) -> Result<()> {
    let shape = ds.shape();
    if shape.is_empty() {
        return Ok(());
    }

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
            print_selection_data(ds, selection, fmt, PlotMode::Disabled, DisplayMode::Preview)?;
            if let Some(values) = plot_full_dataset_1d(ds)? {
                maybe_print_plot(&values);
            }
            return Ok(());
        }
    }

    print_selection_data(
        ds,
        Selection::new(..),
        fmt,
        PlotMode::Disabled,
        DisplayMode::Preview,
    )?;
    if let Some(values) = plot_full_dataset_1d(ds)? {
        maybe_print_plot(&values);
    }
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

fn compound_descriptor_supported(desc: &TypeDescriptor) -> bool {
    match desc {
        TypeDescriptor::Integer(_)
        | TypeDescriptor::Unsigned(_)
        | TypeDescriptor::Float(_)
        | TypeDescriptor::Boolean
        | TypeDescriptor::Enum(_)
        | TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_) => true,
        TypeDescriptor::FixedArray(inner, _) => compound_descriptor_supported(inner),
        TypeDescriptor::Compound(compound) => compound
            .fields
            .iter()
            .all(|field| compound_descriptor_supported(&field.ty)),
        TypeDescriptor::VarLenArray(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode
        | TypeDescriptor::Reference(_) => false,
    }
}

fn read_fixed_string_scalar(ds: &Dataset, len: usize) -> Result<String> {
    let arr = read_fixed_string_selection(ds, Selection::new(..), len)?;
    Ok(arr.first().cloned().unwrap_or_default())
}

fn read_fixed_string_selection(
    ds: &Dataset,
    selection: Selection,
    len: usize,
) -> Result<ArrayD<String>> {
    let dtype = ds.dtype()?;
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
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

fn read_enum_selection(
    ds: &Dataset,
    selection: Selection,
    enum_type: &hdf5::types::EnumType,
) -> Result<ArrayD<String>> {
    let obj_space = ds.space()?;
    let out_shape = selection.out_shape(obj_space.shape())?;
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    if out_size == 0 {
        return ArrayD::from_shape_vec(IxDyn(&out_shape), Vec::new()).map_err(|e| anyhow!(e));
    }

    let fspace = obj_space.select(selection)?;
    let mspace = Dataspace::try_new(&out_shape)?;
    let elem_size = enum_type.size as usize;
    let mem_desc = TypeDescriptor::Enum(enum_type.clone());
    let mem_dtype = Datatype::from_descriptor(&mem_desc)?;
    let mut buf = vec![0u8; out_size * elem_size];

    let status = unsafe {
        H5Dread(
            ds.id(),
            mem_dtype.id(),
            mspace.id(),
            fspace.id(),
            H5P_DEFAULT,
            buf.as_mut_ptr().cast(),
        )
    };
    if status < 0 {
        return Err(anyhow!("Error reading enum data"));
    }

    let mut out: Vec<String> = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let start = i * elem_size;
        let end = start + elem_size;
        if end > buf.len() {
            out.push("(out of bounds)".to_string());
            continue;
        }
        out.push(format_enum(enum_type, &buf[start..end]));
    }

    ArrayD::from_shape_vec(IxDyn(&out_shape), out).map_err(|e| anyhow!(e))
}

fn plot_full_dataset_1d(ds: &Dataset) -> Result<Option<Vec<f64>>> {
    let shape = ds.shape();
    if shape.len() != 1 {
        return Ok(None);
    }
    let len = shape[0];
    if len < 2 {
        return Ok(None);
    }

    let dtype = ds.dtype()?;
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => {
            return Ok(None);
        }
    };

    match desc {
        TypeDescriptor::Integer(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                return Ok(None);
            }
            let arr: ArrayD<i64> = ds.read_slice::<i64, _, IxDyn>(Selection::new(..))?;
            Ok(plot_values_from_array(&arr))
        }
        TypeDescriptor::Unsigned(_) => {
            let size = dtype.size();
            if !is_standard_int_size(size) {
                return Ok(None);
            }
            let arr: ArrayD<u64> = ds.read_slice::<u64, _, IxDyn>(Selection::new(..))?;
            Ok(plot_values_from_array(&arr))
        }
        TypeDescriptor::Float(_) => {
            let arr: ArrayD<f64> = ds.read_slice::<f64, _, IxDyn>(Selection::new(..))?;
            Ok(plot_values_from_array(&arr))
        }
        _ => Ok(None),
    }
}

fn plot_values_from_array<T>(arr: &ArrayD<T>) -> Option<Vec<f64>>
where
    T: NumericFormat,
{
    if arr.ndim() != 1 || arr.len() < 2 {
        return None;
    }
    Some(arr.iter().map(|v| v.to_f64()).collect())
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

fn format_array_full<T: std::fmt::Debug>(arr: &ArrayD<T>) -> String {
    format!("{:?}", arr)
}

fn format_array_full_display(arr: &ArrayD<String>, quote_strings: bool) -> String {
    array_format::format_string_array_full(arr, quote_strings)
}

fn compound_has_vlen(compound: &CompoundType) -> bool {
    compound
        .fields
        .iter()
        .any(|field| descriptor_has_vlen(&field.ty))
}

fn descriptor_has_vlen(desc: &TypeDescriptor) -> bool {
    match desc {
        TypeDescriptor::VarLenArray(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode => true,
        TypeDescriptor::FixedArray(inner, _) => descriptor_has_vlen(inner),
        TypeDescriptor::Compound(compound) => compound
            .fields
            .iter()
            .any(|field| descriptor_has_vlen(&field.ty)),
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
    format!(
        "({})",
        idx.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
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

fn format_compound_value(desc: &TypeDescriptor, buf: &[u8]) -> String {
    debug_assert_eq!(desc.size(), buf.len());
    match desc {
        TypeDescriptor::Integer(size) => format_signed_int(buf, *size),
        TypeDescriptor::Unsigned(size) => format_unsigned_int(buf, *size),
        TypeDescriptor::Float(size) => format_float(buf, *size),
        TypeDescriptor::Boolean => format!("{}", buf.first().copied().unwrap_or(0) != 0),
        TypeDescriptor::Enum(enum_type) => format_enum(enum_type, buf),
        TypeDescriptor::FixedAscii(_) | TypeDescriptor::FixedUnicode(_) => {
            let s = utils::decode_fixed_bytes(buf, true);
            format!("{:?}", s)
        }
        TypeDescriptor::FixedArray(inner, len) => format_fixed_array(inner, *len, buf),
        TypeDescriptor::Compound(compound) => format_compound_struct(compound, buf),
        _ => "(unsupported)".to_string(),
    }
}

fn format_signed_int(buf: &[u8], size: IntSize) -> String {
    let value = match size {
        IntSize::U1 => i8::from_ne_bytes(read_bytes::<1>(buf)) as i64,
        IntSize::U2 => i16::from_ne_bytes(read_bytes::<2>(buf)) as i64,
        IntSize::U4 => i32::from_ne_bytes(read_bytes::<4>(buf)) as i64,
        IntSize::U8 => i64::from_ne_bytes(read_bytes::<8>(buf)),
    };
    format!("{:?}", value)
}

fn format_unsigned_int(buf: &[u8], size: IntSize) -> String {
    let value = match size {
        IntSize::U1 => u8::from_ne_bytes(read_bytes::<1>(buf)) as u64,
        IntSize::U2 => u16::from_ne_bytes(read_bytes::<2>(buf)) as u64,
        IntSize::U4 => u32::from_ne_bytes(read_bytes::<4>(buf)) as u64,
        IntSize::U8 => u64::from_ne_bytes(read_bytes::<8>(buf)),
    };
    format!("{:?}", value)
}

fn format_float(buf: &[u8], size: FloatSize) -> String {
    match size {
        FloatSize::U4 => format!("{:?}", f32::from_ne_bytes(read_bytes::<4>(buf))),
        FloatSize::U8 => format!("{:?}", f64::from_ne_bytes(read_bytes::<8>(buf))),
    }
}

fn format_enum(enum_type: &hdf5::types::EnumType, buf: &[u8]) -> String {
    let raw = read_unsigned(buf, enum_type.size);
    if let Some(member) = enum_type.members.iter().find(|m| m.value == raw) {
        return member.name.clone();
    }
    if enum_type.signed {
        format!("{:?}", sign_extend(raw, enum_type.size))
    } else {
        format!("{:?}", raw)
    }
}

fn format_fixed_array(inner: &TypeDescriptor, len: usize, buf: &[u8]) -> String {
    let elem_size = inner.size();
    if elem_size == 0 || len == 0 {
        return "[]".to_string();
    }

    let show_all = len <= ARRAY_EDGE * 2;
    let mut parts: Vec<String> = Vec::new();
    let mut idx = 0usize;
    while idx < len {
        if !show_all && idx == ARRAY_EDGE {
            parts.push("...".to_string());
            idx = len - ARRAY_EDGE;
            continue;
        }

        let start = idx * elem_size;
        let end = start + elem_size;
        if end > buf.len() {
            parts.push("(out of bounds)".to_string());
            break;
        }
        parts.push(format_compound_value(inner, &buf[start..end]));
        idx += 1;
    }
    format!("[{}]", parts.join(", "))
}

fn format_compound_struct(compound: &CompoundType, buf: &[u8]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(compound.fields.len());
    for field in &compound.fields {
        let start = field.offset;
        let end = start + field.ty.size();
        if end > buf.len() {
            parts.push(format!("{}: (out of bounds)", field.name));
            continue;
        }
        let value = format_compound_value(&field.ty, &buf[start..end]);
        parts.push(format!("{}: {}", field.name, value));
    }
    format!("{{{}}}", parts.join(", "))
}

fn read_unsigned(buf: &[u8], size: IntSize) -> u64 {
    match size {
        IntSize::U1 => u8::from_ne_bytes(read_bytes::<1>(buf)) as u64,
        IntSize::U2 => u16::from_ne_bytes(read_bytes::<2>(buf)) as u64,
        IntSize::U4 => u32::from_ne_bytes(read_bytes::<4>(buf)) as u64,
        IntSize::U8 => u64::from_ne_bytes(read_bytes::<8>(buf)),
    }
}

fn sign_extend(value: u64, size: IntSize) -> i64 {
    let bits = (size as u32) * 8;
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn read_bytes<const N: usize>(buf: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&buf[..N]);
    out
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
