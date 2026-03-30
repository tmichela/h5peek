use crate::array_format::{self, EllipsisConfig};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::Datatype;
use hdf5::H5Type;
use hdf5_sys::h5a::H5Aread;
use hdf5_sys::h5p::H5P_DEFAULT;
use hdf5_sys::h5t::{
    H5T_cset_t, H5T_str_t, H5Tclose, H5Tcopy, H5Tget_class, H5Tget_order, H5Tget_sign, H5Tget_size,
    H5Tset_cset, H5Tset_size, H5Tset_strpad, H5T_C_S1, H5T_FLOAT, H5T_INTEGER, H5T_ORDER_BE,
    H5T_SGN_NONE,
};
use ndarray::{ArrayD, IxDyn};
use std::ffi::CString;

#[derive(Clone, Copy, Debug)]
pub struct NumFormat {
    pub precision: usize,
    pub scientific: bool,
}

impl Default for NumFormat {
    fn default() -> Self {
        Self {
            precision: 5,
            scientific: false,
        }
    }
}

impl NumFormat {
    pub fn scalar() -> Self {
        Self {
            precision: usize::MAX,
            scientific: false,
        }
    }
}
#[allow(deprecated)]
pub fn get_object_addr(loc_id: i64) -> Option<u64> {
    use hdf5_sys::h5o::{H5O_info1_t, H5Oget_info1};
    unsafe {
        let mut info: H5O_info1_t = std::mem::zeroed();
        if H5Oget_info1(loc_id, &mut info) >= 0 {
            Some(info.addr)
        } else {
            None
        }
    }
}

pub enum LinkInfo {
    Hard,
    Soft(String),
    External { file: String, path: String },
    Other,
}

#[allow(deprecated)]
pub fn get_link_info(loc_id: i64, name: &str) -> LinkInfo {
    use hdf5_sys::h5l::{
        H5L_info1_t, H5Lget_info1, H5Lget_val, H5L_TYPE_EXTERNAL, H5L_TYPE_HARD, H5L_TYPE_SOFT,
    };

    let c_name = match CString::new(name) {
        Ok(c) => c,
        Err(_) => return LinkInfo::Other,
    };

    unsafe {
        let mut info: H5L_info1_t = std::mem::zeroed();
        let err = H5Lget_info1(loc_id, c_name.as_ptr(), &mut info, H5P_DEFAULT);
        if err < 0 {
            return LinkInfo::Other;
        }

        match info.type_ {
            H5L_TYPE_HARD => LinkInfo::Hard,
            H5L_TYPE_SOFT => {
                let size = *info.u.val_size();
                let mut buf: Vec<u8> = vec![0; size + 1];
                H5Lget_val(
                    loc_id,
                    c_name.as_ptr(),
                    buf.as_mut_ptr() as *mut _,
                    size,
                    H5P_DEFAULT,
                );
                // Remove trailing nulls/garbage if any, CString::from_vec_with_nul handles one null.
                // The buf size from val_size usually includes null terminator for soft links?
                // Or we can just parse up to first null.
                let s = parse_null_terminated(&buf);
                LinkInfo::Soft(s)
            }
            H5L_TYPE_EXTERNAL => {
                let size = *info.u.val_size();
                let mut buf: Vec<u8> = vec![0; size + 1];
                H5Lget_val(
                    loc_id,
                    c_name.as_ptr(),
                    buf.as_mut_ptr() as *mut _,
                    size,
                    H5P_DEFAULT,
                );

                // External link value: filename \0 path \0
                let full = buf;
                let mut parts = full.split(|&b| b == 0).filter(|p| !p.is_empty());
                let file = parts
                    .next()
                    .map(|p| String::from_utf8_lossy(p).into_owned())
                    .unwrap_or_default();
                let path = parts
                    .next()
                    .map(|p| String::from_utf8_lossy(p).into_owned())
                    .unwrap_or_default();
                LinkInfo::External { file, path }
            }
            _ => LinkInfo::Other,
        }
    }
}

fn parse_null_terminated(buf: &[u8]) -> String {
    buf.iter()
        .position(|&b| b == 0)
        .map(|pos| String::from_utf8_lossy(&buf[..pos]).into_owned())
        .unwrap_or_else(|| String::from_utf8_lossy(buf).into_owned())
}

pub fn fmt_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(" × ")
}

pub fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{} B", bytes)
    } else if value >= 10.0 {
        format!("{:.0} {}", value, UNITS[unit])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}

pub fn fmt_i64(value: i64, _fmt: &NumFormat) -> String {
    value.to_string()
}

pub fn fmt_u64(value: u64, _fmt: &NumFormat) -> String {
    value.to_string()
}

pub fn fmt_f64(value: f64, fmt: &NumFormat) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if fmt.precision == usize::MAX && !fmt.scientific {
        return value.to_string();
    }
    if fmt.scientific {
        return format!("{:.*e}", fmt.precision, value);
    }
    format!("{:.*}", fmt.precision, value)
}

pub fn fmt_maxshape(shape: &[Option<usize>]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape
        .iter()
        .map(|s| match s {
            Some(v) => v.to_string(),
            None => "unlimited".to_string(),
        })
        .collect::<Vec<_>>()
        .join(" × ")
}

pub fn fmt_dtype(dtype: &Datatype) -> String {
    match dtype.to_descriptor() {
        Ok(desc) => fmt_descriptor_short(&desc, dtype),
        Err(_) => fmt_dtype_fallback(dtype),
    }
}

fn fmt_dtype_fallback(dtype: &Datatype) -> String {
    unsafe {
        let id = dtype.id();
        let class = H5Tget_class(id);
        let size = H5Tget_size(id);
        let suffix = get_endian_suffix(dtype);

        if class == H5T_INTEGER {
            let sign = H5Tget_sign(id);
            let sign_str = if sign == H5T_SGN_NONE {
                "unsigned"
            } else {
                "signed"
            };
            format!("{}-byte {} integer{}", size, sign_str, suffix)
        } else if class == H5T_FLOAT {
            format!("custom {}-byte float{}", size, suffix)
        } else {
            format!("unknown {}-byte type", size)
        }
    }
}

#[allow(dead_code)]
pub fn dtype_description(dtype: &Datatype) -> Option<String> {
    match dtype.to_descriptor() {
        Ok(desc) => dtype_description_from_desc(&desc, dtype),
        Err(_) => None,
    }
}

fn get_endian_suffix(dtype: &Datatype) -> String {
    unsafe {
        let order = H5Tget_order(dtype.id());
        if order == H5T_ORDER_BE {
            " (big-endian)".to_string()
        } else {
            String::new() // Little endian is default/standard usually
        }
    }
}

fn is_custom_size(dtype: &Datatype, standard_size_bytes: usize) -> bool {
    unsafe {
        let real_size = H5Tget_size(dtype.id());
        real_size != standard_size_bytes
    }
}

fn get_real_size(dtype: &Datatype) -> usize {
    unsafe { H5Tget_size(dtype.id()) }
}

#[allow(dead_code)]
fn dtype_description_from_desc(desc: &TypeDescriptor, dtype: &Datatype) -> Option<String> {
    match desc {
        TypeDescriptor::Integer(size) => {
            let bits = int_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                Some(format!(
                    "custom {}-byte signed integer{}",
                    get_real_size(dtype),
                    suffix
                ))
            } else {
                Some(format!("{}-bit signed integer{}", bits, suffix))
            }
        }
        TypeDescriptor::Unsigned(size) => {
            let bits = int_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                Some(format!(
                    "custom {}-byte unsigned integer{}",
                    get_real_size(dtype),
                    suffix
                ))
            } else {
                Some(format!("{}-bit unsigned integer{}", bits, suffix))
            }
        }
        TypeDescriptor::Float(size) => {
            let bits = float_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                Some(format!(
                    "custom {}-byte floating point{}",
                    get_real_size(dtype),
                    suffix
                ))
            } else {
                Some(format!("{}-bit floating point{}", bits, suffix))
            }
        }
        TypeDescriptor::Reference(r) => Some(match r {
            hdf5::types::Reference::Object => "object reference".to_string(),
            hdf5::types::Reference::Region => "region reference".to_string(),
            _ => "reference".to_string(),
        }),
        _ => None,
    }
}

fn fmt_descriptor_short(desc: &TypeDescriptor, dtype: &Datatype) -> String {
    match desc {
        TypeDescriptor::Integer(size) => {
            let bits = int_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                format!("{}-byte signed integer{}", get_real_size(dtype), suffix)
            } else {
                format!("int{}{}", bits, suffix)
            }
        }
        TypeDescriptor::Unsigned(size) => {
            let bits = int_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                format!("{}-byte unsigned integer{}", get_real_size(dtype), suffix)
            } else {
                format!("uint{}{}", bits, suffix)
            }
        }
        TypeDescriptor::Float(size) => {
            let bits = float_size_to_bits(*size);
            let std_bytes = (bits / 8) as usize;
            let suffix = get_endian_suffix(dtype);
            if is_custom_size(dtype, std_bytes) {
                format!("custom {}-byte float{}", get_real_size(dtype), suffix)
            } else {
                format!("float{}{}", bits, suffix)
            }
        }
        TypeDescriptor::Boolean => "bool".to_string(),
        TypeDescriptor::Enum(e) => {
            if e.members.len() >= 5 {
                format!("enum ({} options)", e.members.len())
            } else {
                let options: Vec<String> = e.members.iter().map(|m| m.name.clone()).collect();
                format!("enum ({})", options.join(", "))
            }
        }
        TypeDescriptor::Compound(c) => {
            let fields: Vec<String> = c
                .fields
                .iter()
                .map(|f| {
                    // Recursive call needs a Datatype. But CompoundField only has TypeDescriptor.
                    // We CANNOT easily get the inner Datatype from just TypeDescriptor here without reconstructing it or assuming default props.
                    // This is a limitation of the current recursion approach.
                    // However, for nested types in Compound, endianness is usually inherited or specified per field?
                    // `f.ty` is a TypeDescriptor. It doesn't carry the raw ID.
                    // So we can't call H5Tget_order on `f.ty`.
                    // We'll have to fall back to the simple formatter for nested types inside Compound/Array
                    // unless we change how we traverse.
                    // For now, let's use a version of fmt that doesn't require Datatype for nested items.
                    format!("{}: {}", f.name, fmt_descriptor_short_nodefs(&f.ty))
                })
                .collect();
            format!("({})", fields.join(", "))
        }
        TypeDescriptor::FixedArray(ty, len) => {
            let mut dims = vec![*len];
            let mut inner = ty;
            while let TypeDescriptor::FixedArray(next_ty, next_len) = inner.as_ref() {
                dims.push(*next_len);
                inner = next_ty;
            }
            let shape_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(" × ");
            format!(
                "{} array of {}",
                shape_str,
                fmt_descriptor_short_nodefs(inner)
            )
        }
        TypeDescriptor::FixedAscii(len) => format!("{}-byte ASCII string", len),
        TypeDescriptor::FixedUnicode(len) => format!("{}-byte UTF-8 string", len),
        TypeDescriptor::VarLenArray(ty) => {
            format!("vlen array of {}", fmt_descriptor_short_nodefs(ty))
        }
        TypeDescriptor::VarLenAscii => "ASCII string".to_string(),
        TypeDescriptor::VarLenUnicode => "UTF-8 string".to_string(),
        TypeDescriptor::Reference(r) => match r {
            hdf5::types::Reference::Object => "obj-ref".to_string(),
            hdf5::types::Reference::Region => "reg-ref".to_string(),
            _ => "ref".to_string(),
        },
    }
}

// Version of fmt_descriptor that works on pure TypeDescriptors (for nested types where we lack Datatype object)
fn fmt_descriptor_short_nodefs(desc: &TypeDescriptor) -> String {
    match desc {
        TypeDescriptor::Integer(size) => format!("int{}", int_size_to_bits(*size)),
        TypeDescriptor::Unsigned(size) => format!("uint{}", int_size_to_bits(*size)),
        TypeDescriptor::Float(size) => format!("float{}", float_size_to_bits(*size)),
        TypeDescriptor::Boolean => "bool".to_string(),
        TypeDescriptor::Enum(e) => {
            if e.members.len() >= 5 {
                format!("enum ({} options)", e.members.len())
            } else {
                let options: Vec<String> = e.members.iter().map(|m| m.name.clone()).collect();
                format!("enum ({})", options.join(", "))
            }
        }
        TypeDescriptor::Compound(c) => {
            let fields: Vec<String> = c
                .fields
                .iter()
                .map(|f| format!("{}: {}", f.name, fmt_descriptor_short_nodefs(&f.ty)))
                .collect();
            format!("({})", fields.join(", "))
        }
        TypeDescriptor::FixedArray(ty, len) => {
            let mut dims = vec![*len];
            let mut inner = ty;
            while let TypeDescriptor::FixedArray(next_ty, next_len) = inner.as_ref() {
                dims.push(*next_len);
                inner = next_ty;
            }
            let shape_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(" × ");
            format!(
                "{} array of {}",
                shape_str,
                fmt_descriptor_short_nodefs(inner)
            )
        }
        TypeDescriptor::FixedAscii(len) => format!("{}-byte ASCII string", len),
        TypeDescriptor::FixedUnicode(len) => format!("{}-byte UTF-8 string", len),
        TypeDescriptor::VarLenArray(ty) => {
            format!("vlen array of {}", fmt_descriptor_short_nodefs(ty))
        }
        TypeDescriptor::VarLenAscii => "ASCII string".to_string(),
        TypeDescriptor::VarLenUnicode => "UTF-8 string".to_string(),
        TypeDescriptor::Reference(r) => match r {
            hdf5::types::Reference::Object => "obj-ref".to_string(),
            hdf5::types::Reference::Region => "reg-ref".to_string(),
            _ => "ref".to_string(),
        },
    }
}

const MAX_ATTR_STRING_ELEMS: usize = 200;
const MAX_ATTR_ARRAY_ELEMS: usize = 10;
const ATTR_ARRAY_EDGE: usize = 3;
const ATTR_ARRAY_FORMAT: EllipsisConfig = EllipsisConfig {
    max_elems: MAX_ATTR_STRING_ELEMS,
    edge: ATTR_ARRAY_EDGE,
};

pub fn format_attribute_value(
    attr: &hdf5::Attribute,
    fmt: &NumFormat,
    truncate_strings: bool,
) -> String {
    let dtype = match attr.dtype() {
        Ok(dt) => dt,
        Err(_) => return "unreadable".to_string(),
    };
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => return format!("[{}]", fmt_dtype(&dtype)),
    };

    let shape = attr.shape();

    if let Some(value) = format_string_attribute(attr, &desc, &shape, truncate_strings) {
        return value;
    }

    if !shape.is_empty() {
        if let Some(value) = format_numeric_attribute_array(attr, &desc, &shape, fmt) {
            return value;
        }
        return format!("array [{}: {}]", fmt_dtype(&dtype), fmt_shape(&shape));
    }

    match desc {
        TypeDescriptor::Integer(_) => attr
            .read_scalar::<i64>()
            .map(|v| fmt_i64(v, fmt))
            .unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Unsigned(_) => attr
            .read_scalar::<u64>()
            .map(|v| fmt_u64(v, fmt))
            .unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Float(_) => attr
            .read_scalar::<f64>()
            .map(|v| fmt_f64(v, fmt))
            .unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Boolean => attr
            .read_scalar::<bool>()
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode
        | TypeDescriptor::FixedAscii(_)
        | TypeDescriptor::FixedUnicode(_) => "unreadable".to_string(),
        _ => format!("[{}]", fmt_dtype(&dtype)),
    }
}

fn format_string_attribute(
    attr: &hdf5::Attribute,
    desc: &TypeDescriptor,
    shape: &[usize],
    truncate_strings: bool,
) -> Option<String> {
    match desc {
        TypeDescriptor::VarLenAscii => {
            format_varlen_attr::<VarLenAscii>(attr, shape, truncate_strings)
        }
        TypeDescriptor::VarLenUnicode => {
            format_varlen_attr::<VarLenUnicode>(attr, shape, truncate_strings)
        }
        TypeDescriptor::FixedAscii(len) => format_fixed_string_attr(
            attr,
            *len,
            shape,
            truncate_strings,
            H5T_cset_t::H5T_CSET_ASCII,
        )
        .or_else(|| format_varlen_attr::<VarLenUnicode>(attr, shape, truncate_strings)),
        TypeDescriptor::FixedUnicode(len) => format_fixed_string_attr(
            attr,
            *len,
            shape,
            truncate_strings,
            H5T_cset_t::H5T_CSET_UTF8,
        )
        .or_else(|| format_varlen_attr::<VarLenUnicode>(attr, shape, truncate_strings)),
        _ => None,
    }
}

fn format_numeric_attribute_array(
    attr: &hdf5::Attribute,
    desc: &TypeDescriptor,
    shape: &[usize],
    fmt: &NumFormat,
) -> Option<String> {
    let total = total_size_checked(shape)?;
    if total > MAX_ATTR_ARRAY_ELEMS {
        return None;
    }

    match desc {
        TypeDescriptor::Integer(_) => {
            let arr: ArrayD<i64> = attr.read_dyn().ok()?;
            let values: Vec<String> = arr.iter().map(|v| fmt_i64(*v, fmt)).collect();
            format_array_display(values, shape, false, MAX_ATTR_ARRAY_ELEMS)
        }
        TypeDescriptor::Unsigned(_) => {
            let arr: ArrayD<u64> = attr.read_dyn().ok()?;
            let values: Vec<String> = arr.iter().map(|v| fmt_u64(*v, fmt)).collect();
            format_array_display(values, shape, false, MAX_ATTR_ARRAY_ELEMS)
        }
        TypeDescriptor::Float(_) => {
            let arr: ArrayD<f64> = attr.read_dyn().ok()?;
            let values: Vec<String> = arr.iter().map(|v| fmt_f64(*v, fmt)).collect();
            format_array_display(values, shape, false, MAX_ATTR_ARRAY_ELEMS)
        }
        TypeDescriptor::Boolean => {
            let arr: ArrayD<bool> = attr.read_dyn().ok()?;
            let values: Vec<String> = arr.iter().map(|v| v.to_string()).collect();
            format_array_display(values, shape, false, MAX_ATTR_ARRAY_ELEMS)
        }
        _ => None,
    }
}

fn format_varlen_attr<T>(
    attr: &hdf5::Attribute,
    shape: &[usize],
    truncate_strings: bool,
) -> Option<String>
where
    T: H5Type + AsRef<str>,
{
    if shape.is_empty() {
        let v: T = attr.read_scalar().ok()?;
        return Some(format_scalar_string(v.as_ref(), truncate_strings));
    }
    if string_array_too_large(shape) {
        return None;
    }
    let vec: Vec<T> = attr.read_raw().ok()?;
    let values: Vec<String> = vec
        .iter()
        .map(|v| maybe_truncate_string(v.as_ref(), truncate_strings))
        .collect();
    format_string_array(values, shape)
}

fn format_fixed_string_attr(
    attr: &hdf5::Attribute,
    len: usize,
    shape: &[usize],
    truncate_strings: bool,
    cset: H5T_cset_t,
) -> Option<String> {
    if !shape.is_empty() && string_array_too_large(shape) {
        return None;
    }
    let values = read_fixed_string_attr(attr, len, shape, cset)?;
    if shape.is_empty() {
        return values
            .first()
            .map(|v| format_scalar_string(v, truncate_strings));
    }
    let truncated: Vec<String> = values
        .into_iter()
        .map(|v| maybe_truncate_string(&v, truncate_strings))
        .collect();
    format_string_array(truncated, shape)
}

fn read_fixed_string_attr(
    attr: &hdf5::Attribute,
    len: usize,
    shape: &[usize],
    cset: H5T_cset_t,
) -> Option<Vec<String>> {
    let total = if shape.is_empty() {
        1
    } else {
        total_size_checked(shape)?
    };
    if len == 0 {
        return Some(vec![String::new(); total]);
    }

    let mut elem_len = len;
    let mut buf: Vec<u8> = Vec::new();
    let mut read_ok = false;

    if let Ok(dtype) = attr.dtype() {
        let file_len = dtype.size();
        if file_len == 0 {
            return Some(vec![String::new(); total]);
        }
        elem_len = file_len;
        buf.resize(total.saturating_mul(elem_len), 0);
        let status = unsafe { H5Aread(attr.id(), dtype.id(), buf.as_mut_ptr().cast()) };
        if status >= 0 {
            read_ok = true;
        }
    }

    if !read_ok {
        elem_len = len.max(1);
        buf.resize(total.saturating_mul(elem_len), 0);

        let mem_type = unsafe { H5Tcopy(*H5T_C_S1) };
        if mem_type < 0 {
            return None;
        }
        if unsafe { H5Tset_size(mem_type, elem_len as _) } < 0 {
            unsafe { H5Tclose(mem_type) };
            return None;
        }

        // Best-effort: some HDF5 builds reject these, but the default is fine.
        let _ = unsafe { H5Tset_cset(mem_type, cset) };
        let _ = unsafe { H5Tset_strpad(mem_type, H5T_str_t::H5T_STR_NULLTERM) };

        let status = unsafe { H5Aread(attr.id(), mem_type, buf.as_mut_ptr().cast()) };
        unsafe { H5Tclose(mem_type) };
        if status < 0 {
            return None;
        }
    }

    let mut out: Vec<String> = Vec::with_capacity(total);
    for i in 0..total {
        let start = i * elem_len;
        let end = start + elem_len;
        if end > buf.len() {
            return None;
        }
        out.push(decode_fixed_bytes(&buf[start..end], true));
    }

    Some(out)
}

pub(crate) fn decode_fixed_bytes(bytes: &[u8], trim_spaces: bool) -> String {
    if let Some(end) = bytes.iter().position(|b| *b == 0) {
        return String::from_utf8_lossy(&bytes[..end]).into_owned();
    }
    let mut end = bytes.len();
    if trim_spaces {
        while end > 0 && bytes[end - 1] == b' ' {
            end -= 1;
        }
    }
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn format_array_display(
    values: Vec<String>,
    shape: &[usize],
    quote_strings: bool,
    max_elems: usize,
) -> Option<String> {
    let total = total_size_checked(shape)?;
    if total != values.len() {
        return None;
    }
    if total > max_elems {
        return None;
    }
    let arr = ArrayD::from_shape_vec(IxDyn(shape), values).ok()?;
    Some(array_format::format_string_array_with_ellipsis(
        &arr,
        ATTR_ARRAY_FORMAT,
        quote_strings,
    ))
}

fn format_string_array(values: Vec<String>, shape: &[usize]) -> Option<String> {
    format_array_display(values, shape, true, MAX_ATTR_STRING_ELEMS)
}

fn string_array_too_large(shape: &[usize]) -> bool {
    if shape.is_empty() {
        return false;
    }
    match total_size_checked(shape) {
        Some(size) => size > MAX_ATTR_STRING_ELEMS,
        None => true,
    }
}

fn format_scalar_string(value: &str, truncate: bool) -> String {
    if truncate && value.len() > 50 {
        let head = utf8_prefix_by_bytes(value, 20);
        let tail = utf8_suffix_by_bytes(value, 20);
        format!("{}...{}", head, tail)
    } else {
        format!("'{}'", value)
    }
}

fn maybe_truncate_string(value: &str, truncate: bool) -> String {
    if truncate && value.len() > 50 {
        let head = utf8_prefix_by_bytes(value, 20);
        let tail = utf8_suffix_by_bytes(value, 20);
        format!("{}...{}", head, tail)
    } else {
        value.to_string()
    }
}

fn total_size_checked(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

fn utf8_prefix_by_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }

    let mut end = 0;
    for (idx, ch) in s.char_indices() {
        let next = idx + ch.len_utf8();
        if next > max_bytes {
            break;
        }
        end = next;
    }

    &s[..end]
}

fn utf8_suffix_by_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }

    let target = s.len() - max_bytes;
    for (idx, _) in s.char_indices() {
        if idx >= target {
            return &s[idx..];
        }
    }

    s
}

fn int_size_to_bits(size: IntSize) -> u32 {
    match size {
        IntSize::U1 => 8,
        IntSize::U2 => 16,
        IntSize::U4 => 32,
        IntSize::U8 => 64,
    }
}

fn float_size_to_bits(size: FloatSize) -> u32 {
    match size {
        FloatSize::U4 => 32,
        FloatSize::U8 => 64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::{CompoundField, CompoundType, EnumType, FixedAscii, VarLenUnicode};
    use ndarray::array;
    use std::str::FromStr;
    use std::sync::Mutex;
    use std::time::{SystemTime, UNIX_EPOCH};

    static HDF5_LOCK: Mutex<()> = Mutex::new(());

    fn with_hdf5_lock(f: impl FnOnce()) {
        let _guard = HDF5_LOCK.lock().unwrap();
        f();
    }

    #[test]
    fn test_standard_float() {
        let desc = TypeDescriptor::Float(FloatSize::U4);
        assert_eq!(fmt_descriptor_short_nodefs(&desc), "float32");
    }

    #[test]
    fn test_standard_int() {
        let i4 = TypeDescriptor::Integer(IntSize::U4);
        assert_eq!(fmt_descriptor_short_nodefs(&i4), "int32");

        let u8 = TypeDescriptor::Unsigned(IntSize::U8);
        assert_eq!(fmt_descriptor_short_nodefs(&u8), "uint64");
    }

    #[test]
    fn test_string() {
        // vlen string
        let vst = TypeDescriptor::VarLenUnicode;
        assert_eq!(fmt_descriptor_short_nodefs(&vst), "UTF-8 string");

        // fixed-length string
        let fst = TypeDescriptor::FixedAscii(3);
        assert_eq!(fmt_descriptor_short_nodefs(&fst), "3-byte ASCII string");
    }

    #[test]
    fn test_compound() {
        let ct = TypeDescriptor::Compound(CompoundType {
            fields: vec![
                CompoundField {
                    name: "x".to_string(),
                    ty: TypeDescriptor::Float(FloatSize::U4),
                    offset: 0,
                    index: 0,
                },
                CompoundField {
                    name: "y".to_string(),
                    ty: TypeDescriptor::Float(FloatSize::U4),
                    offset: 4,
                    index: 1,
                },
            ],
            size: 8,
        });
        assert_eq!(fmt_descriptor_short_nodefs(&ct), "(x: float32, y: float32)");
    }

    #[test]
    fn test_enum() {
        let et = TypeDescriptor::Enum(EnumType {
            members: vec![
                hdf5::types::EnumMember {
                    name: "apple".to_string(),
                    value: 1,
                },
                hdf5::types::EnumMember {
                    name: "banana".to_string(),
                    value: 2,
                },
            ],
            size: IntSize::U1,
            signed: false,
        });
        assert_eq!(fmt_descriptor_short_nodefs(&et), "enum (apple, banana)");
    }

    #[test]
    fn test_vlen() {
        let vt = TypeDescriptor::VarLenArray(Box::new(TypeDescriptor::Integer(IntSize::U2)));
        assert_eq!(fmt_descriptor_short_nodefs(&vt), "vlen array of int16");
    }

    #[test]
    fn test_endianness_from_file() {
        with_hdf5_lock(|| {
            use hdf5::File;
            use std::path::PathBuf;

            // Use the sample file we generated
            let path = PathBuf::from("data/sample.h5");
            if !path.exists() {
                return; // Skip if file not found (e.g. in environments where it wasn't generated)
            }

            let file = File::open(&path).unwrap();
            let ds = file.dataset("custom_types/int32_be").unwrap();
            let dtype = ds.dtype().unwrap();

            assert!(fmt_dtype(&dtype).contains("(big-endian)"));

            // Test Custom 6-byte Integer
            if let Ok(ds_custom) = file.dataset("custom_types/int48") {
                let dtype_custom = ds_custom.dtype().unwrap();
                assert_eq!(fmt_dtype(&dtype_custom), "6-byte signed integer");
            }
        });
    }

    #[test]
    fn test_fmt_bytes() {
        assert_eq!(fmt_bytes(0), "0 B");
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(1024), "1.0 KiB");
        assert_eq!(fmt_bytes(10 * 1024), "10 KiB");
        assert_eq!(fmt_bytes(1536), "1.5 KiB");
    }

    #[test]
    fn test_number_formatting() {
        let fmt = NumFormat {
            precision: 2,
            scientific: false,
        };
        assert_eq!(fmt_i64(1234567, &fmt), "1234567");
        assert_eq!(fmt_u64(42, &fmt), "42");
        assert_eq!(fmt_f64(1234.5, &fmt), "1234.50");

        let fmt_scientific = NumFormat {
            precision: 3,
            scientific: true,
        };
        assert_eq!(fmt_f64(1234.5, &fmt_scientific), "1.234e3");
    }

    #[test]
    fn test_format_attribute_value_utf8_truncation_safe() {
        with_hdf5_lock(|| {
            use hdf5::File;

            let mut path = std::env::temp_dir();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!(
                "h5peek_attr_utf8_{}_{}.h5",
                std::process::id(),
                nanos
            ));

            let file = File::create(&path).unwrap();
            let attr = file
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("utf8_attr")
                .unwrap();

            let unit = "\u{00E9}";
            let long = unit.repeat(60);
            let v = VarLenUnicode::from_str(&long).unwrap();
            attr.as_writer().write_scalar(&v).unwrap();

            let formatted = format_attribute_value(&attr, &NumFormat::default(), true);
            let expected = format!("{}...{}", unit.repeat(10), unit.repeat(10));
            assert_eq!(formatted, expected);

            drop(file);
            let _ = std::fs::remove_file(path);
        });
    }

    #[test]
    fn test_format_attribute_value_string_array_truncation_toggle() {
        with_hdf5_lock(|| {
            use hdf5::File;

            let mut path = std::env::temp_dir();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!(
                "h5peek_attr_utf8_arr_{}_{}.h5",
                std::process::id(),
                nanos
            ));

            let file = File::create(&path).unwrap();
            let attr = file
                .new_attr::<VarLenUnicode>()
                .shape((1, 2))
                .create("utf8_arr")
                .unwrap();

            let long = "a".repeat(60);
            let arr = array![[
                VarLenUnicode::from_str("alpha").unwrap(),
                VarLenUnicode::from_str(&long).unwrap(),
            ]];
            attr.as_writer().write(&arr).unwrap();

            let formatted_trunc = format_attribute_value(&attr, &NumFormat::default(), true);
            let head = "a".repeat(20);
            let tail = "a".repeat(20);
            let expected = format!("[\n  [\"alpha\", \"{}...{}\"]\n]", head, tail);
            assert_eq!(formatted_trunc, expected);

            let formatted_full = format_attribute_value(&attr, &NumFormat::default(), false);
            assert!(formatted_full.contains(&long));
            assert!(!formatted_full.contains("..."));

            drop(file);
            let _ = std::fs::remove_file(path);
        });
    }

    #[test]
    fn test_format_attribute_value_fixed_ascii_array() {
        with_hdf5_lock(|| {
            use hdf5::File;

            let mut path = std::env::temp_dir();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!(
                "h5peek_attr_ascii_arr_{}_{}.h5",
                std::process::id(),
                nanos
            ));

            let file = File::create(&path).unwrap();
            let attr = file
                .new_attr::<FixedAscii<8>>()
                .shape((1, 2))
                .create("ascii_arr")
                .unwrap();

            let arr = array![[
                FixedAscii::<8>::from_ascii(b"foo").unwrap(),
                FixedAscii::<8>::from_ascii(b"bar").unwrap(),
            ]];
            attr.as_writer().write(&arr).unwrap();

            let formatted = format_attribute_value(&attr, &NumFormat::default(), true);
            let expected = "[\n  [\"foo\", \"bar\"]\n]";
            assert_eq!(formatted, expected);

            drop(file);
            let _ = std::fs::remove_file(path);
        });
    }

    #[test]
    fn test_format_attribute_value_numeric_array() {
        with_hdf5_lock(|| {
            use hdf5::File;

            let mut path = std::env::temp_dir();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!(
                "h5peek_attr_num_arr_{}_{}.h5",
                std::process::id(),
                nanos
            ));

            let file = File::create(&path).unwrap();
            let attr = file
                .new_attr::<u64>()
                .shape((1, 3))
                .create("num_arr")
                .unwrap();

            let arr = array![[1u64, 2, 3]];
            attr.as_writer().write(&arr).unwrap();

            let formatted = format_attribute_value(&attr, &NumFormat::default(), true);
            let expected = "[\n  [1, 2, 3]\n]";
            assert_eq!(formatted, expected);

            drop(file);
            let _ = std::fs::remove_file(path);
        });
    }
}
