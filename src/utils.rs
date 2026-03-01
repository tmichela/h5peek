use hdf5::Datatype;
use hdf5::types::{TypeDescriptor, IntSize, FloatSize, VarLenUnicode};
use hdf5_sys::h5t::{H5Tget_order, H5Tget_size, H5Tget_class, H5Tget_sign, H5T_ORDER_BE, H5T_INTEGER, H5T_FLOAT, H5T_SGN_NONE};
use hdf5_sys::h5p::H5P_DEFAULT;
use std::ffi::CString;

#[allow(deprecated)]
pub fn get_object_addr(loc_id: i64) -> Option<u64> {
    use hdf5_sys::h5o::{H5Oget_info1, H5O_info1_t};
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
    use hdf5_sys::h5l::{H5Lget_info1, H5Lget_val, H5L_info1_t, H5L_TYPE_SOFT, H5L_TYPE_EXTERNAL, H5L_TYPE_HARD};

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
                H5Lget_val(loc_id, c_name.as_ptr(), buf.as_mut_ptr() as *mut _, size, H5P_DEFAULT);
                // Remove trailing nulls/garbage if any, CString::from_vec_with_nul handles one null.
                // The buf size from val_size usually includes null terminator for soft links?
                // Or we can just parse up to first null.
                let s = parse_null_terminated(&buf);
                LinkInfo::Soft(s)
            },
            H5L_TYPE_EXTERNAL => {
                let size = *info.u.val_size();
                let mut buf: Vec<u8> = vec![0; size + 1];
                H5Lget_val(loc_id, c_name.as_ptr(), buf.as_mut_ptr() as *mut _, size, H5P_DEFAULT);

                // External link value: filename \0 path \0
                let full = buf;
                let mut parts = full.split(|&b| b == 0).filter(|p| !p.is_empty());
                let file = parts.next().map(|p| String::from_utf8_lossy(p).into_owned()).unwrap_or_default();
                let path = parts.next().map(|p| String::from_utf8_lossy(p).into_owned()).unwrap_or_default();
                LinkInfo::External { file, path }
            },
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
    shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" × ")
}

pub fn fmt_maxshape(shape: &[Option<usize>]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape.iter().map(|s| match s {
        Some(v) => v.to_string(),
        None => "unlimited".to_string(),
    }).collect::<Vec<_>>().join(" × ")
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
            let sign_str = if sign == H5T_SGN_NONE { "unsigned" } else { "signed" };
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
                 Some(format!("custom {}-byte signed integer{}", get_real_size(dtype), suffix))
             } else {
                 Some(format!("{}-bit signed integer{}", bits, suffix))
             }
        },
        TypeDescriptor::Unsigned(size) => {
             let bits = int_size_to_bits(*size);
             let std_bytes = (bits / 8) as usize;
             let suffix = get_endian_suffix(dtype);
             if is_custom_size(dtype, std_bytes) {
                 Some(format!("custom {}-byte unsigned integer{}", get_real_size(dtype), suffix))
             } else {
                 Some(format!("{}-bit unsigned integer{}", bits, suffix))
             }
        },
        TypeDescriptor::Float(size) => {
             let bits = float_size_to_bits(*size);
             let std_bytes = (bits / 8) as usize;
             let suffix = get_endian_suffix(dtype);
             if is_custom_size(dtype, std_bytes) {
                 Some(format!("custom {}-byte floating point{}", get_real_size(dtype), suffix))
             } else {
                 Some(format!("{}-bit floating point{}", bits, suffix))
             }
        },
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
        },
        TypeDescriptor::Unsigned(size) => {
             let bits = int_size_to_bits(*size);
             let std_bytes = (bits / 8) as usize;
             let suffix = get_endian_suffix(dtype);
             if is_custom_size(dtype, std_bytes) {
                 format!("{}-byte unsigned integer{}", get_real_size(dtype), suffix)
             } else {
                 format!("uint{}{}", bits, suffix)
             }
        },
        TypeDescriptor::Float(size) => {
             let bits = float_size_to_bits(*size);
             let std_bytes = (bits / 8) as usize;
             let suffix = get_endian_suffix(dtype);
             if is_custom_size(dtype, std_bytes) {
                 format!("custom {}-byte float{}", get_real_size(dtype), suffix)
             } else {
                 format!("float{}{}", bits, suffix)
             }
        },
        TypeDescriptor::Boolean => "bool".to_string(),
        TypeDescriptor::Enum(e) => {
            if e.members.len() >= 5 {
                format!("enum ({} options)", e.members.len())
            } else {
                 let options: Vec<String> = e.members.iter().map(|m| m.name.clone()).collect();
                 format!("enum ({})", options.join(", "))
            }
        },
        TypeDescriptor::Compound(c) => {
            let fields: Vec<String> = c.fields.iter().map(|f| {
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
            }).collect();
            format!("({})", fields.join(", "))
        },
        TypeDescriptor::FixedArray(ty, len) => {
             let mut dims = vec![*len];
             let mut inner = ty;
             while let TypeDescriptor::FixedArray(next_ty, next_len) = inner.as_ref() {
                 dims.push(*next_len);
                 inner = next_ty;
             }
             let shape_str = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ");
             format!("{} array of {}", shape_str, fmt_descriptor_short_nodefs(inner))
        },
        TypeDescriptor::FixedAscii(len) => format!("{}-byte ASCII string", len),
        TypeDescriptor::FixedUnicode(len) => format!("{}-byte UTF-8 string", len),
        TypeDescriptor::VarLenArray(ty) => format!("vlen array of {}", fmt_descriptor_short_nodefs(ty)),
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
        },
        TypeDescriptor::Compound(c) => {
            let fields: Vec<String> = c.fields.iter().map(|f| {
                format!("{}: {}", f.name, fmt_descriptor_short_nodefs(&f.ty))
            }).collect();
            format!("({})", fields.join(", "))
        },
        TypeDescriptor::FixedArray(ty, len) => {
             let mut dims = vec![*len];
             let mut inner = ty;
             while let TypeDescriptor::FixedArray(next_ty, next_len) = inner.as_ref() {
                 dims.push(*next_len);
                 inner = next_ty;
             }
             let shape_str = dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(" × ");
             format!("{} array of {}", shape_str, fmt_descriptor_short_nodefs(inner))
        },
        TypeDescriptor::FixedAscii(len) => format!("{}-byte ASCII string", len),
        TypeDescriptor::FixedUnicode(len) => format!("{}-byte UTF-8 string", len),
        TypeDescriptor::VarLenArray(ty) => format!("vlen array of {}", fmt_descriptor_short_nodefs(ty)),
        TypeDescriptor::VarLenAscii => "ASCII string".to_string(),
        TypeDescriptor::VarLenUnicode => "UTF-8 string".to_string(),
        TypeDescriptor::Reference(r) => match r {
            hdf5::types::Reference::Object => "obj-ref".to_string(),
            hdf5::types::Reference::Region => "reg-ref".to_string(),
            _ => "ref".to_string(),
        },
    }
}

pub fn format_attribute_value(attr: &hdf5::Attribute) -> String {
    let dtype = match attr.dtype() {
        Ok(dt) => dt,
        Err(_) => return "unreadable".to_string(),
    };
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => return format!("[{}]", fmt_dtype(&dtype)),
    };

    let shape = attr.shape();
    if !shape.is_empty() {
        return format!("array [{}: {}]", fmt_dtype(&dtype), fmt_shape(&shape));
    }

    match desc {
        TypeDescriptor::Integer(_) => attr.read_scalar::<i64>().map(|v| v.to_string()).unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Unsigned(_) => attr.read_scalar::<u64>().map(|v| v.to_string()).unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Float(_) => attr.read_scalar::<f64>().map(|v| format!("{:.5}", v)).unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::Boolean => attr.read_scalar::<bool>().map(|v| v.to_string()).unwrap_or_else(|_| "unreadable".to_string()),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) | TypeDescriptor::VarLenAscii => {
            attr.read_scalar::<VarLenUnicode>().map(|v| {
                let s = v.as_str().to_string();
                if s.len() > 50 {
                    let head = utf8_prefix_by_bytes(&s, 20);
                    let tail = utf8_suffix_by_bytes(&s, 20);
                    format!("{}...{}", head, tail)
                } else {
                    format!("'{}'", s)
                }
            }).unwrap_or_else(|_| "unreadable".to_string())
        },
        _ => format!("[{}]", fmt_dtype(&dtype)),
    }
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
    use hdf5::types::{EnumType, CompoundType, CompoundField, VarLenUnicode};
    use std::str::FromStr;
    use std::time::{SystemTime, UNIX_EPOCH};

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
                CompoundField { name: "x".to_string(), ty: TypeDescriptor::Float(FloatSize::U4), offset: 0, index: 0 },
                CompoundField { name: "y".to_string(), ty: TypeDescriptor::Float(FloatSize::U4), offset: 4, index: 1 },
            ],
            size: 8,
        });
        assert_eq!(fmt_descriptor_short_nodefs(&ct), "(x: float32, y: float32)");
    }

    #[test]
    fn test_enum() {
        let et = TypeDescriptor::Enum(EnumType {
            members: vec![
                hdf5::types::EnumMember { name: "apple".to_string(), value: 1 },
                hdf5::types::EnumMember { name: "banana".to_string(), value: 2 },
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
        use hdf5::File;
        use std::path::PathBuf;
        
        // Use the sample file we generated
        let path = PathBuf::from("../sample.h5");
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
    }

    #[test]
    fn test_format_attribute_value_utf8_truncation_safe() {
        use hdf5::File;

        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        path.push(format!("h5peek_attr_utf8_{}_{}.h5", std::process::id(), nanos));

        let file = File::create(&path).unwrap();
        let attr = file.new_attr::<VarLenUnicode>().shape(()).create("utf8_attr").unwrap();

        let unit = "\u{00E9}";
        let long = unit.repeat(60);
        let v = VarLenUnicode::from_str(&long).unwrap();
        attr.as_writer().write_scalar(&v).unwrap();

        let formatted = format_attribute_value(&attr);
        let expected = format!("{}...{}", unit.repeat(10), unit.repeat(10));
        assert_eq!(formatted, expected);

        drop(file);
        let _ = std::fs::remove_file(path);
    }
}
