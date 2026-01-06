use hdf5::Datatype;
use hdf5::types::{TypeDescriptor, IntSize, FloatSize};

pub fn fmt_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" × ")
}

pub fn fmt_dtype(dtype: &Datatype) -> String {
    match dtype.to_descriptor() {
        Ok(desc) => fmt_descriptor_short(&desc),
        Err(_) => "unknown".to_string(),
    }
}

pub fn dtype_description(dtype: &Datatype) -> Option<String> {
    match dtype.to_descriptor() {
        Ok(desc) => dtype_description_from_desc(&desc),
        Err(_) => None,
    }
}

fn dtype_description_from_desc(desc: &TypeDescriptor) -> Option<String> {
    match desc {
        TypeDescriptor::Integer(size) => Some(format!("{}-bit signed integer", int_size_to_bits(*size))),
        TypeDescriptor::Unsigned(size) => Some(format!("{}-bit unsigned integer", int_size_to_bits(*size))),
        TypeDescriptor::Float(size) => Some(format!("{}-bit floating point", float_size_to_bits(*size))),
        _ => None,
    }
}

fn fmt_descriptor_short(desc: &TypeDescriptor) -> String {
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
                format!("{}: {}", f.name, fmt_descriptor_short(&f.ty))
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
             format!("{} array of {}", shape_str, fmt_descriptor_short(inner))
        },
        TypeDescriptor::FixedAscii(len) => format!("{}-byte ASCII string", len),
        TypeDescriptor::FixedUnicode(len) => format!("{}-byte UTF-8 string", len),
        TypeDescriptor::VarLenArray(ty) => format!("vlen array of {}", fmt_descriptor_short(ty)),
        TypeDescriptor::VarLenAscii => "ASCII string".to_string(),
        TypeDescriptor::VarLenUnicode => "UTF-8 string".to_string(),
    }
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
    use hdf5::types::{EnumType, CompoundType, CompoundField};

    #[test]
    fn test_standard_float() {
        let desc = TypeDescriptor::Float(FloatSize::U4);
        assert_eq!(fmt_descriptor_short(&desc), "float32");
        assert_eq!(dtype_description_from_desc(&desc), Some("32-bit floating point".to_string()));
    }

    #[test]
    fn test_standard_int() {
        let i4 = TypeDescriptor::Integer(IntSize::U4);
        assert_eq!(fmt_descriptor_short(&i4), "int32");
        assert_eq!(dtype_description_from_desc(&i4), Some("32-bit signed integer".to_string()));

        let u8 = TypeDescriptor::Unsigned(IntSize::U8);
        assert_eq!(fmt_descriptor_short(&u8), "uint64");
        assert_eq!(dtype_description_from_desc(&u8), Some("64-bit unsigned integer".to_string()));
    }

    #[test]
    fn test_string() {
        // vlen string
        let vst = TypeDescriptor::VarLenUnicode;
        assert_eq!(fmt_descriptor_short(&vst), "UTF-8 string");

        // fixed-length string
        let fst = TypeDescriptor::FixedAscii(3);
        assert_eq!(fmt_descriptor_short(&fst), "3-byte ASCII string");
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
        assert_eq!(fmt_descriptor_short(&ct), "(x: float32, y: float32)");
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
        assert_eq!(fmt_descriptor_short(&et), "enum (apple, banana)");
    }

    #[test]
    fn test_vlen() {
        let vt = TypeDescriptor::VarLenArray(Box::new(TypeDescriptor::Integer(IntSize::U2)));
        assert_eq!(fmt_descriptor_short(&vt), "vlen array of int16");
    }

    #[test]
    fn test_array() {
        // 3 x 4 array of float64
        // Nested FixedArray to simulate multidimensional array
        let t_inner = TypeDescriptor::FixedArray(Box::new(TypeDescriptor::Float(FloatSize::U8)), 4);
        let at = TypeDescriptor::FixedArray(Box::new(t_inner), 3);
        assert_eq!(fmt_descriptor_short(&at), "3 × 4 array of float64");
    }
}
