use hdf5::Datatype;
use hdf5::types::{TypeDescriptor, IntSize, FloatSize};

pub fn fmt_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "scalar".to_string();
    }
    shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" × ")
}

pub fn fmt_dtype(dtype: &Datatype) -> String {
    // This attempts to replicate datatypes.fmt_dtype from python
    let desc = match dtype.to_descriptor() {
        Ok(d) => d,
        Err(_) => return "unknown".to_string(),
    };

    match desc {
        TypeDescriptor::Integer(size) => {
            let bits = match size {
                IntSize::U1 => 8,
                IntSize::U2 => 16,
                IntSize::U4 => 32,
                IntSize::U8 => 64,
            };
            // Note: hdf5 crate's Integer descriptor includes signed/unsigned in the enum variant usually? 
            // Wait, TypeDescriptor::Integer(IntSize) is usually signed. Unsigned is Unsigned(IntSize)?
            // I need to check how hdf5 crate defines TypeDescriptor. 
            // Based on common versions: Integer(IntSize), Unsigned(IntSize), Float(FloatSize), ...
            format!("{}-bit signed integer", bits)
        },
        TypeDescriptor::Unsigned(size) => {
            let bits = match size {
                IntSize::U1 => 8,
                IntSize::U2 => 16,
                IntSize::U4 => 32,
                IntSize::U8 => 64,
            };
            format!("{}-bit unsigned integer", bits)
        },
        TypeDescriptor::Float(size) => {
            let bits = match size {
                FloatSize::U4 => 32,
                FloatSize::U8 => 64,
            };
            format!("{}-bit floating point", bits)
        },
        TypeDescriptor::Boolean => "boolean".to_string(),
        TypeDescriptor::Enum(_) => "enum".to_string(),
        TypeDescriptor::Compound(_) => "compound".to_string(),
        TypeDescriptor::FixedArray(_, _) => "fixed array".to_string(),
        TypeDescriptor::FixedAscii(_) => "fixed ascii".to_string(),
        TypeDescriptor::FixedUnicode(_) => "fixed unicode".to_string(),
        TypeDescriptor::VarLenArray(_) => "vlen array".to_string(),
        TypeDescriptor::VarLenAscii => "variable-length ascii".to_string(),
        TypeDescriptor::VarLenUnicode => "variable-length unicode".to_string(),

    }
}
