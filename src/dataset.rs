use hdf5::{Dataset, H5Type};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{TypeDescriptor, VarLenUnicode};
use crate::utils;
use anyhow::Result;

pub fn print_dataset_info(ds: &Dataset, slice_expr: Option<&str>) -> Result<()> {
    println!("      dtype: {}", utils::fmt_dtype(&ds.dtype()?));
    println!("      shape: {}", utils::fmt_shape(&ds.shape()));

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
        println!("(Slicing implementation pending)");
    } else if ds.ndim() == 0 {
            print_scalar(ds)?;
    } else {
            println!("\nsample data:");
            print_sample_data(ds)?;
    }
    Ok(())
}

fn print_scalar(ds: &Dataset) -> Result<()> {
    println!("\ndata:");
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;
    
    match desc {
        TypeDescriptor::Integer(_) => {
            if let Ok(v) = ds.read_scalar::<i64>() { println!("{}", v); }
        },
        TypeDescriptor::Unsigned(_) => {
            if let Ok(v) = ds.read_scalar::<u64>() { println!("{}", v); }
        },
        TypeDescriptor::Float(_) => {
             if let Ok(v) = ds.read_scalar::<f64>() { println!("{}", v); }
        },
        TypeDescriptor::Boolean => {
             if let Ok(v) = ds.read_scalar::<bool>() { println!("{}", v); }
        },
        TypeDescriptor::VarLenUnicode | TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) | TypeDescriptor::VarLenAscii => {
             if let Ok(v) = ds.read_scalar::<VarLenUnicode>() { println!("{}", v.as_str()); }
        },
        _ => println!("(data type not supported for display)"),
    }
    Ok(())
}

fn print_sample_data(ds: &Dataset) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;
    
    let shape = ds.shape();
    let total_size = shape.iter().product::<usize>();
    if total_size == 0 {
        println!("[]");
        return Ok(());
    }
    
    // Simplification: just read 1D slice if 1D.
    if ds.ndim() == 1 {
        let len = shape[0];
        let take = len.min(10);
        let range = 0..take;
        
        match desc {
             TypeDescriptor::Integer(_) => print_1d_slice::<i64>(ds, range),
             TypeDescriptor::Unsigned(_) => print_1d_slice::<u64>(ds, range),
             TypeDescriptor::Float(_) => print_1d_slice::<f64>(ds, range),
             TypeDescriptor::Boolean => print_1d_slice::<bool>(ds, range),
             TypeDescriptor::VarLenUnicode | TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) | TypeDescriptor::VarLenAscii => print_1d_slice_string(ds, range),
             _ => {
                 println!("(data type not supported for display)");
                 Ok(())
             }
        }
    } else {
        println!("(Multi-dimensional sample printing pending)");
        Ok(())
    }
}

fn print_1d_slice<T>(ds: &Dataset, range: std::ops::Range<usize>) -> Result<()> 
where T: H5Type + std::fmt::Debug
{
    let arr = ds.read_slice_1d::<T, _>(range)?;
    println!("{:?}", arr);
    Ok(())
}

fn print_1d_slice_string(ds: &Dataset, range: std::ops::Range<usize>) -> Result<()> {
    let arr = ds.read_slice_1d::<VarLenUnicode, _>(range)?;
    let s_arr = arr.map(|v| v.as_str().to_string());
    println!("{:?}", s_arr);
    Ok(())
}
