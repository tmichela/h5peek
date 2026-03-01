use hdf5::{Dataset, H5Type, Selection};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::{TypeDescriptor, VarLenUnicode};
use crate::utils;
use crate::slicing;
use anyhow::Result;
use ndarray::{IxDyn, ArrayD};

pub fn print_dataset_info(ds: &Dataset, slice_expr: Option<&str>) -> Result<()> {
    println!("      dtype: {}", utils::fmt_dtype(&ds.dtype()?));
    println!("      shape: {}", utils::fmt_shape(&ds.shape()));
    
    if let Ok(space) = ds.space() {
        let maxshape = space.maxdims();
        let current_shape = ds.shape();
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
        match slicing::parse_slice(expr, &ds.shape()) {
            Ok(selection) => {
                if let Err(e) = print_selection_data(ds, selection) {
                    println!("Error reading sliced data: {}", e);
                }
            }
            Err(e) => println!("Error parsing slice: {}", e),
        }
    } else if ds.ndim() == 0 {
            print_scalar(ds)?;
    } else {
            println!("\nsample data:");
            print_sample_data(ds)?;
    }

    Ok(())
}

fn print_selection_data(ds: &Dataset, selection: Selection) -> Result<()> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor()?;

    match desc {
        TypeDescriptor::Integer(_) => print_selection::<i64>(ds, selection),
        TypeDescriptor::Unsigned(_) => print_selection::<u64>(ds, selection),
        TypeDescriptor::Float(_) => print_selection::<f64>(ds, selection),
        TypeDescriptor::Boolean => print_selection::<bool>(ds, selection),
        TypeDescriptor::VarLenUnicode | TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) | TypeDescriptor::VarLenAscii => print_selection_string(ds, selection),
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
    println!("{:?}", arr);
    Ok(())
}

fn print_selection_string(ds: &Dataset, selection: Selection) -> Result<()> {
    let arr: ArrayD<VarLenUnicode> = ds.read_slice::<VarLenUnicode, _, IxDyn>(selection)?;
    let s_arr = arr.map(|v: &VarLenUnicode| v.as_str().to_string());
    println!("{:?}", s_arr);
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
    let shape = ds.shape();
    if shape.is_empty() { return Ok(()); }

    // If dataset is very large, take a slice
    let total_size = shape.iter().product::<usize>();
    if total_size > 100 {
        // Construct a sample slice string like "0:10, 0:10, ..."
        let mut sample_parts = Vec::new();
        for dim_len in &shape {
            sample_parts.push(format!("0:{}", dim_len.min(&10)));
        }
        let sample_expr = sample_parts.join(",");
        if let Ok(selection) = slicing::parse_slice(&sample_expr, &shape) {
            return print_selection_data(ds, selection);
        }
    }

    print_selection_data(ds, Selection::new(..))
}
