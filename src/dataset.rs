use hdf5::{Dataset, H5Type};
use hdf5::plist::dataset_create::Layout;
use crate::utils;
use anyhow::Result;
use hdf5::types::VarLenUnicode;

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
            println!("\ndata:");
            if let Ok(scalar) = ds.read_scalar::<VarLenUnicode>() {
                 println!("{}", scalar.as_str());
            } else if let Ok(scalar) = ds.read_scalar::<i64>() {
                 println!("{}", scalar);
            } else if let Ok(scalar) = ds.read_scalar::<f64>() {
                 println!("{}", scalar);
            } else {
                 println!("(data type not supported for display)");
            }
        } else if ds.ndim() == 1 {
             println!("\nsample data:");
             let len = ds.shape()[0];
             let take = len.min(10);
             
             if print_1d::<i64>(ds, take).is_ok() { return Ok(()); }
             if print_1d::<f64>(ds, take).is_ok() { return Ok(()); }
             if print_1d_str(ds, take).is_ok() { return Ok(()); }

             println!("(Could not read data for display)");
        } else {
            println!("\nsample data:");
            println!("(Multi-dimensional sample printing pending)");
        }
    Ok(())
}

fn print_1d<T>(ds: &Dataset, take: usize) -> Result<()> 
where T: H5Type + std::fmt::Debug
{
    let arr = ds.read_slice_1d::<T, _>(0..take)?;
    println!("{:?}", arr);
    Ok(())
}

fn print_1d_str(ds: &Dataset, take: usize) -> Result<()> {
    let arr = ds.read_slice_1d::<VarLenUnicode, _>(0..take)?;
    let s_arr = arr.map(|v| v.as_str().to_string());
    println!("{:?}", s_arr);
    Ok(())
}