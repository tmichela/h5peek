use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use hdf5::types::{FixedAscii, VarLenAscii};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static HDF5_LOCK: Mutex<()> = Mutex::new(());

pub fn hdf5_lock() -> MutexGuard<'static, ()> {
    HDF5_LOCK.lock().unwrap()
}

pub fn sample_file_path() -> PathBuf {
    std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("sample.h5");
    assert!(
        path.exists(),
        "Sample file not found at {}. Run `python3 script/generate_sample.py data/sample.h5`.",
        path.display()
    );
    path
}

pub fn temp_h5_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    path.push(format!("h5peek_{}_{}_{}.h5", name, std::process::id(), counter));
    path
}

pub fn create_fixed_ascii_array(path: &Path) {
    let file = hdf5::File::create(path).unwrap();
    let data = vec![
        FixedAscii::<8>::from_ascii(b"alpha").unwrap(),
        FixedAscii::<8>::from_ascii(b"beta").unwrap(),
        FixedAscii::<8>::from_ascii(b"gamma").unwrap(),
    ];
    file.new_dataset_builder()
        .with_data(&data)
        .create("strings")
        .unwrap();
    file.flush().unwrap();
    drop(file);
}

pub fn create_varlen_ascii_array(path: &Path) {
    let file = hdf5::File::create(path).unwrap();
    let data = vec![
        VarLenAscii::from_ascii(b"alpha").unwrap(),
        VarLenAscii::from_ascii(b"beta").unwrap(),
        VarLenAscii::from_ascii(b"gamma").unwrap(),
    ];
    file.new_dataset_builder()
        .with_data(&data)
        .create("strings")
        .unwrap();
    file.flush().unwrap();
    drop(file);
}

pub fn create_varlen_ascii_scalar(path: &Path) {
    let file = hdf5::File::create(path).unwrap();
    let value = VarLenAscii::from_ascii(b"alpha").unwrap();
    let ds = file
        .new_dataset::<VarLenAscii>()
        .shape(())
        .create("string_scalar")
        .unwrap();
    ds.write_scalar(&value).unwrap();
    file.flush().unwrap();
    drop(file);
}

pub fn create_external_link_fixture() -> (PathBuf, PathBuf, String) {
    let target = temp_h5_path("ext_target");
    let source = temp_h5_path("ext_source");

    {
        let file = hdf5::File::create(&target).unwrap();
        let group = file.create_group("g").unwrap();
        group
            .new_dataset_builder()
            .with_data(&[1_i32])
            .create("d")
            .unwrap();
        file.flush().unwrap();
    }

    {
        let file = hdf5::File::create(&source).unwrap();
        let group = file.create_group("links").unwrap();
        group
            .link_external(target.to_str().unwrap(), "/g/d", "ext_d")
            .unwrap();
        file.flush().unwrap();
    }

    let expected = format!("{}{}", target.display(), "/g/d");
    (source, target, expected)
}

#[allow(dead_code)]
pub fn create_packed_compound_fixture(path: &Path) {
    use hdf5_sys::h5d::{H5Dcreate2, H5Dclose, H5Dwrite};
    use hdf5_sys::h5p::H5P_DEFAULT;
    use hdf5_sys::h5s::{H5Screate_simple, H5Sclose};
    use hdf5_sys::h5t::{H5Tclose, H5Tcreate, H5Tinsert, H5T_COMPOUND, H5T_NATIVE_DOUBLE, H5T_NATIVE_INT};
    use std::ffi::{c_void, CString};

    let file = hdf5::File::create(path).unwrap();
    let group = file.create_group("compound").unwrap();

    unsafe {
        let type_id = H5Tcreate(H5T_COMPOUND, 12);
        assert!(type_id >= 0);

        let name_x = CString::new("x").unwrap();
        let name_y = CString::new("y").unwrap();
        assert!(H5Tinsert(type_id, name_x.as_ptr(), 0, *H5T_NATIVE_INT) >= 0);
        assert!(H5Tinsert(type_id, name_y.as_ptr(), 4, *H5T_NATIVE_DOUBLE) >= 0);

        let dims = [2_u64];
        let space_id = H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        assert!(space_id >= 0);

        let dset_name = CString::new("particles").unwrap();
        let dset_id = H5Dcreate2(
            group.id(),
            dset_name.as_ptr(),
            type_id,
            space_id,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
        );
        assert!(dset_id >= 0);

        let mut buf = Vec::with_capacity(24);
        let mut record = [0_u8; 12];

        record[0..4].copy_from_slice(&1_i32.to_ne_bytes());
        record[4..12].copy_from_slice(&1.5_f64.to_ne_bytes());
        buf.extend_from_slice(&record);

        record[0..4].copy_from_slice(&2_i32.to_ne_bytes());
        record[4..12].copy_from_slice(&2.5_f64.to_ne_bytes());
        buf.extend_from_slice(&record);

        let status = H5Dwrite(
            dset_id,
            type_id,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT,
            buf.as_ptr() as *const c_void,
        );
        assert!(status >= 0);

        H5Dclose(dset_id);
        H5Sclose(space_id);
        H5Tclose(type_id);
    }

    file.flush().unwrap();
    drop(file);
}
