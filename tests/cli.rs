use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use predicates::str::{contains, is_match};

mod common;
use common::{create_external_link_fixture, create_fixed_ascii_array, create_varlen_ascii_array, create_varlen_ascii_scalar, sample_file_path, temp_h5_path};

fn base_cmd() -> Command {
    let mut cmd = cargo_bin_cmd!("h5peek");
    cmd.env("NO_COLOR", "1");
    cmd.env("HDF5_USE_FILE_LOCKING", "FALSE");
    cmd
}

fn base_cmd_allow_color() -> Command {
    let mut cmd = cargo_bin_cmd!("h5peek");
    cmd.env("HDF5_USE_FILE_LOCKING", "FALSE");
    cmd
}

fn with_hdf5_lock(f: impl FnOnce()) {
    let _guard = common::hdf5_lock();
    f();
}

#[test]
fn tree_view_shows_group_and_dataset() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .assert()
            .success()
            .stdout(contains("arrays_1d"))
            .stdout(contains("int32"))
            .stdout(contains("[int32: 10]"));
    });
}

#[test]
fn dataset_info_includes_dtype_shape_and_sample() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/arrays_1d/int32")
            .assert()
            .success()
            .stdout(contains("dtype: int32"))
            .stdout(contains("shape: 10"))
            .stdout(contains("sample data:"))
            .stdout(contains("[0, 1, 2, ..., 7, 8, 9]"));
    });
}

#[test]
fn slice_full_array_shows_all_elements_when_small() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/arrays_1d/int32")
            .arg("--slice")
            .arg(":")
            .assert()
            .success()
            .stdout(contains("selected data [:]:"))
            .stdout(contains("[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"));
    });
}

#[test]
fn slice_on_group_errors() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/arrays_1d")
            .arg("--slice")
            .arg("0:1")
            .assert()
            .failure()
            .stderr(contains("Slicing is only allowed for datasets"));
    });
}

#[test]
fn filter_no_match_prints_message() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("--filter")
            .arg("no_such_path")
            .assert()
            .success()
            .stderr(contains("No paths matched the filter"));
    });
}

#[test]
fn precision_applies_to_datasets_only() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/arrays_1d/float32")
            .arg("--precision")
            .arg("2")
            .assert()
            .success()
            .stdout(contains("sample data:"))
            .stdout(is_match("(?s)sample data:.*\\d+\\.\\d{2}").unwrap());

        base_cmd()
            .arg(&path)
            .arg("/scalars/float64")
            .arg("--precision")
            .arg("2")
            .assert()
            .success()
            .stdout(contains("3.141592653589793"));
    });
}

#[test]
fn compound_dataset_display_is_graceful() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/compound/particles")
            .assert()
            .success()
            .stdout(contains("sample data:"))
            .stdout(contains("compound data"));
    });
}

#[test]
fn fixed_length_string_scalar_is_displayed() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/scalars/string_utf8")
            .assert()
            .success()
            .stdout(contains("data:"))
            .stdout(contains("Hello"));
    });
}

#[test]
fn fixed_length_string_array_is_displayed() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("fixed_ascii_array");
        create_fixed_ascii_array(&path);

        base_cmd()
            .arg(&path)
            .arg("/strings")
            .assert()
            .success()
            .stdout(contains("sample data:"))
            .stdout(contains("alpha"))
            .stdout(contains("beta"));

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn varlen_ascii_string_array_is_displayed() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("varlen_ascii_array");
        create_varlen_ascii_array(&path);

        base_cmd()
            .arg(&path)
            .arg("/strings")
            .assert()
            .success()
            .stdout(contains("sample data:"))
            .stdout(contains("alpha"))
            .stdout(contains("beta"));

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn varlen_ascii_string_scalar_is_displayed() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("varlen_ascii_scalar");
        create_varlen_ascii_scalar(&path);

        base_cmd()
            .arg(&path)
            .arg("/string_scalar")
            .assert()
            .success()
            .stdout(contains("data:"))
            .stdout(contains("alpha"));

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn default_tree_output_is_sorted() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("sorted_tree");
        {
            let file = hdf5::File::create(&path).unwrap();
            let group = file.create_group("root").unwrap();
            group.new_dataset_builder().with_data(&[1_i32]).create("zz_b").unwrap();
            group.new_dataset_builder().with_data(&[2_i32]).create("aa_a").unwrap();
            file.flush().unwrap();
        }

        base_cmd()
            .arg(&path)
            .arg("/root")
            .assert()
            .success()
            .stdout(is_match("(?s)aa_a.*zz_b").unwrap());

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn default_tree_output_uses_natural_sort() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("natural_sort");
        {
            let file = hdf5::File::create(&path).unwrap();
            let group = file.create_group("root").unwrap();
            group.new_dataset_builder().with_data(&[1_i32]).create("group_2asdf").unwrap();
            group.new_dataset_builder().with_data(&[2_i32]).create("group_10asdf").unwrap();
            group.new_dataset_builder().with_data(&[3_i32]).create("group_1asdf").unwrap();
            file.flush().unwrap();
        }

        base_cmd()
            .arg(&path)
            .arg("/root")
            .assert()
            .success()
            .stdout(is_match("(?s)group_1asdf.*group_2asdf.*group_10asdf").unwrap());

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn external_link_formatting_avoids_double_slash() {
    with_hdf5_lock(|| {
        let (source, target, expected) = create_external_link_fixture();
        let double = format!("{}//g/d", target.display());

        base_cmd()
            .arg(&source)
            .arg("/links")
            .assert()
            .success()
            .stdout(contains(expected))
            .stdout(contains(double).not());

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(target);
    });
}

#[test]
fn non_group_or_dataset_object_reports_clear_error() {
    use hdf5_sys::h5t::{H5Tcopy, H5Tclose, H5Tset_size, H5T_STD_I32LE};
    use hdf5_sys::h5t::H5Tcommit2;
    use hdf5_sys::h5p::H5P_DEFAULT;
    use std::ffi::CString;

    with_hdf5_lock(|| {
        let path = temp_h5_path("named_dtype");
        let file = hdf5::File::create(&path).unwrap();

        unsafe {
            let tid = H5Tcopy(*H5T_STD_I32LE);
            assert!(tid >= 0);
            assert!(H5Tset_size(tid, 4) >= 0);

            let name = CString::new("named_dtype").unwrap();
            let status = H5Tcommit2(file.id(), name.as_ptr(), tid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            assert!(status >= 0);

            H5Tclose(tid);
        }
        file.flush().unwrap();
        drop(file);

        base_cmd()
            .arg(&path)
            .arg("/named_dtype")
            .assert()
            .failure()
            .stderr(contains("Object exists but is not a group or dataset: /named_dtype"));

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn custom_int_size_is_handled_gracefully() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/custom_types/int48")
            .assert()
            .success()
            .stdout(contains("data display not supported for integer size 6 bytes"));
    });
}

#[test]
fn invalid_slice_is_an_error() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/arrays_1d/int32")
            .arg("--slice")
            .arg("not-a-slice")
            .assert()
            .failure()
            .stderr(contains("Error parsing slice"));
    });
}

#[test]
fn color_always_emits_ansi() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd_allow_color()
            .arg(&path)
            .arg("--color")
            .arg("always")
            .assert()
            .success()
            .stdout(contains("\u{1b}["));
    });
}

#[test]
fn color_never_suppresses_ansi() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd_allow_color()
            .arg(&path)
            .arg("--color")
            .arg("never")
            .assert()
            .success()
            .stdout(contains("\u{1b}[").not());
    });
}

#[test]
fn color_auto_respects_no_color_env() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("--color")
            .arg("auto")
            .assert()
            .success()
            .stdout(contains("\u{1b}[").not());
    });
}

#[test]
fn tree_output_has_no_tabs() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("--depth")
            .arg("2")
            .assert()
            .success()
            .stdout(contains("\t").not());
    });
}
