use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use predicates::str::{contains, is_match};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn sample_file_path() -> PathBuf {
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

fn temp_h5_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    path.push(format!("h5peek_{}_{}_{}.h5", name, std::process::id(), nanos));
    path
}


fn base_cmd() -> Command {
    let mut cmd = cargo_bin_cmd!("h5peek");
    cmd.env("NO_COLOR", "1");
    cmd.env("HDF5_USE_FILE_LOCKING", "FALSE");
    cmd
}

#[test]
fn tree_view_shows_group_and_dataset() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .assert()
        .success()
        .stdout(contains("arrays_1d"))
        .stdout(contains("int32"))
        .stdout(contains("[int32: 10]"));
}

#[test]
fn dataset_info_includes_dtype_shape_and_sample() {
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
}

#[test]
fn slice_on_group_errors() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("/arrays_1d")
        .arg("--slice")
        .arg("0:1")
        .assert()
        .failure()
        .stderr(contains("Slicing is only allowed for datasets"));
}

#[test]
fn filter_no_match_prints_message() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("--filter")
        .arg("no_such_path")
        .assert()
        .success()
        .stderr(contains("No paths matched the filter"));
}

#[test]
fn precision_applies_to_datasets_only() {
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
}

#[test]
fn compound_dataset_display_is_graceful() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("/compound/particles")
        .assert()
        .success()
        .stdout(contains("sample data:"))
        .stdout(contains("compound data"));
}

#[test]
fn fixed_length_string_scalar_is_displayed() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("/scalars/string_utf8")
        .assert()
        .success()
        .stdout(contains("data:"))
        .stdout(contains("Hello"));
}

#[test]
fn fixed_length_string_array_is_displayed() {
    use hdf5::types::FixedAscii;

    let path = temp_h5_path("fixed_ascii_array");
    let file = hdf5::File::create(&path).unwrap();
    let data = vec![
        FixedAscii::<8>::from_ascii(b"alpha").unwrap(),
        FixedAscii::<8>::from_ascii(b"beta").unwrap(),
        FixedAscii::<8>::from_ascii(b"gamma").unwrap(),
    ];
    file.new_dataset_builder().with_data(&data).create("strings").unwrap();
    drop(file);

    base_cmd()
        .arg(&path)
        .arg("/strings")
        .assert()
        .success()
        .stdout(contains("sample data:"))
        .stdout(contains("alpha"))
        .stdout(contains("beta"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn default_tree_output_is_sorted() {
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
}

#[test]
fn default_tree_output_uses_natural_sort() {
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
}

#[test]
fn external_link_formatting_avoids_double_slash() {
    let target = temp_h5_path("ext_target");
    let source = temp_h5_path("ext_source");

    {
        let file = hdf5::File::create(&target).unwrap();
        let group = file.create_group("g").unwrap();
        group.new_dataset_builder().with_data(&[1_i32]).create("d").unwrap();
        file.flush().unwrap();
    }

    {
        let file = hdf5::File::create(&source).unwrap();
        let group = file.create_group("links").unwrap();
        group.link_external(target.to_str().unwrap(), "/g/d", "ext_d").unwrap();
        file.flush().unwrap();
    }

    let expected = format!("{}{}", target.display(), "/g/d");
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
}

#[test]
fn custom_int_size_is_handled_gracefully() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("/custom_types/int48")
        .assert()
        .success()
        .stdout(contains("data display not supported for integer size 6 bytes"));
}

#[test]
fn invalid_slice_is_an_error() {
    let path = sample_file_path();

    base_cmd()
        .arg(&path)
        .arg("/arrays_1d/int32")
        .arg("--slice")
        .arg("not-a-slice")
        .assert()
        .failure()
        .stderr(contains("Error parsing slice"));
}
