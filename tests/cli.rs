use assert_cmd::cargo::cargo_bin_cmd;
use assert_cmd::Command;
use predicates::prelude::*;
use predicates::str::{contains, is_match};
use serde_json::Value;

mod common;
use common::{
    create_external_link_fixture, create_fixed_ascii_array, create_varlen_ascii_array,
    create_varlen_ascii_scalar, sample_file_path, temp_h5_path,
};

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

fn tree_contains_path(node: &Value, target: &str) -> bool {
    if node.get("path").and_then(Value::as_str) == Some(target) {
        return true;
    }
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for child in children {
            if tree_contains_path(child, target) {
                return true;
            }
        }
    }
    false
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
            .stdout(contains("1.5"))
            .stdout(contains("2.5"));
    });
}

#[test]
fn enum_dataset_displays_names() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/enums/colors")
            .assert()
            .success()
            .stdout(contains("sample data:"))
            .stdout(is_match("(?s)sample data:.*\\[RED, GREEN, BLUE, GREEN, RED\\]").unwrap());
    });
}

#[test]
fn enum_scalar_displays_name() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        base_cmd()
            .arg(&path)
            .arg("/enums/color_scalar")
            .assert()
            .success()
            .stdout(contains("data:"))
            .stdout(contains("GREEN"));
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
            group
                .new_dataset_builder()
                .with_data(&[1_i32])
                .create("zz_b")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[2_i32])
                .create("aa_a")
                .unwrap();
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
            group
                .new_dataset_builder()
                .with_data(&[1_i32])
                .create("group_2asdf")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[2_i32])
                .create("group_10asdf")
                .unwrap();
            group
                .new_dataset_builder()
                .with_data(&[3_i32])
                .create("group_1asdf")
                .unwrap();
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
    use hdf5_sys::h5p::H5P_DEFAULT;
    use hdf5_sys::h5t::H5Tcommit2;
    use hdf5_sys::h5t::{H5Tclose, H5Tcopy, H5Tset_size, H5T_STD_I32LE};
    use std::ffi::CString;

    with_hdf5_lock(|| {
        let path = temp_h5_path("named_dtype");
        let file = hdf5::File::create(&path).unwrap();

        unsafe {
            let tid = H5Tcopy(*H5T_STD_I32LE);
            assert!(tid >= 0);
            assert!(H5Tset_size(tid, 4) >= 0);

            let name = CString::new("named_dtype").unwrap();
            let status = H5Tcommit2(
                file.id(),
                name.as_ptr(),
                tid,
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT,
            );
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
            .stderr(contains(
                "Object exists but is not a group or dataset: /named_dtype",
            ));

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
            .stdout(contains(
                "data display not supported for integer size 6 bytes",
            ));
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

#[test]
fn json_group_output_includes_tree() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        let assert = base_cmd().arg(&path).arg("--json").assert().success();
        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        assert_eq!(value["kind"], "group");
        assert!(value["tree"].is_object());
    });
}

#[test]
fn json_dataset_output_is_metadata_only() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        let assert = base_cmd()
            .arg(&path)
            .arg("/arrays_1d/int32")
            .arg("--json")
            .assert()
            .success();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        assert_eq!(value["kind"], "dataset");
        assert_eq!(value["dataset"]["data_included"].as_bool(), Some(false));
        assert!(value["dataset"].get("data").is_none());
    });
}

#[test]
fn json_pretty_output_is_valid() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        let assert = base_cmd()
            .arg(&path)
            .arg("--json-pretty")
            .assert()
            .success();

        let stdout = String::from_utf8_lossy(&assert.get_output().stdout);
        assert!(stdout.contains('\n'));
        let value: Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(value["kind"], "group");
    });
}

#[test]
fn json_error_output_on_missing_object() {
    with_hdf5_lock(|| {
        let path = sample_file_path();

        let assert = base_cmd()
            .arg(&path)
            .arg("/no_such_object")
            .arg("--json")
            .assert()
            .failure();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        assert!(value.get("error").is_some());
        assert_eq!(value["code"].as_i64(), Some(1));
    });
}

#[test]
fn json_includes_attributes_when_requested() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("json_attrs");

        {
            let file = hdf5::File::create(&path).unwrap();
            let ds = file
                .new_dataset_builder()
                .with_data(&[1_i32])
                .create("d")
                .unwrap();
            let attr = ds.new_attr::<i32>().shape(()).create("answer").unwrap();
            attr.as_writer().write_scalar(&42).unwrap();
            file.flush().unwrap();
            drop(file);
        }

        let assert = base_cmd()
            .arg(&path)
            .arg("/d")
            .arg("--json")
            .arg("--attrs")
            .assert()
            .success();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        let attrs = value["dataset"]["attributes"].as_array().unwrap();
        assert_eq!(attrs.len(), 1);

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn json_tree_uses_natural_sort() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("json_natural_sort");
        {
            let file = hdf5::File::create(&path).unwrap();
            let group = file.create_group("root").unwrap();
            group.create_group("group_2asdf").unwrap();
            group.create_group("group_10asdf").unwrap();
            group.create_group("group_1asdf").unwrap();
            file.flush().unwrap();
        }

        let assert = base_cmd()
            .arg(&path)
            .arg("/root")
            .arg("--json")
            .assert()
            .success();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        let children = value["tree"]["children"].as_array().unwrap();
        let names: Vec<&str> = children
            .iter()
            .map(|child| child["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["group_1asdf", "group_2asdf", "group_10asdf"]);

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn json_filter_includes_descendant_paths() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("json_filter_descendant");
        {
            let file = hdf5::File::create(&path).unwrap();
            let root = file.create_group("root").unwrap();
            let level1 = root.create_group("level1").unwrap();
            level1
                .new_dataset_builder()
                .with_data(&[1_i32])
                .create("target")
                .unwrap();
            file.flush().unwrap();
        }

        let assert = base_cmd()
            .arg(&path)
            .arg("/root")
            .arg("--json")
            .arg("--filter")
            .arg("target")
            .assert()
            .success();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        let tree = &value["tree"];
        assert!(tree_contains_path(tree, "/root"));
        assert!(tree_contains_path(tree, "/root/level1/target"));

        let _ = std::fs::remove_file(path);
    });
}

#[test]
fn json_hard_link_outputs_deduped_entries() {
    with_hdf5_lock(|| {
        let path = temp_h5_path("json_hard_link");
        {
            let file = hdf5::File::create(&path).unwrap();
            let group = file.create_group("group").unwrap();
            group
                .new_dataset_builder()
                .with_data(&[1_i32])
                .create("var")
                .unwrap();
            group.link_hard("var", "hard1").unwrap();
            group.link_hard("var", "hard2").unwrap();
            file.flush().unwrap();
        }

        let assert = base_cmd()
            .arg(&path)
            .arg("/group")
            .arg("--json")
            .assert()
            .success();

        let value: Value = serde_json::from_slice(&assert.get_output().stdout).unwrap();
        let children = value["tree"]["children"].as_array().unwrap();
        let mut dataset_paths = Vec::new();
        let mut hard_links = Vec::new();
        for child in children {
            match child["kind"].as_str().unwrap() {
                "dataset" => dataset_paths.push(child["path"].as_str().unwrap().to_string()),
                "hard_link" => {
                    hard_links.push(child["hard_link_to"].as_str().unwrap().to_string())
                }
                _ => {}
            }
        }

        assert_eq!(dataset_paths.len(), 1);
        assert!(!hard_links.is_empty());
        for link in hard_links {
            assert_eq!(link, dataset_paths[0]);
        }

        let _ = std::fs::remove_file(path);
    });
}
