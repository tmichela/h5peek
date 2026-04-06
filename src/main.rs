use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::exit;

mod array_format;
mod completer;
mod dataset;
mod json_output;
mod plot;
mod slicing;
mod tree;
mod utils;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// HDF5 file to view
    #[arg(required = true)]
    file: PathBuf,

    /// Object to show within the file, or '-' to prompt for a name
    path: Option<String>,

    /// Show attributes of groups
    #[arg(long)]
    attrs: bool,

    /// Pager command to use (default: $PAGER or "less -R")
    #[arg(long, value_name = "PAGER")]
    pager: Option<String>,

    /// Disable pager
    #[arg(long, action = ArgAction::SetTrue)]
    no_pager: bool,

    /// Show group children only up to a certain depth, all by default.
    #[arg(short, long)]
    depth: Option<usize>,

    /// Select part of a dataset to examine, using Python or Rust-style slicing syntax
    #[arg(short, long)]
    slice: Option<String>,

    /// Filter displayed paths using a Unix-like glob pattern (can be used multiple times)
    #[arg(short, long, action = ArgAction::Append)]
    filter: Vec<String>,

    /// Preserve the original (unsorted) HDF5 member order in tree output
    #[arg(long, action = ArgAction::SetTrue)]
    unsorted: bool,

    /// Float precision for display
    #[arg(long, default_value_t = 5)]
    precision: usize,

    /// Use scientific notation for floats
    #[arg(long, action = ArgAction::SetTrue)]
    scientific: bool,

    /// Disable truncation for long attribute strings
    #[arg(long, action = ArgAction::SetTrue)]
    no_attr_truncate: bool,

    /// Emit JSON output instead of text
    #[arg(long, action = ArgAction::SetTrue)]
    json: bool,

    /// Pretty-print JSON output (implies --json)
    #[arg(long, action = ArgAction::SetTrue)]
    json_pretty: bool,

    /// Color output: auto, always, or never
    #[arg(long, value_enum, default_value_t = ColorMode::Auto)]
    color: ColorMode,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ColorMode {
    Auto,
    Always,
    Never,
}

fn install_broken_pipe_handler() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(err) = info.payload().downcast_ref::<std::io::Error>() {
            if err.kind() == std::io::ErrorKind::BrokenPipe {
                std::process::exit(0);
            }
        }

        let mut broken_pipe = false;
        if let Some(msg) = info.payload().downcast_ref::<String>() {
            broken_pipe = msg.contains("Broken pipe");
        } else if let Some(msg) = info.payload().downcast_ref::<&str>() {
            broken_pipe = msg.contains("Broken pipe");
        }

        if broken_pipe {
            std::process::exit(0);
        }

        default_hook(info);
    }));
}

fn resolve_pager_command(args: &Args) -> Option<String> {
    if args.no_pager {
        return None;
    }
    if let Some(cmd) = args.pager.as_deref() {
        return Some(cmd.to_string());
    }
    if let Ok(cmd) = std::env::var("PAGER") {
        return Some(cmd);
    }
    Some("less -R".to_string())
}

fn main() -> Result<()> {
    install_broken_pipe_handler();
    let args = Args::parse();
    if json_enabled(&args) {
        return run_json(args);
    }
    run(args)
}

fn run(args: Args) -> Result<()> {
    validate_file(&args.file);

    let file = hdf5::File::open(&args.file).context("Failed to open HDF5 file")?;
    let path_str = resolve_target_path(&args)?;

    configure_output(&args);

    let array_format = utils::NumFormat {
        precision: args.precision,
        scientific: args.scientific,
    };
    let scalar_format = utils::NumFormat::scalar();
    let truncate_attr_strings = !args.no_attr_truncate;
    let filter = build_filter(&args)?;

    handle_target(
        &file,
        &path_str,
        &args,
        &array_format,
        &scalar_format,
        truncate_attr_strings,
        filter.as_ref(),
    )
}

fn json_enabled(args: &Args) -> bool {
    args.json || args.json_pretty
}

fn validate_file(path: &PathBuf) {
    if !path.exists() {
        eprintln!("File not found: {:?}", path);
        exit(2);
    }
    if !path.is_file() {
        eprintln!("Not a file: {:?}", path);
        exit(2);
    }
}

fn emit_json_error(message: impl Into<String>, code: i32, pretty: bool) -> ! {
    let err = json_output::JsonErrorOutput {
        error: message.into(),
        code,
    };
    if let Err(write_err) = json_output::write_json(&err, pretty) {
        eprintln!("Failed to write JSON error: {}", write_err);
    }
    exit(code);
}

fn run_json(args: Args) -> Result<()> {
    let pretty = args.json_pretty;
    if !args.file.exists() {
        emit_json_error(format!("File not found: {:?}", args.file), 2, pretty);
    }
    if !args.file.is_file() {
        emit_json_error(format!("Not a file: {:?}", args.file), 2, pretty);
    }
    if args.path.as_deref() == Some("-") {
        emit_json_error("Interactive mode is not supported with --json", 1, pretty);
    }

    let file = match hdf5::File::open(&args.file) {
        Ok(f) => f,
        Err(e) => {
            emit_json_error(format!("Failed to open HDF5 file: {}", e), 1, pretty);
        }
    };

    let path_str = match args.path.as_deref() {
        Some(p) => p.to_string(),
        None => "/".to_string(),
    };

    let filter = match build_filter(&args) {
        Ok(filter) => filter,
        Err(e) => {
            emit_json_error(e.to_string(), 1, pretty);
        }
    };

    let scalar_format = utils::NumFormat::scalar();
    let truncate_attr_strings = !args.no_attr_truncate;

    let output = if let Ok(group) = file.group(&path_str) {
        if args.slice.is_some() {
            emit_json_error("Slicing is only allowed for datasets", 1, pretty);
        }

        let mut tree_opts = tree::TreePrintOptions::new(scalar_format);
        tree_opts.expand_attrs = args.attrs;
        tree_opts.max_depth = args.depth;
        tree_opts.sort_members = !args.unsorted;
        tree_opts.filter = filter.as_ref();
        tree_opts.truncate_attr_strings = truncate_attr_strings;

        let tree = match json_output::build_group_tree_json(&group, &tree_opts) {
            Ok(tree) => tree,
            Err(e) => emit_json_error(e.to_string(), 1, pretty),
        };

        let mut out = json_output::JsonOutput {
            kind: json_output::OutputKind::Group,
            file: format!("{}", args.file.display()),
            path: group.name(),
            matched: None,
            warnings: Vec::new(),
            tree,
            dataset: None,
        };

        if filter.is_some() {
            if out.tree.is_some() {
                out.matched = Some(true);
            } else {
                out.matched = Some(false);
                out.warnings.push("No paths matched the filter".to_string());
            }
        }
        out
    } else if let Ok(ds) = file.dataset(&path_str) {
        if let Some(filter) = filter.as_ref() {
            if !filter.is_match(&ds.name()) {
                let out = json_output::JsonOutput {
                    kind: json_output::OutputKind::Dataset,
                    file: format!("{}", args.file.display()),
                    path: ds.name(),
                    matched: Some(false),
                    warnings: vec!["No paths matched the filter".to_string()],
                    tree: None,
                    dataset: None,
                };
                json_output::write_json(&out, pretty)?;
                return Ok(());
            }
        }

        let dataset = match json_output::build_dataset_info(
            &ds,
            &scalar_format,
            truncate_attr_strings,
            args.attrs,
            args.slice.as_deref(),
        ) {
            Ok(info) => info,
            Err(e) => emit_json_error(e.to_string(), 1, pretty),
        };

        json_output::JsonOutput {
            kind: json_output::OutputKind::Dataset,
            file: format!("{}", args.file.display()),
            path: ds.name(),
            matched: filter.as_ref().map(|_| true),
            warnings: Vec::new(),
            tree: None,
            dataset: Some(dataset),
        }
    } else {
        if file.link_exists(&path_str) {
            emit_json_error(
                format!("Object exists but is not a group or dataset: {}", path_str),
                1,
                pretty,
            );
        } else {
            emit_json_error(format!("Object not found: {}", path_str), 1, pretty);
        }
    };

    json_output::write_json(&output, pretty)?;
    Ok(())
}

fn resolve_target_path(args: &Args) -> Result<String> {
    match args.path.as_deref() {
        Some("-") => prompt_for_path(&args.file),
        Some(p) => Ok(p.to_string()),
        None => Ok("/".to_string()),
    }
}

fn build_filter(args: &Args) -> Result<Option<tree::PathFilter>> {
    if args.filter.is_empty() {
        Ok(None)
    } else {
        Ok(Some(tree::PathFilter::new(&args.filter)?))
    }
}

fn configure_output(args: &Args) {
    let no_color_env = std::env::var_os("NO_COLOR").is_some();
    let pager_cmd = resolve_pager_command(args);

    if pager_cmd.is_some() && std::io::stdout().is_terminal() {
        if std::env::var("LESSCHARSET").is_err() {
            std::env::set_var("LESSCHARSET", "utf-8");
        }
        if let Some(cmd) = pager_cmd.as_deref() {
            std::env::set_var("PAGER", cmd);
        }
        pager::Pager::new().setup();
        // Force colors because pager might make is_a_tty false for the process
        let pager_color = match args.color {
            ColorMode::Always => true,
            ColorMode::Never => false,
            ColorMode::Auto => !no_color_env,
        };
        colored::control::set_override(pager_color);
    } else {
        match args.color {
            ColorMode::Always => colored::control::set_override(true),
            ColorMode::Never => colored::control::set_override(false),
            ColorMode::Auto => {
                if no_color_env {
                    colored::control::set_override(false);
                }
            }
        }
    }
}

fn handle_target(
    file: &hdf5::File,
    path_str: &str,
    args: &Args,
    array_format: &utils::NumFormat,
    scalar_format: &utils::NumFormat,
    truncate_attr_strings: bool,
    filter: Option<&tree::PathFilter>,
) -> Result<()> {
    if let Ok(group) = file.group(path_str) {
        if args.slice.is_some() {
            eprintln!("Slicing is only allowed for datasets");
            exit(1);
        }

        let root_name = if path_str == "/" {
            format!("{}", args.file.display())
        } else {
            format!(
                "{}/{}",
                args.file.display(),
                path_str.trim_start_matches('/')
            )
        };

        let mut tree_opts = tree::TreePrintOptions::new(*scalar_format);
        tree_opts.expand_attrs = args.attrs;
        tree_opts.max_depth = args.depth;
        tree_opts.sort_members = !args.unsorted;
        tree_opts.filter = filter;
        tree_opts.truncate_attr_strings = truncate_attr_strings;

        let printed = tree::print_group_tree(&group, &root_name, &tree_opts)?;
        if !printed && filter.is_some() {
            eprintln!("No paths matched the filter");
        }
    } else if let Ok(ds) = file.dataset(path_str) {
        if let Some(filter) = filter {
            if !filter.is_match(&ds.name()) {
                eprintln!("No paths matched the filter");
                return Ok(());
            }
        }
        let root_name = format!(
            "{}/{}",
            args.file.display(),
            path_str.trim_start_matches('/')
        );
        println!("{}", root_name);
        dataset::print_dataset_info(
            &ds,
            args.slice.as_deref(),
            array_format,
            scalar_format,
            truncate_attr_strings,
        )?;
    } else {
        if file.link_exists(path_str) {
            eprintln!("Object exists but is not a group or dataset: {}", path_str);
        } else {
            eprintln!("Object not found: {}", path_str);
        }
        eprintln!(
            "Tip: use 'h5peek {} -' for interactive mode",
            args.file.display()
        );
        exit(1);
    }

    Ok(())
}

fn prompt_for_path(file_path: &PathBuf) -> Result<String> {
    use crate::completer::H5Completer;
    use rustyline::config::Config;
    use rustyline::error::ReadlineError;
    use rustyline::Editor;

    let config = Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();

    let h5_completer = H5Completer::new(file_path.clone());
    let mut rl = Editor::<H5Completer, rustyline::history::DefaultHistory>::with_config(config)?;
    rl.set_helper(Some(h5_completer));

    println!("Interactive mode for {}", file_path.display());

    loop {
        let readline = rl.readline(&format!("Object path: {}/", file_path.display()));
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if let Ok(file) = hdf5::File::open(file_path) {
                    let path_to_check = if line.starts_with('/') {
                        line.to_string()
                    } else {
                        format!("/{}", line)
                    };

                    if file.link_exists(&path_to_check) {
                        return Ok(path_to_check);
                    } else {
                        println!("No object at '{}'", line);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                exit(0);
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                exit(0);
            }
            Err(err) => {
                println!("Error: {:?}", err);
                exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env_var(key: &str, value: Option<&str>, f: impl FnOnce()) {
        let _lock = ENV_LOCK.lock().unwrap();
        let prev = std::env::var_os(key);
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));

        match prev {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }

        if let Err(err) = result {
            std::panic::resume_unwind(err);
        }
    }

    #[test]
    fn resolve_pager_command_prefers_arg() {
        with_env_var("PAGER", Some("more"), || {
            let args = Args::parse_from(["h5peek", "file.h5", "--pager", "most"]);
            assert_eq!(resolve_pager_command(&args).as_deref(), Some("most"));
        });
    }

    #[test]
    fn resolve_pager_command_no_pager_wins() {
        with_env_var("PAGER", Some("more"), || {
            let args = Args::parse_from(["h5peek", "file.h5", "--pager", "most", "--no-pager"]);
            assert_eq!(resolve_pager_command(&args), None);
        });
    }

    #[test]
    fn resolve_pager_command_defaults() {
        with_env_var("PAGER", Some("more"), || {
            let args = Args::parse_from(["h5peek", "file.h5"]);
            assert_eq!(resolve_pager_command(&args).as_deref(), Some("more"));
        });
        with_env_var("PAGER", None, || {
            let args = Args::parse_from(["h5peek", "file.h5"]);
            assert_eq!(resolve_pager_command(&args).as_deref(), Some("less -R"));
        });
    }

    #[test]
    fn args_no_attr_truncate_flag() {
        let args = Args::parse_from(["h5peek", "file.h5", "--no-attr-truncate"]);
        assert!(args.no_attr_truncate);
    }
}
