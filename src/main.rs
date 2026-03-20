use clap::{Parser, ArgAction, ValueEnum};
use std::path::PathBuf;
use anyhow::{Result, Context};
use std::io::IsTerminal;
use std::process::exit;


mod tree;
mod dataset;
mod utils;
mod completer;
mod slicing;

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

    /// Use a pager to display output if it is too long
    #[arg(long, default_value_t = true)]
    pager: bool,

    /// Disable pager
    #[arg(long, action = ArgAction::SetTrue)]
    no_pager: bool,

    /// Show group children only up to a certain depth, all by default.
    #[arg(short, long)]
    depth: Option<usize>,

    /// Select part of a dataset to examine, using Python slicing syntax
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

    /// Disable thousands separators in numeric output
    #[arg(long, action = ArgAction::SetTrue)]
    no_grouping: bool,
    /// Disable truncation for long attribute strings
    #[arg(long, action = ArgAction::SetTrue)]
    no_attr_truncate: bool,

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

fn main() -> Result<()> {
    let args = Args::parse();
    
    if !args.file.exists() {
        eprintln!("File not found: {:?}", args.file);
        exit(2);
    }
    if !args.file.is_file() {
        eprintln!("Not a file: {:?}", args.file);
        exit(2);
    }

    let file = hdf5::File::open(&args.file).context("Failed to open HDF5 file")?;

    let path_str = match args.path.as_deref() {
        Some("-") => prompt_for_path(&args.file)?,
        Some(p) => p.to_string(),
        None => "/".to_string(),
    };

    let no_color_env = std::env::var_os("NO_COLOR").is_some();
    if args.pager && !args.no_pager && std::io::stdout().is_terminal() {
        if std::env::var("LESSCHARSET").is_err() {
            std::env::set_var("LESSCHARSET", "utf-8");
        }
        // Ensure less handles colors by default if PAGER is not set
        if std::env::var("PAGER").is_err() {
            std::env::set_var("PAGER", "less -R");
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

    let array_format = utils::NumFormat {
        precision: args.precision,
        scientific: args.scientific,
        group_thousands: !args.no_grouping,
    };
    let scalar_format = utils::NumFormat::scalar(!args.no_grouping);

    let truncate_attr_strings = !args.no_attr_truncate;

    let filter = if args.filter.is_empty() {
        None
    } else {
        Some(tree::PathFilter::new(&args.filter)?)
    };

    if let Ok(group) = file.group(&path_str) {
        if args.slice.is_some() {
             eprintln!("Slicing is only allowed for datasets");
             exit(1);
        }
        
        let root_name = if path_str == "/" {
            format!("{}", args.file.display())
        } else {
            format!("{}/{}", args.file.display(), path_str.trim_start_matches('/'))
        };
        
        let printed = tree::print_group_tree(&group, &root_name, args.attrs, args.depth, !args.unsorted, filter.as_ref(), &scalar_format, truncate_attr_strings)?;
        if !printed && filter.is_some() {
            eprintln!("No paths matched the filter");
        }
        
    } else if let Ok(ds) = file.dataset(&path_str) {
        if let Some(filter) = filter.as_ref() {
            if !filter.is_match(&ds.name()) {
                eprintln!("No paths matched the filter");
                return Ok(());
            }
        }
        let root_name = format!("{}/{}", args.file.display(), path_str.trim_start_matches('/'));
        println!("{}", root_name);
        dataset::print_dataset_info(&ds, args.slice.as_deref(), &array_format, &scalar_format, truncate_attr_strings)?;
    } else {
        if file.link_exists(&path_str) {
            eprintln!("Object exists but is not a group or dataset: {}", path_str);
        } else {
            eprintln!("Object not found: {}", path_str);
        }
        eprintln!("Tip: use 'h5peek {} -' for interactive mode", args.file.display());
        exit(1);
    }

    Ok(())
}

fn prompt_for_path(file_path: &PathBuf) -> Result<String> {
    use rustyline::error::ReadlineError;
    use rustyline::Editor;
    use rustyline::config::Config;
    use crate::completer::H5Completer;
    
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
                if line.is_empty() { continue; }
                
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
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                exit(0);
            },
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                exit(0);
            },
            Err(err) => {
                println!("Error: {:?}", err);
                exit(1);
            }
        }
    }
}
