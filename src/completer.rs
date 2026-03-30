use hdf5::File;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::Context;
use rustyline::Helper;
use std::path::PathBuf;

pub struct H5Completer {
    file_path: PathBuf,
}

impl H5Completer {
    pub fn new(file_path: PathBuf) -> Self {
        Self { file_path }
    }
}

impl Completer for H5Completer {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> Result<(usize, Vec<Pair>), ReadlineError> {
        // Find the start of the current path component
        let (start, path_prefix) = if let Some(last_slash) = line[..pos].rfind('/') {
            (last_slash + 1, &line[last_slash + 1..pos])
        } else {
            (0, &line[..pos])
        };

        // Determine the parent group path to list members from
        let parent_path = if start > 0 { &line[..start] } else { "/" };

        // Open the file (we do this every time to ensure up-to-date info and avoid keeping a handle if not needed,
        // though keeping a handle would be more efficient)
        let file = match File::open(&self.file_path) {
            Ok(f) => f,
            Err(_) => return Ok((start, vec![])),
        };

        // Open the group
        let group = if parent_path == "/" {
            file.group("/")
        } else {
            // HDF5 paths shouldn't really end in /, but for the root it does.
            // If parent_path is "group/", trim it to "group"
            let trimmed = parent_path.trim_end_matches('/');
            if trimmed.is_empty() {
                file.group("/")
            } else {
                file.group(trimmed)
            }
        };

        let mut matches = Vec::new();

        if let Ok(g) = group {
            if let Ok(members) = g.member_names() {
                for member in members {
                    if member.starts_with(path_prefix) {
                        // Check if it's a group to append '/'
                        // We try to open it as a group. This resolves links too.
                        let is_group = g.group(&member).is_ok();

                        let display = member.clone();
                        let mut replacement = member.clone();
                        if is_group {
                            replacement.push('/');
                        }

                        matches.push(Pair {
                            display,
                            replacement,
                        });
                    }
                }
            }
        }

        Ok((start, matches))
    }
}

impl Hinter for H5Completer {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for H5Completer {}
impl Validator for H5Completer {}
impl Helper for H5Completer {}
