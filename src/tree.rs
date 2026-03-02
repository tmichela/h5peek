use hdf5::{Group, Dataset};
use colored::Colorize;
use crate::utils;
use anyhow::{anyhow, Result};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::collections::{HashMap, HashSet};

pub struct PathFilter {
    set: GlobSet,
    substrings: Vec<String>,
}

impl PathFilter {
    pub fn new(patterns: &[String]) -> Result<Self> {
        let mut builder = GlobSetBuilder::new();
        let mut substrings = Vec::new();
        for pattern in patterns {
            let normalized = if has_glob_meta(pattern) {
                normalize_pattern(pattern)
            } else {
                substrings.push(pattern.clone());
                continue;
            };
            let glob = GlobBuilder::new(&normalized)
                .literal_separator(true)
                .backslash_escape(true)
                .build()
                .map_err(|e| anyhow!("Invalid filter pattern '{}': {}", pattern, e))?;
            builder.add(glob);
        }
        let set = builder.build().map_err(|e| anyhow!("Invalid filter patterns: {}", e))?;
        Ok(Self { set, substrings })
    }

    pub fn is_match(&self, path: &str) -> bool {
        if self.substrings.iter().any(|needle| path.contains(needle)) {
            return true;
        }
        self.set.is_match(path)
    }
}

fn normalize_pattern(pattern: &str) -> String {
    if pattern.starts_with('/') || pattern.starts_with("**/") {
        pattern.to_string()
    } else {
        format!("**/{}", pattern)
    }
}

fn has_glob_meta(pattern: &str) -> bool {
    let mut chars = pattern.chars();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                chars.next();
            }
            '*' | '?' | '[' | ']' | '{' | '}' => return true,
            _ => {}
        }
    }
    false
}

pub fn print_group_tree(group: &Group, name: &str, expand_attrs: bool, max_depth: Option<usize>, filter: Option<&PathFilter>, fmt: &utils::NumFormat) -> Result<bool> {
    let mut visited = HashMap::new();
    let mut filter_guard = HashSet::new();
    print_node_impl(Node::Group(group.clone()), name, "", true, 0, expand_attrs, max_depth, &mut visited, filter, &mut filter_guard, false, fmt)
}

enum Node {
    Group(Group),
    Dataset(Dataset),
}

fn print_node_impl(node: Node, name: &str, prefix: &str, is_last: bool, depth: usize, expand_attrs: bool, max_depth: Option<usize>, visited: &mut HashMap<u64, String>, filter: Option<&PathFilter>, filter_guard: &mut HashSet<u64>, force_show: bool, fmt: &utils::NumFormat) -> Result<bool> {
    let connector = if depth == 0 { "" } else if is_last { "└" } else { "├" };
    
    // Check for visited (Hard Link cycle detection)
    // We need the ID of the object.
    let (obj_id, full_path) = match &node {
        Node::Group(g) => (g.id(), g.name()),
        Node::Dataset(d) => (d.id(), d.name()),
    };

    let matches_filter = filter.map_or(true, |f| f.is_match(&full_path));
    let addr = utils::get_object_addr(obj_id).unwrap_or(0); // If fails (0), we just don't dedup, or risk infinite loop? 0 is likely invalid addr.

    // If we have seen this address before, prints reference
    if addr != 0 && visited.contains_key(&addr) {
        if filter.is_some() && !matches_filter && !force_show {
            return Ok(false);
        }
        let first_path = visited.get(&addr).unwrap();
        let display_name = match &node {
             Node::Group(_) => name.bright_blue().to_string(),
             Node::Dataset(_) => name.bold().to_string(),
        };
        println!("{}{}{} \t= {}", prefix, connector, display_name, first_path);
        return Ok(true);
    }

    let show_children = max_depth.map_or(true, |d| depth < d);
    let show_all_children = force_show || matches_filter;
    let mut child_entries: Vec<ChildEntry> = Vec::new();
    if let Node::Group(g) = &node {
        if show_children {
            let members = g.member_names()?;
            for member_name in members {
                let child_full_path = join_hdf5_path(&full_path, &member_name);
                match utils::get_link_info(g.id(), &member_name) {
                    utils::LinkInfo::Soft(target) => {
                        if show_all_children || filter.map_or(true, |f| f.is_match(&child_full_path)) {
                            child_entries.push(ChildEntry::Soft { name: member_name, target });
                        }
                        continue;
                    },
                    utils::LinkInfo::External { file, path } => {
                        if show_all_children || filter.map_or(true, |f| f.is_match(&child_full_path)) {
                            child_entries.push(ChildEntry::External { name: member_name, file, path });
                        }
                        continue;
                    },
                    _ => {}
                }

                let child_node = if let Ok(cg) = g.group(&member_name) {
                    Node::Group(cg)
                } else if let Ok(cd) = g.dataset(&member_name) {
                    Node::Dataset(cd)
                } else {
                    continue;
                };

                let child_should_print = if show_all_children {
                    true
                } else if let Some(filter) = filter {
                    node_matches_or_descendant(&child_node, filter, depth + 1, max_depth, filter_guard)?
                } else {
                    true
                };

                if child_should_print {
                    child_entries.push(ChildEntry::Node { name: member_name, node: child_node });
                }
            }
        }
    }

    let should_print = if filter.is_some() && !force_show {
        matches_filter || !child_entries.is_empty()
    } else {
        true
    };

    if !should_print {
        return Ok(false);
    }

    if addr != 0 {
        visited.insert(addr, full_path);
    }

    let (display_name, info, n_attrs) = match &node {
        Node::Group(g) => {
             let dname = name.bright_blue().to_string();
             let mut info = String::new();
             let n_attrs = g.attr_names()?.len();
             
             if !show_children {
                 let n = g.member_names()?.len();
                 info = format!("\t({} children)", n);
             }
             (dname, info, n_attrs)
        },
        Node::Dataset(ds) => {
             let dname = name.bold().to_string();
             let dtype = utils::fmt_dtype(&ds.dtype()?);
             let shape = utils::fmt_shape(&ds.shape());
             let info = format!("\t[{}: {}]", dtype, shape);
             let n_attrs = ds.attr_names()?.len();
             (dname, info, n_attrs)
        }
    };

    let mut final_info = info;
    if n_attrs > 0 && !expand_attrs {
         final_info.push_str(&format!(" ({} attributes)", n_attrs));
    }

    println!("{}{}{}{}", prefix, connector, display_name, final_info);

    let child_prefix_base = if depth == 0 { "" } else if is_last { "  " } else { "│ " };
    let child_prefix = format!("{}{}", prefix, child_prefix_base);

    if expand_attrs && n_attrs > 0 {
         match &node {
             Node::Group(g) => print_attrs(g, &child_prefix, fmt)?,
             Node::Dataset(d) => print_attrs(d, &child_prefix, fmt)?,
         }
    }

    if let Node::Group(_) = node {
        if show_children {
            let n_members = child_entries.len();
            for (i, entry) in child_entries.into_iter().enumerate() {
                let is_last_child = i == n_members - 1;
                
                match entry {
                    ChildEntry::Soft { name: member_name, target } => {
                        let connector = if is_last_child { "└" } else { "├" };
                        println!("{}{}{} -> {}", child_prefix, connector, member_name.bright_magenta(), target);
                        continue;
                    },
                    ChildEntry::External { name: member_name, file, path } => {
                        let connector = if is_last_child { "└" } else { "├" };
                        println!("{}{}{} -> {}/{}", child_prefix, connector, member_name.bright_magenta(), file, path);
                        continue;
                    },
                    ChildEntry::Node { name: member_name, node: child_node } => {
                        print_node_impl(child_node, &member_name, &child_prefix, is_last_child, depth + 1, expand_attrs, max_depth, visited, filter, filter_guard, show_all_children, fmt)?;
                    }
                };
            }
        }
    }

    Ok(true)
}

fn print_attrs(obj: &hdf5::Location, prefix: &str, fmt: &utils::NumFormat) -> Result<()> {
     let attrs = obj.attr_names()?;
     let n = attrs.len();
     println!("{}│ {} attributes:", prefix, n.to_string().yellow());
     
     for name in attrs {
         let attr = obj.attr(&name)?;
         let value = utils::format_attribute_value(&attr, fmt);
         println!("{}│  {}: {}", prefix, name, value);
     }
     Ok(())
}

fn join_hdf5_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", parent.trim_end_matches('/'), child)
    }
}

enum ChildEntry {
    Node { name: String, node: Node },
    Soft { name: String, target: String },
    External { name: String, file: String, path: String },
}

fn node_matches_or_descendant(node: &Node, filter: &PathFilter, depth: usize, max_depth: Option<usize>, guard: &mut HashSet<u64>) -> Result<bool> {
    let (obj_id, full_path) = match node {
        Node::Group(g) => (g.id(), g.name()),
        Node::Dataset(d) => (d.id(), d.name()),
    };

    if filter.is_match(&full_path) {
        return Ok(true);
    }

    let show_children = max_depth.map_or(true, |d| depth < d);
    let Node::Group(g) = node else {
        return Ok(false);
    };
    if !show_children {
        return Ok(false);
    }

    let addr = utils::get_object_addr(obj_id).unwrap_or(0);
    if addr != 0 {
        if guard.contains(&addr) {
            return Ok(false);
        }
        guard.insert(addr);
    }

    for member_name in g.member_names()? {
        let child_full_path = join_hdf5_path(&full_path, &member_name);
        match utils::get_link_info(g.id(), &member_name) {
            utils::LinkInfo::Soft(_) | utils::LinkInfo::External { .. } => {
                if filter.is_match(&child_full_path) {
                    if addr != 0 {
                        guard.remove(&addr);
                    }
                    return Ok(true);
                }
                continue;
            }
            _ => {}
        }

        let child_node = if let Ok(cg) = g.group(&member_name) {
            Node::Group(cg)
        } else if let Ok(cd) = g.dataset(&member_name) {
            Node::Dataset(cd)
        } else {
            continue;
        };

        if node_matches_or_descendant(&child_node, filter, depth + 1, max_depth, guard)? {
            if addr != 0 {
                guard.remove(&addr);
            }
            return Ok(true);
        }
    }

    if addr != 0 {
        guard.remove(&addr);
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_filter_matches_normalized_patterns() {
        let filter = PathFilter::new(&vec!["data".to_string(), "/entry/*/meta".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/run/meta"));
        assert!(filter.is_match("/entry/foo/data/bar"));
        assert!(!filter.is_match("/entry/run/other"));
    }

    #[test]
    fn path_filter_plain_string_matches_substring() {
        let filter = PathFilter::new(&vec!["data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/metadata"));
        assert!(filter.is_match("/entry/foo/data/bar"));
        assert!(!filter.is_match("/entry/other"));
    }

    #[test]
    fn path_filter_plain_string_with_leading_slash_is_substring() {
        let filter = PathFilter::new(&vec!["/entry/data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/foo/entry/data"));
    }

    #[test]
    fn path_filter_glob_star_does_not_cross_segments() {
        let filter = PathFilter::new(&vec!["/entry/*".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(!filter.is_match("/entry/data/foo"));
    }

    #[test]
    fn path_filter_glob_double_star_crosses_segments() {
        let filter = PathFilter::new(&vec!["/entry/**/data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/foo/data"));
        assert!(filter.is_match("/entry/a/b/data"));
        assert!(!filter.is_match("/entry/a/b/data/x"));
    }

    #[test]
    fn path_filter_glob_question_mark_matches_single_char() {
        let filter = PathFilter::new(&vec!["/entry/da?a".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/daaa"));
        assert!(!filter.is_match("/entry/daa"));
    }
}
