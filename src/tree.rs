use hdf5::{Group, Dataset};
use colored::Colorize;
use crate::utils;
use anyhow::{anyhow, Result};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use std::collections::{HashMap, HashSet};
use natord::compare;

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

pub fn print_group_tree(
    group: &Group,
    name: &str,
    expand_attrs: bool,
    max_depth: Option<usize>,
    sort_members: bool,
    filter: Option<&PathFilter>,
    fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
) -> Result<bool> {
    let mut printer = TreePrinter::new(
        expand_attrs,
        max_depth,
        sort_members,
        filter,
        fmt,
        truncate_attr_strings,
    );
    printer.print_node(Node::Group(group.clone()), name, "", true, 0, false)
}

enum Node {
    Group(Group),
    Dataset(Dataset),
}

struct TreePrinter<'a> {
    expand_attrs: bool,
    max_depth: Option<usize>,
    sort_members: bool,
    filter: Option<&'a PathFilter>,
    fmt: &'a utils::NumFormat,
    truncate_attr_strings: bool,
    visited: HashMap<u64, String>,
    filter_guard: HashSet<u64>,
}

impl<'a> TreePrinter<'a> {
    fn new(
        expand_attrs: bool,
        max_depth: Option<usize>,
        sort_members: bool,
        filter: Option<&'a PathFilter>,
        fmt: &'a utils::NumFormat,
        truncate_attr_strings: bool,
    ) -> Self {
        Self {
            expand_attrs,
            max_depth,
            sort_members,
            filter,
            fmt,
            truncate_attr_strings,
            visited: HashMap::new(),
            filter_guard: HashSet::new(),
        }
    }

    fn print_node(
        &mut self,
        node: Node,
        name: &str,
        prefix: &str,
        is_last: bool,
        depth: usize,
        force_show: bool,
    ) -> Result<bool> {
        let connector = if depth == 0 { "" } else if is_last { "└ " } else { "├ " };
        let filter = self.filter;

        // Check for visited (Hard Link cycle detection)
        // We need the ID of the object.
        let (obj_id, full_path) = match &node {
            Node::Group(g) => (g.id(), g.name()),
            Node::Dataset(d) => (d.id(), d.name()),
        };

        let matches_filter = filter.is_none_or(|f| f.is_match(&full_path));
        let addr = utils::get_object_addr(obj_id).unwrap_or(0); // If fails (0), we just don't dedup, or risk infinite loop? 0 is likely invalid addr.

        // If we have seen this address before, prints reference
        if addr != 0 && self.visited.contains_key(&addr) {
            if filter.is_some() && !matches_filter && !force_show {
                return Ok(false);
            }
            let first_path = self.visited.get(&addr).unwrap();
            let display_name = match &node {
                Node::Group(_) => name.bright_blue().to_string(),
                Node::Dataset(_) => name.bold().to_string(),
            };
            println!("{}{}{}  = {}", prefix, connector, display_name, first_path);
            return Ok(true);
        }

        let show_children = self.max_depth.is_none_or(|d| depth < d);
        let show_all_children = force_show || matches_filter;
        let mut child_entries: Vec<Child> = Vec::new();
        if let Node::Group(g) = &node {
            if show_children {
                for child in self.collect_children(g, &full_path)? {
                    match child {
                        Child::Soft { name, target, full_path: child_full_path } => {
                            if show_all_children || filter.is_none_or(|f| f.is_match(&child_full_path)) {
                                child_entries.push(Child::Soft { name, target, full_path: child_full_path });
                            }
                        }
                        Child::External { name, file, path, full_path: child_full_path } => {
                            if show_all_children || filter.is_none_or(|f| f.is_match(&child_full_path)) {
                                child_entries.push(Child::External { name, file, path, full_path: child_full_path });
                            }
                        }
                        Child::Node { name, node: child_node, full_path: child_full_path } => {
                            let child_should_print = if show_all_children {
                                true
                            } else if filter.is_some() {
                                self.node_matches_or_descendant(&child_node, depth + 1)?
                            } else {
                                true
                            };

                            if child_should_print {
                                child_entries.push(Child::Node { name, node: child_node, full_path: child_full_path });
                            }
                        }
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
            self.visited.insert(addr, full_path);
        }

        let (display_name, info, n_attrs) = match &node {
            Node::Group(g) => {
                let dname = name.bright_blue().to_string();
                let mut info = String::new();
                let n_attrs = g.attr_names()?.len();

                if !show_children {
                    let n = g.member_names()?.len();
                    info = format!(" ({0} children)", n);
                }
                (dname, info, n_attrs)
            }
            Node::Dataset(ds) => {
                let dname = name.bold().to_string();
                let dtype = utils::fmt_dtype(&ds.dtype()?);
                let shape = utils::fmt_shape(&ds.shape());
                let info = format!(" [{0}: {1}]", dtype, shape);
                let n_attrs = ds.attr_names()?.len();
                (dname, info, n_attrs)
            }
        };

        let mut final_info = info;
        if n_attrs > 0 && !self.expand_attrs {
            final_info.push_str(&format!(" ({} attributes)", n_attrs));
        }

        println!("{}{}{}{}", prefix, connector, display_name, final_info);

        let child_prefix_base = if depth == 0 { "" } else if is_last { "  " } else { "│ " };
        let child_prefix = format!("{}{}", prefix, child_prefix_base);

        if self.expand_attrs && n_attrs > 0 {
            match &node {
                Node::Group(g) => print_attrs(g, &child_prefix, self.fmt, self.truncate_attr_strings)?,
                Node::Dataset(d) => print_attrs(d, &child_prefix, self.fmt, self.truncate_attr_strings)?,
            }
        }

        if let Node::Group(_) = node {
            if show_children {
                let n_members = child_entries.len();
                for (i, entry) in child_entries.into_iter().enumerate() {
                    let is_last_child = i == n_members - 1;

                    match entry {
                        Child::Soft { name: member_name, target, .. } => {
                            let connector = if is_last_child { "└" } else { "├" };
                            println!("{}{}{} -> {}", child_prefix, connector, member_name.bright_magenta(), target);
                            continue;
                        }
                        Child::External { name: member_name, file, path, .. } => {
                            let connector = if is_last_child { "└" } else { "├" };
                            let display_target = if path.starts_with('/') {
                                format!("{}{}", file, path)
                            } else {
                                format!("{}/{}", file, path)
                            };
                            println!("{}{}{} -> {}", child_prefix, connector, member_name.bright_magenta(), display_target);
                            continue;
                        }
                        Child::Node { name: member_name, node: child_node, .. } => {
                            self.print_node(child_node, &member_name, &child_prefix, is_last_child, depth + 1, show_all_children)?;
                        }
                    };
                }
            }
        }

        Ok(true)
    }

    fn node_matches_or_descendant(&mut self, node: &Node, depth: usize) -> Result<bool> {
        let Some(filter) = self.filter else {
            return Ok(true);
        };

        let (obj_id, full_path) = match node {
            Node::Group(g) => (g.id(), g.name()),
            Node::Dataset(d) => (d.id(), d.name()),
        };

        if filter.is_match(&full_path) {
            return Ok(true);
        }

        let show_children = self.max_depth.is_none_or(|d| depth < d);
        let Node::Group(g) = node else {
            return Ok(false);
        };
        if !show_children {
            return Ok(false);
        }

        let addr = utils::get_object_addr(obj_id).unwrap_or(0);
        if addr != 0 {
            if self.filter_guard.contains(&addr) {
                return Ok(false);
            }
            self.filter_guard.insert(addr);
        }

        for child in self.collect_children(g, &full_path)? {
            match child {
                Child::Soft { full_path: child_full_path, .. }
                | Child::External { full_path: child_full_path, .. } => {
                    if filter.is_match(&child_full_path) {
                        if addr != 0 {
                            self.filter_guard.remove(&addr);
                        }
                        return Ok(true);
                    }
                }
                Child::Node { node: child_node, .. } => {
                    if self.node_matches_or_descendant(&child_node, depth + 1)? {
                        if addr != 0 {
                            self.filter_guard.remove(&addr);
                        }
                        return Ok(true);
                    }
                }
            }
        }

        if addr != 0 {
            self.filter_guard.remove(&addr);
        }
        Ok(false)
    }

    fn collect_children(&self, group: &Group, parent_path: &str) -> Result<Vec<Child>> {
        let mut members = group.member_names()?;
        if self.sort_members {
            members.sort_by(|a, b| compare(a, b));
        }

        let mut children = Vec::with_capacity(members.len());
        for member_name in members {
            let child_full_path = join_hdf5_path(parent_path, &member_name);
            match utils::get_link_info(group.id(), &member_name) {
                utils::LinkInfo::Soft(target) => {
                    children.push(Child::Soft { name: member_name, target, full_path: child_full_path });
                    continue;
                }
                utils::LinkInfo::External { file, path } => {
                    children.push(Child::External { name: member_name, file, path, full_path: child_full_path });
                    continue;
                }
                _ => {}
            }

            let child_node = if let Ok(cg) = group.group(&member_name) {
                Node::Group(cg)
            } else if let Ok(cd) = group.dataset(&member_name) {
                Node::Dataset(cd)
            } else {
                continue;
            };

            children.push(Child::Node { name: member_name, node: child_node, full_path: child_full_path });
        }

        Ok(children)
    }
}

fn print_attrs(obj: &hdf5::Location, prefix: &str, fmt: &utils::NumFormat, truncate_attr_strings: bool) -> Result<()> {
     let attrs = obj.attr_names()?;
     let n = attrs.len();
     println!("{}│ {} attributes:", prefix, n.to_string().yellow());
     
     for name in attrs {
         let attr = obj.attr(&name)?;
         let value = utils::format_attribute_value(&attr, fmt, truncate_attr_strings);
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

enum Child {
    Node { name: String, node: Node, full_path: String },
    Soft { name: String, target: String, full_path: String },
    External { name: String, file: String, path: String, full_path: String },
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
