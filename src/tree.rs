use crate::utils;
use anyhow::{anyhow, Result};
use colored::Colorize;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use hdf5::{Dataset, Group};
use natord::compare;
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
        let set = builder
            .build()
            .map_err(|e| anyhow!("Invalid filter patterns: {}", e))?;
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

pub struct TreePrintOptions<'a> {
    pub expand_attrs: bool,
    pub max_depth: Option<usize>,
    pub sort_members: bool,
    pub filter: Option<&'a PathFilter>,
    pub fmt: utils::NumFormat,
    pub truncate_attr_strings: bool,
}

impl<'a> TreePrintOptions<'a> {
    pub fn new(fmt: utils::NumFormat) -> Self {
        Self {
            expand_attrs: false,
            max_depth: None,
            sort_members: true,
            filter: None,
            fmt,
            truncate_attr_strings: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TreeNodeKind {
    Group,
    Dataset,
    SoftLink,
    ExternalLink,
}

#[derive(Clone, Debug)]
pub enum TreeLink {
    Soft { target: String },
    External { file: String, path: String },
}

#[derive(Clone, Debug)]
pub struct TreeAttribute {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub kind: TreeNodeKind,
    pub name: String,
    pub path: String,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
    pub attributes_count: Option<usize>,
    pub attributes: Option<Vec<TreeAttribute>>,
    pub children: Option<Vec<TreeNode>>,
    pub children_count: Option<usize>,
    pub hard_link_to: Option<String>,
    pub link: Option<TreeLink>,
}

pub fn build_group_tree_model(
    group: &Group,
    name: &str,
    opts: &TreePrintOptions<'_>,
) -> Result<Option<TreeNode>> {
    let mut builder = TreeBuilder::new(
        opts.expand_attrs,
        opts.max_depth,
        opts.sort_members,
        opts.filter,
        &opts.fmt,
        opts.truncate_attr_strings,
    );
    let full_path = group.name();
    builder.build_node(
        Node::Group(group.clone()),
        name.to_string(),
        full_path,
        0,
        false,
    )
}

pub fn print_group_tree(group: &Group, name: &str, opts: &TreePrintOptions<'_>) -> Result<bool> {
    let tree = build_group_tree_model(group, name, opts)?;
    if let Some(node) = tree {
        print_tree(&node, "", true, 0)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

enum Node {
    Group(Group),
    Dataset(Dataset),
}

enum Child {
    Node {
        name: String,
        node: Node,
        full_path: String,
    },
    Soft {
        name: String,
        target: String,
        full_path: String,
    },
    External {
        name: String,
        file: String,
        path: String,
        full_path: String,
    },
}

struct TreeBuilder<'a> {
    expand_attrs: bool,
    max_depth: Option<usize>,
    sort_members: bool,
    filter: Option<&'a PathFilter>,
    fmt: &'a utils::NumFormat,
    truncate_attr_strings: bool,
    visited: HashMap<u64, String>,
    filter_guard: HashSet<u64>,
}

impl<'a> TreeBuilder<'a> {
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

    fn build_node(
        &mut self,
        node: Node,
        name: String,
        full_path: String,
        depth: usize,
        force_show: bool,
    ) -> Result<Option<TreeNode>> {
        let filter = self.filter;
        let (obj_id, _) = match &node {
            Node::Group(g) => (g.id(), g.name()),
            Node::Dataset(d) => (d.id(), d.name()),
        };

        let matches_filter = filter.is_none_or(|f| f.is_match(&full_path));
        let addr = utils::get_object_addr(obj_id).unwrap_or(0);

        if addr != 0 && self.visited.contains_key(&addr) {
            if filter.is_some() && !matches_filter && !force_show {
                return Ok(None);
            }
            let first_path = self.visited.get(&addr).cloned().unwrap_or_default();
            let kind = match &node {
                Node::Group(_) => TreeNodeKind::Group,
                Node::Dataset(_) => TreeNodeKind::Dataset,
            };
            return Ok(Some(TreeNode {
                kind,
                name,
                path: full_path,
                dtype: None,
                shape: None,
                attributes_count: None,
                attributes: None,
                children: None,
                children_count: None,
                hard_link_to: Some(first_path),
                link: None,
            }));
        }

        let show_children = self.max_depth.is_none_or(|d| depth < d);
        let show_all_children = force_show || matches_filter;
        let mut child_entries: Vec<Child> = Vec::new();
        if let Node::Group(g) = &node {
            if show_children {
                for child in self.collect_children(g, &full_path)? {
                    match child {
                        Child::Soft {
                            name,
                            target,
                            full_path: child_full_path,
                        } => {
                            if show_all_children
                                || filter.is_none_or(|f| f.is_match(&child_full_path))
                            {
                                child_entries.push(Child::Soft {
                                    name,
                                    target,
                                    full_path: child_full_path,
                                });
                            }
                        }
                        Child::External {
                            name,
                            file,
                            path,
                            full_path: child_full_path,
                        } => {
                            if show_all_children
                                || filter.is_none_or(|f| f.is_match(&child_full_path))
                            {
                                child_entries.push(Child::External {
                                    name,
                                    file,
                                    path,
                                    full_path: child_full_path,
                                });
                            }
                        }
                        Child::Node {
                            name,
                            node: child_node,
                            full_path: child_full_path,
                        } => {
                            let child_should_include = if show_all_children {
                                true
                            } else if filter.is_some() {
                                self.node_matches_or_descendant(&child_node, depth + 1)?
                            } else {
                                true
                            };

                            if child_should_include {
                                child_entries.push(Child::Node {
                                    name,
                                    node: child_node,
                                    full_path: child_full_path,
                                });
                            }
                        }
                    }
                }
            }
        }

        let should_include = if filter.is_some() && !force_show {
            matches_filter || !child_entries.is_empty()
        } else {
            true
        };

        if !should_include {
            return Ok(None);
        }

        if addr != 0 {
            self.visited.insert(addr, full_path.clone());
        }

        let mut node_out = match &node {
            Node::Group(g) => {
                let (attributes_count, attributes) = collect_attributes_info(
                    g,
                    self.fmt,
                    self.truncate_attr_strings,
                    self.expand_attrs,
                )?;
                let children_count = if show_children {
                    None
                } else {
                    Some(g.member_names()?.len())
                };
                TreeNode {
                    kind: TreeNodeKind::Group,
                    name,
                    path: full_path,
                    dtype: None,
                    shape: None,
                    attributes_count: Some(attributes_count),
                    attributes,
                    children: None,
                    children_count,
                    hard_link_to: None,
                    link: None,
                }
            }
            Node::Dataset(d) => {
                let (attributes_count, attributes) = collect_attributes_info(
                    d,
                    self.fmt,
                    self.truncate_attr_strings,
                    self.expand_attrs,
                )?;
                TreeNode {
                    kind: TreeNodeKind::Dataset,
                    name,
                    path: full_path,
                    dtype: Some(utils::fmt_dtype(&d.dtype()?)),
                    shape: Some(d.shape()),
                    attributes_count: Some(attributes_count),
                    attributes,
                    children: None,
                    children_count: None,
                    hard_link_to: None,
                    link: None,
                }
            }
        };

        if let Node::Group(_) = node {
            if show_children {
                let mut children = Vec::new();
                for child in child_entries {
                    match child {
                        Child::Soft {
                            name,
                            target,
                            full_path,
                        } => children.push(TreeNode {
                            kind: TreeNodeKind::SoftLink,
                            name,
                            path: full_path,
                            dtype: None,
                            shape: None,
                            attributes_count: None,
                            attributes: None,
                            children: None,
                            children_count: None,
                            hard_link_to: None,
                            link: Some(TreeLink::Soft { target }),
                        }),
                        Child::External {
                            name,
                            file,
                            path,
                            full_path,
                        } => children.push(TreeNode {
                            kind: TreeNodeKind::ExternalLink,
                            name,
                            path: full_path,
                            dtype: None,
                            shape: None,
                            attributes_count: None,
                            attributes: None,
                            children: None,
                            children_count: None,
                            hard_link_to: None,
                            link: Some(TreeLink::External { file, path }),
                        }),
                        Child::Node {
                            name,
                            node,
                            full_path,
                        } => {
                            if let Some(child_node) = self.build_node(
                                node,
                                name,
                                full_path,
                                depth + 1,
                                show_all_children,
                            )? {
                                children.push(child_node);
                            }
                        }
                    }
                }
                node_out.children = Some(children);
            }
        }

        Ok(Some(node_out))
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
                Child::Soft {
                    full_path: child_full_path,
                    ..
                }
                | Child::External {
                    full_path: child_full_path,
                    ..
                } => {
                    if filter.is_match(&child_full_path) {
                        if addr != 0 {
                            self.filter_guard.remove(&addr);
                        }
                        return Ok(true);
                    }
                }
                Child::Node {
                    node: child_node, ..
                } => {
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
            let child_full_path = utils::join_hdf5_path(parent_path, &member_name);
            match utils::get_link_info(group.id(), &member_name) {
                utils::LinkInfo::Soft(target) => {
                    children.push(Child::Soft {
                        name: member_name,
                        target,
                        full_path: child_full_path,
                    });
                    continue;
                }
                utils::LinkInfo::External { file, path } => {
                    children.push(Child::External {
                        name: member_name,
                        file,
                        path,
                        full_path: child_full_path,
                    });
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

            children.push(Child::Node {
                name: member_name,
                node: child_node,
                full_path: child_full_path,
            });
        }

        Ok(children)
    }
}

fn collect_attributes_info(
    obj: &hdf5::Location,
    fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
    include: bool,
) -> Result<(usize, Option<Vec<TreeAttribute>>)> {
    let names = obj.attr_names()?;
    let count = names.len();
    if !include {
        return Ok((count, None));
    }

    let mut attrs = Vec::with_capacity(count);
    for name in names {
        let attr = obj.attr(&name)?;
        let value = utils::format_attribute_value(&attr, fmt, truncate_attr_strings);
        attrs.push(TreeAttribute { name, value });
    }
    Ok((count, Some(attrs)))
}

fn print_tree(node: &TreeNode, prefix: &str, is_last: bool, depth: usize) -> Result<()> {
    let connector = if depth == 0 {
        ""
    } else if is_last {
        "└ "
    } else {
        "├ "
    };

    match node.kind {
        TreeNodeKind::SoftLink | TreeNodeKind::ExternalLink => {
            let connector = if depth == 0 {
                ""
            } else if is_last {
                "└"
            } else {
                "├"
            };
            let target = match node.link.as_ref() {
                Some(TreeLink::Soft { target }) => target.clone(),
                Some(TreeLink::External { file, path }) => {
                    if path.starts_with('/') {
                        format!("{}{}", file, path)
                    } else {
                        format!("{}/{}", file, path)
                    }
                }
                None => String::new(),
            };
            println!(
                "{}{}{} -> {}",
                prefix,
                connector,
                node.name.bright_magenta(),
                target
            );
            return Ok(());
        }
        _ => {}
    }

    if let Some(target) = node.hard_link_to.as_ref() {
        let display_name = match node.kind {
            TreeNodeKind::Group => node.name.bright_blue().to_string(),
            TreeNodeKind::Dataset => node.name.bold().to_string(),
            _ => node.name.clone(),
        };
        println!("{}{}{}  = {}", prefix, connector, display_name, target);
        return Ok(());
    }

    let mut info = String::new();
    let display_name = match node.kind {
        TreeNodeKind::Group => {
            if let Some(n_children) = node.children_count {
                info = format!(" ({0} children)", n_children);
            }
            node.name.bright_blue().to_string()
        }
        TreeNodeKind::Dataset => {
            if let (Some(dtype), Some(shape)) = (node.dtype.as_ref(), node.shape.as_ref()) {
                info = format!(" [{0}: {1}]", dtype, utils::fmt_shape(shape));
            }
            node.name.bold().to_string()
        }
        _ => node.name.clone(),
    };

    let mut final_info = info;
    if let Some(n_attrs) = node.attributes_count {
        if n_attrs > 0 && node.attributes.is_none() {
            final_info.push_str(&format!(" ({} attributes)", n_attrs));
        }
    }

    println!("{}{}{}{}", prefix, connector, display_name, final_info);

    let child_prefix_base = if depth == 0 {
        ""
    } else if is_last {
        "  "
    } else {
        "│ "
    };
    let child_prefix = format!("{}{}", prefix, child_prefix_base);

    if let Some(attrs) = node.attributes.as_ref() {
        if node.attributes_count.unwrap_or(0) > 0 {
            let n = attrs.len();
            println!("{}│ {} attributes:", child_prefix, n.to_string().yellow());
            for attr in attrs {
                println!("{}│  {}: {}", child_prefix, attr.name, attr.value);
            }
        }
    }

    if let Some(children) = node.children.as_ref() {
        let n_members = children.len();
        for (i, child) in children.iter().enumerate() {
            let is_last_child = i == n_members - 1;
            print_tree(child, &child_prefix, is_last_child, depth + 1)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_filter_matches_normalized_patterns() {
        let filter = PathFilter::new(&["data".to_string(), "/entry/*/meta".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/run/meta"));
        assert!(filter.is_match("/entry/foo/data/bar"));
        assert!(!filter.is_match("/entry/run/other"));
    }

    #[test]
    fn path_filter_plain_string_matches_substring() {
        let filter = PathFilter::new(&["data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/metadata"));
        assert!(filter.is_match("/entry/foo/data/bar"));
        assert!(!filter.is_match("/entry/other"));
    }

    #[test]
    fn path_filter_plain_string_with_leading_slash_is_substring() {
        let filter = PathFilter::new(&["/entry/data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/foo/entry/data"));
    }

    #[test]
    fn path_filter_glob_star_does_not_cross_segments() {
        let filter = PathFilter::new(&["/entry/*".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(!filter.is_match("/entry/data/foo"));
    }

    #[test]
    fn path_filter_glob_double_star_crosses_segments() {
        let filter = PathFilter::new(&["/entry/**/data".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/foo/data"));
        assert!(filter.is_match("/entry/a/b/data"));
        assert!(!filter.is_match("/entry/a/b/data/x"));
    }

    #[test]
    fn path_filter_glob_question_mark_matches_single_char() {
        let filter = PathFilter::new(&["/entry/da?a".to_string()]).unwrap();
        assert!(filter.is_match("/entry/data"));
        assert!(filter.is_match("/entry/daaa"));
        assert!(!filter.is_match("/entry/daa"));
    }
}
