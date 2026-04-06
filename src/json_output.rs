use crate::slicing;
use crate::tree::PathFilter;
use crate::utils;
use anyhow::{anyhow, Result};
use hdf5::plist::dataset_create::Layout;
use hdf5::types::TypeDescriptor;
use hdf5::{Dataset, Group};
use natord::compare;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

#[derive(Serialize)]
pub struct JsonErrorOutput {
    pub error: String,
    pub code: i32,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Group,
    Dataset,
}

#[derive(Serialize)]
pub struct JsonOutput {
    pub kind: OutputKind,
    pub file: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree: Option<JsonNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset: Option<DatasetInfo>,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Group,
    Dataset,
    SoftLink,
    ExternalLink,
    HardLink,
}

#[derive(Serialize)]
pub struct JsonNode {
    pub kind: NodeKind,
    pub name: String,
    pub path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<JsonAttribute>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<JsonNode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hard_link_to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub link: Option<JsonLink>,
}

#[derive(Serialize)]
pub struct JsonAttribute {
    pub name: String,
    pub value: String,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum JsonLink {
    Soft { target: String },
    External { file: String, path: String },
}

#[derive(Serialize)]
pub struct DatasetInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub elements: Option<u64>,
    pub storage_bytes: u64,
    pub logical_size_bytes: Option<u64>,
    pub compression_ratio: Option<f64>,
    pub layout: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<Vec<usize>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maxshape: Option<Vec<Option<usize>>>,
    pub attributes_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attributes: Option<Vec<JsonAttribute>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slice: Option<String>,
    pub data_included: bool,
}

pub fn write_json<T: Serialize>(value: &T, pretty: bool) -> Result<()> {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    if pretty {
        serde_json::to_writer_pretty(&mut handle, value)?;
    } else {
        serde_json::to_writer(&mut handle, value)?;
    }
    handle.write_all(b"\n")?;
    Ok(())
}

pub struct TreeJsonOptions<'a> {
    pub expand_attrs: bool,
    pub max_depth: Option<usize>,
    pub sort_members: bool,
    pub filter: Option<&'a PathFilter>,
    pub fmt: utils::NumFormat,
    pub truncate_attr_strings: bool,
}

impl<'a> TreeJsonOptions<'a> {
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

pub fn build_group_tree(group: &Group, opts: &TreeJsonOptions<'_>) -> Result<Option<JsonNode>> {
    let mut builder = TreeJsonBuilder::new(
        opts.expand_attrs,
        opts.max_depth,
        opts.sort_members,
        opts.filter,
        &opts.fmt,
        opts.truncate_attr_strings,
    );
    let full_path = group.name();
    let name = name_from_path(&full_path);
    builder.build_node(Node::Group(group.clone()), name, full_path, 0, false)
}

pub fn build_dataset_info(
    ds: &Dataset,
    fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
    include_attrs: bool,
    slice_expr: Option<&str>,
) -> Result<DatasetInfo> {
    let dtype = ds.dtype()?;
    let desc = dtype.to_descriptor().ok();
    let shape = ds.shape();

    if let Some(expr) = slice_expr {
        slicing::parse_slice(expr, &shape)
            .map_err(|e| anyhow!("Error parsing slice: {}", e))?;
    }

    let elements = elem_count_u64(&shape);
    let storage_bytes = ds.storage_size();
    let logical_size_bytes = desc
        .as_ref()
        .and_then(|desc| {
            if descriptor_has_vlen(desc) {
                None
            } else {
                elements.and_then(|count| count.checked_mul(dtype.size() as u64))
            }
        });

    let compression_ratio = match logical_size_bytes {
        Some(logical) if logical > 0 && storage_bytes > 0 => {
            Some(logical as f64 / storage_bytes as f64)
        }
        _ => None,
    };

    let create_plist = ds.create_plist()?;
    let layout = create_plist.layout();
    let mut chunk = None;
    let mut compression = None;
    if layout == Layout::Chunked {
        if let Some(chunks) = create_plist.chunk() {
            chunk = Some(chunks);
        }
        let filters = create_plist.filters();
        if !filters.is_empty() {
            compression = Some(filters.iter().map(|f| format!("{:?}", f)).collect());
        }
    }

    let maxshape = maxshape_if_different(ds, &shape)?;

    let (attributes_count, attributes) =
        collect_attributes_info(ds, fmt, truncate_attr_strings, include_attrs)?;

    Ok(DatasetInfo {
        dtype: utils::fmt_dtype(&dtype),
        shape,
        elements,
        storage_bytes,
        logical_size_bytes,
        compression_ratio,
        layout: format!("{:?}", layout),
        chunk,
        compression,
        maxshape,
        attributes_count,
        attributes,
        slice: slice_expr.map(|s| s.to_string()),
        data_included: false,
    })
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

struct TreeJsonBuilder<'a> {
    expand_attrs: bool,
    max_depth: Option<usize>,
    sort_members: bool,
    filter: Option<&'a PathFilter>,
    fmt: &'a utils::NumFormat,
    truncate_attr_strings: bool,
    visited: HashMap<u64, String>,
    filter_guard: HashSet<u64>,
}

impl<'a> TreeJsonBuilder<'a> {
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
    ) -> Result<Option<JsonNode>> {
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
            return Ok(Some(JsonNode {
                kind: NodeKind::HardLink,
                name,
                path: full_path,
                dtype: None,
                shape: None,
                attributes_count: None,
                attributes: None,
                children: None,
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

        let mut json = match &node {
            Node::Group(g) => {
                let (attributes_count, attributes) = collect_attributes_info(
                    g,
                    self.fmt,
                    self.truncate_attr_strings,
                    self.expand_attrs,
                )?;
                JsonNode {
                    kind: NodeKind::Group,
                    name,
                    path: full_path,
                    dtype: None,
                    shape: None,
                    attributes_count: Some(attributes_count),
                    attributes,
                    children: None,
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
                JsonNode {
                    kind: NodeKind::Dataset,
                    name,
                    path: full_path,
                    dtype: Some(utils::fmt_dtype(&d.dtype()?)),
                    shape: Some(d.shape()),
                    attributes_count: Some(attributes_count),
                    attributes,
                    children: None,
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
                        } => children.push(JsonNode {
                            kind: NodeKind::SoftLink,
                            name,
                            path: full_path,
                            dtype: None,
                            shape: None,
                            attributes_count: None,
                            attributes: None,
                            children: None,
                            hard_link_to: None,
                            link: Some(JsonLink::Soft { target }),
                        }),
                        Child::External {
                            name,
                            file,
                            path,
                            full_path,
                        } => children.push(JsonNode {
                            kind: NodeKind::ExternalLink,
                            name,
                            path: full_path,
                            dtype: None,
                            shape: None,
                            attributes_count: None,
                            attributes: None,
                            children: None,
                            hard_link_to: None,
                            link: Some(JsonLink::External { file, path }),
                        }),
                        Child::Node {
                            name,
                            node,
                            full_path,
                        } => {
                            if let Some(child_json) =
                                self.build_node(node, name, full_path, depth + 1, show_all_children)?
                            {
                                children.push(child_json);
                            }
                        }
                    }
                }
                json.children = Some(children);
            }
        }

        Ok(Some(json))
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

fn name_from_path(path: &str) -> String {
    if path == "/" {
        "/".to_string()
    } else {
        path.rsplit('/').next().unwrap_or(path).to_string()
    }
}

fn join_hdf5_path(parent: &str, child: &str) -> String {
    if parent == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", parent.trim_end_matches('/'), child)
    }
}

fn collect_attributes_info(
    obj: &hdf5::Location,
    fmt: &utils::NumFormat,
    truncate_attr_strings: bool,
    include: bool,
) -> Result<(usize, Option<Vec<JsonAttribute>>)> {
    let names = obj.attr_names()?;
    let count = names.len();
    if !include {
        return Ok((count, None));
    }

    let mut attrs = Vec::with_capacity(count);
    for name in names {
        let attr = obj.attr(&name)?;
        let value = utils::format_attribute_value(&attr, fmt, truncate_attr_strings);
        attrs.push(JsonAttribute { name, value });
    }
    Ok((count, Some(attrs)))
}

fn elem_count_u64(shape: &[usize]) -> Option<u64> {
    if shape.is_empty() {
        Some(1u64)
    } else {
        shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
    }
}

fn descriptor_has_vlen(desc: &TypeDescriptor) -> bool {
    match desc {
        TypeDescriptor::VarLenArray(_)
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::VarLenUnicode => true,
        TypeDescriptor::FixedArray(inner, _) => descriptor_has_vlen(inner),
        TypeDescriptor::Compound(compound) => compound
            .fields
            .iter()
            .any(|field| descriptor_has_vlen(&field.ty)),
        _ => false,
    }
}

fn maxshape_if_different(ds: &Dataset, shape: &[usize]) -> Result<Option<Vec<Option<usize>>>> {
    let Ok(space) = ds.space() else {
        return Ok(None);
    };
    let maxshape = space.maxdims();
    let different = maxshape.len() != shape.len()
        || maxshape
            .iter()
            .zip(shape.iter())
            .any(|(m, s)| m.is_none_or(|mv| mv != *s));
    if different {
        Ok(Some(maxshape))
    } else {
        Ok(None)
    }
}
