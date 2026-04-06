use crate::slicing;
use crate::tree;
use crate::utils;
use anyhow::{anyhow, Result};
use hdf5::plist::dataset_create::Layout;
use hdf5::{Dataset, Group};
use serde::Serialize;
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

pub fn build_group_tree_json(
    group: &Group,
    opts: &tree::TreePrintOptions<'_>,
) -> Result<Option<JsonNode>> {
    let name = name_from_path(&group.name());
    let model = tree::build_group_tree_model(group, &name, opts)?;
    Ok(model.map(tree_node_to_json))
}

fn tree_node_to_json(node: tree::TreeNode) -> JsonNode {
    let is_hard_link = node.hard_link_to.is_some();
    let kind = if is_hard_link {
        NodeKind::HardLink
    } else {
        match node.kind {
            tree::TreeNodeKind::Group => NodeKind::Group,
            tree::TreeNodeKind::Dataset => NodeKind::Dataset,
            tree::TreeNodeKind::SoftLink => NodeKind::SoftLink,
            tree::TreeNodeKind::ExternalLink => NodeKind::ExternalLink,
        }
    };

    let link = match node.link {
        Some(tree::TreeLink::Soft { target }) => Some(JsonLink::Soft { target }),
        Some(tree::TreeLink::External { file, path }) => Some(JsonLink::External { file, path }),
        None => None,
    };

    let children = node
        .children
        .map(|children| children.into_iter().map(tree_node_to_json).collect());

    let attributes = node.attributes.map(|attrs| {
        attrs
            .into_iter()
            .map(|attr| JsonAttribute {
                name: attr.name,
                value: attr.value,
            })
            .collect()
    });

    JsonNode {
        kind,
        name: node.name,
        path: node.path,
        dtype: if is_hard_link { None } else { node.dtype },
        shape: if is_hard_link { None } else { node.shape },
        attributes_count: if is_hard_link {
            None
        } else {
            node.attributes_count
        },
        attributes,
        children,
        hard_link_to: node.hard_link_to,
        link,
    }
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
        slicing::parse_slice(expr, &shape).map_err(|e| anyhow!("Error parsing slice: {}", e))?;
    }

    let elements = utils::elem_count_u64(&shape);
    let storage_bytes = ds.storage_size();
    let logical_size_bytes = desc.as_ref().and_then(|desc| {
        if utils::descriptor_has_vlen(desc) {
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

fn name_from_path(path: &str) -> String {
    if path == "/" {
        "/".to_string()
    } else {
        path.rsplit('/').next().unwrap_or(path).to_string()
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
