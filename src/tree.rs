use hdf5::{Group, Dataset};
use colored::Colorize;
use crate::utils;
use anyhow::Result;
use std::collections::HashMap;

pub fn print_group_tree(group: &Group, name: &str, expand_attrs: bool, max_depth: Option<usize>) -> Result<()> {
    let mut visited = HashMap::new();
    print_node_impl(Node::Group(group.clone()), name, "", true, 0, expand_attrs, max_depth, &mut visited)
}

enum Node {
    Group(Group),
    Dataset(Dataset),
}

fn print_node_impl(node: Node, name: &str, prefix: &str, is_last: bool, depth: usize, expand_attrs: bool, max_depth: Option<usize>, visited: &mut HashMap<u64, String>) -> Result<()> {
    let connector = if depth == 0 { "" } else if is_last { "└" } else { "├" };
    
    // Check for visited (Hard Link cycle detection)
    // We need the ID of the object.
    let (obj_id, full_path) = match &node {
        Node::Group(g) => (g.id(), g.name()),
        Node::Dataset(d) => (d.id(), d.name()),
    };

    let addr = utils::get_object_addr(obj_id).unwrap_or(0); // If fails (0), we just don't dedup, or risk infinite loop? 0 is likely invalid addr.

    // If we have seen this address before, prints reference
    if addr != 0 && visited.contains_key(&addr) {
        let first_path = visited.get(&addr).unwrap();
        let display_name = match &node {
             Node::Group(_) => name.bright_blue().to_string(),
             Node::Dataset(_) => name.bold().to_string(),
        };
        println!("{}{}{} \t= {}", prefix, connector, display_name, first_path);
        return Ok(());
    }

    if addr != 0 {
        visited.insert(addr, full_path);
    }

    let (display_name, info, n_attrs) = match &node {
        Node::Group(g) => {
             let dname = name.bright_blue().to_string();
             let mut info = String::new();
             let n_attrs = g.attr_names()?.len();
             
             let show_children = max_depth.map_or(true, |d| depth < d);
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
             Node::Group(g) => print_attrs(g, &child_prefix)?,
             Node::Dataset(d) => print_attrs(d, &child_prefix)?,
         }
    }

    if let Node::Group(g) = node {
        let show_children = max_depth.map_or(true, |d| depth < d);
        if show_children {
            let members = g.member_names()?;
            let n_members = members.len();
            for (i, member_name) in members.iter().enumerate() {
                let is_last_child = i == n_members - 1;
                
                match utils::get_link_info(g.id(), member_name) {
                    utils::LinkInfo::Soft(target) => {
                        let connector = if is_last_child { "└" } else { "├" };
                        println!("{}{}{} -> {}", child_prefix, connector, member_name.bright_magenta(), target);
                        continue;
                    },
                    utils::LinkInfo::External { file, path } => {
                        let connector = if is_last_child { "└" } else { "├" };
                        println!("{}{}{} -> {}/{}", child_prefix, connector, member_name.bright_magenta(), file, path);
                        continue;
                    },
                    _ => {}
                }

                let child_node = if let Ok(cg) = g.group(member_name) {
                    Node::Group(cg)
                } else if let Ok(cd) = g.dataset(member_name) {
                    Node::Dataset(cd)
                } else {
                    continue; 
                };
                
                print_node_impl(child_node, member_name, &child_prefix, is_last_child, depth + 1, expand_attrs, max_depth, visited)?;
            }
        }
    }

    Ok(())
}

fn print_attrs(obj: &hdf5::Location, prefix: &str) -> Result<()> {
     let attrs = obj.attr_names()?;
     let n = attrs.len();
     println!("{}│ {} attributes:", prefix, n.to_string().yellow());
     
     for name in attrs {
         let attr = obj.attr(&name)?;
         let value = utils::format_attribute_value(&attr);
         println!("{}│  {}: {}", prefix, name, value);
     }
     Ok(())
}
