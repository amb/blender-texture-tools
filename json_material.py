import json


def read_material_nodes_to_json(mat):
    node_tree = mat.node_tree
    d_nodes = {}

    # TODO: calc max and min bounding box, save center value
    for n in node_tree.nodes.values():
        d_temp = {}

        d_temp["bl_idname"] = n.bl_idname
        d_temp["dimensions"] = tuple(i for i in n.dimensions)
        d_temp["location"] = tuple(i for i in n.location)
        d_temp["name"] = n.name

        n_inputs = []
        for i in n.inputs.values():
            o_val = {"input_name": i.name, "input_type": "none"}
            if len(i.links) == 0:
                if hasattr(i, "default_value"):
                    if i.bl_idname == "NodeSocketFloatFactor":
                        o_val["input_type"] = "float"
                        o_val["value"] = i.default_value.real
            else:
                l = i.links[0]
                o_val["input_type"] = "node"
                o_val["from_node_name"] = l.from_node.name
                o_val["from_socket_name"] = l.from_socket.name
            n_inputs.append(o_val)
        d_temp["inputs"] = n_inputs

        # n_outputs = []
        # for i in n.outputs.values():
        #     if len(i.links) == 0:
        #         continue
        #     for l in i.links:
        #         n_outputs.append(l.from_node.name)
        # d_temp["outputs"] = n_outputs

        d_nodes[n.name] = d_temp

    # return d_nodes
    return json.dumps(d_nodes, indent=4)


def overwrite_material_from_json(mat, json_in):
    d_nodes = json.loads(json_in)

    # Enable 'Use nodes':
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Remove existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create nodes and store them into dict
    new_nodes = {}
    for nk, nv in d_nodes.items():
        node = nodes.new(nv["bl_idname"])
        node.location = nv["location"]
        node.name = nv["name"]
        new_nodes[nk] = node

    # Link nodes (use only inputs data)
    for nk, nv in d_nodes.items():
        for l in nv["inputs"]:
            # name, type, value, (from_socket.name)
            if l["input_type"] != "none":
                if l["input_type"] == "node":
                    mat.node_tree.links.new(
                        new_nodes[nk].inputs[l["input_name"]],
                        new_nodes[l["from_node_name"]].outputs[l["from_socket_name"]],
                    )
                elif l["input_type"] == "float":
                    new_nodes[nk].inputs[l["input_name"]].default_value = l["value"]

        # test_group.links.new(node_add.inputs[1], node_less.outputs[0])

    return d_nodes
