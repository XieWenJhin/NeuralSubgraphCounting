import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import re
import os
import json
from torch.optim.lr_scheduler import LambdaLR
from collections import OrderedDict
from multiprocessing import Pool
from tqdm import tqdm
import random
import sys

def _get_subdirs(dirpath, leaf_only=True):
    subdirs = list()
    is_leaf = True
    for filename in os.listdir(dirpath):
        if os.path.isdir(os.path.join(dirpath, filename)):
            is_leaf = False
            subdirs.extend(_get_subdirs(os.path.join(dirpath, filename), leaf_only=leaf_only))
    if not leaf_only or is_leaf:
        subdirs.append(dirpath)
    return subdirs

def _read_graphs_from_dir(dirpath):
    import igraph as ig
    graphs = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".gml":
                continue
            try:
                graph = ig.read(os.path.join(dirpath, filename))
                graph.vs["label"] = [int(x) for x in graph.vs["label"]]
                graph.es["label"] = [int(x) for x in graph.es["label"]]
                graph.es["key"] = [int(x) for x in graph.es["key"]]
                graphs[names[0]] = graph
            except BaseException as e:
                print(e)
                break
    return graphs

def read_graphs_from_dir(dirpath, num_workers=4):
    graphs = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            graphs[os.path.basename(subdir)] = x
    return graphs

def read_patterns_from_dir(dirpath, num_workers=4):
    patterns = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_graphs_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            patterns.update(x)
    return patterns

def _read_metadata_from_dir(dirpath):
    meta = dict()
    for filename in os.listdir(dirpath):
        if not os.path.isdir(os.path.join(dirpath, filename)):
            names = os.path.splitext(os.path.basename(filename))
            if names[1] != ".meta":
                continue
            try:
                with open(os.path.join(dirpath, filename), "r") as f:
                    meta[names[0]] = json.load(f)
            except BaseException as e:
                print(e)
    return meta

def read_metadata_from_dir(dirpath, num_workers=4):
    meta = dict()
    subdirs = _get_subdirs(dirpath)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        results = list()
        for subdir in subdirs:
            results.append((subdir, pool.apply_async(_read_metadata_from_dir, args=(subdir, ))))
        pool.close()
        
        for subdir, x in tqdm(results):
            x = x.get()
            meta[os.path.basename(subdir)] = x
    return meta

def set_fixed_vertices(pattern, graph, subisomorphisms):
    #fix two vertices mapping and update ground truth
    mapping = list()
    counts = -1
    fixed_isos = list()
    #random select two vertices from pattern
    u = random.randint(0,len(pattern.vs) - 1)
    v = random.randint(0,len(pattern.vs) - 1)
    while u == v:
        v = random.randint(0, len(pattern.vs) - 1)
    u_label = int(pattern.vs[u]["label"])
    v_label = int(pattern.vs[v]["label"])
    
    #none match
    if len(subisomorphisms) == 0:
        counts = 0
        #select vertices from graph with same label
        #Warning: find() maybe return error!
        ul = graph.vs.find(label = u_label)
        vl = graph.vs.find(label = v_label)
        mapping.append((u,ul.index))
        mapping.append((v,vl.index))
        return counts, mapping, fixed_isos
    else:
        #select corresponding vertices from isomorphisms
        if random.random() >= 0.5:
            counts = 1
            index = random.randint(0, len(subisomorphisms) - 1)
            ul = subisomorphisms[index][u]
            vl = subisomorphisms[index][v]
            #ul vl maybe appeare in multi matches
            for subisomorphism in subisomorphisms:
                if ul in subisomorphism and vl in subisomorphism:
                    fixed_isos.append(subisomorphism)
            mapping.append((u,ul))
            mapping.append((v,vl))
            return counts, mapping, fixed_isos 
        #select vertices from g - isomorphisms with same label
        else:
            counts = 0
            #select vertex sequence from graph
            #Warning: maybe return vertices are all in subisomorphisms!
            u_seq = graph.vs(label = u_label)
            v_seq = graph.vs(label = v_label)
            #get union of subisomorphisms
            union = set()
            for subisomorphism in subisomorphisms:
                union = union | set(subisomorphism)
            #actually even if ul and vl both in union, if they are not in same iso, counts can be 0 
            for ul in u_seq:
                if ul.index in union:
                    for vl in v_seq:
                        if vl.index in union:
                            continue
                        else:
                            mapping.append((u,ul.index))
                            mapping.append((v,vl.index))
                            return counts, mapping, fixed_isos
                else:
                    mapping.append((u,ul.index))
                    mapping.append((v,v_seq[0].index))
                    return counts, mapping, fixed_isos
            #if vertices are all in subisomorphisms
            counts = 1
            index = random.randint(0, len(subisomorphisms) - 1)
            ul = subisomorphisms[index][u]
            vl = subisomorphisms[index][v]
            #ul vl maybe appeare in multi matches
            for subisomorphism in subisomorphisms:
                if ul in subisomorphism and vl in subisomorphism:
                    fixed_isos.append(subisomorphism)
            mapping.append((u,ul))
            mapping.append((v,vl))
            return counts, mapping, fixed_isos

def mono_generate(pattern, graph, subisomorphisms, mapping):
    MAX_E_LABEL_VALUE = max(graph.es["label"])
    counts = -1
    if len(subisomorphisms) == 0:
        g_v_count = graph.vcount()
        u = random.randint(0, g_v_count - 1)
        v = random.randint(0, g_v_count - 1)
        # guarantee u != v
        while u == v:
            v = random.randint(0, g_v_count - 1)
        graph.add_edge(u, v)
        eid = graph.get_eid(u, v)
        graph.es[eid]["label"] = random.randint(0, MAX_E_LABEL_VALUE)
        counts = 0
    #Warning: if delete edge lead to disconnected?
    elif random.random() > 0.5:
        m1, m2 = meta[p][g]["mapping"]
        p_v, g_v = m1
        p_u, g_u = m2
        subisomorphism = subisomorphisms[0]
        p_edges = p.get_edgelist()
        index = random.randint(0, len(p_edges) - 1)
        p_x, p_y = p_edges[index]
        g_x, g_y = subisomorphism[p_x], subisomorphism[p_y] 
        eid = graph.get_eid(g_x, g_y)
        graph.delete_edges(eid) 
        counts = 0
        subisomorphisms = list()  
    else:
        g_v_count = graph.vcount()
        u = random.randint(0, g_v_count - 1)
        v = random.randint(0, g_v_count - 1)
        # guarantee u != v
        while u == v:
            v = random.randint(0, g_v_count - 1)
        graph.add_edge(u, v)
        eid = graph.get_eid(u, v)
        graph.es[eid]["label"] = random.randint(0, MAX_E_LABEL_VALUE)
        counts = 1
    return graph, counts, subisomorphisms
    
def process(p, g, new_pattern_dir, new_graph_dir, new_metadata_dir, pattern, graph, meta, attr_num, attr_range, constants, variables):
     #generate attributes for pattern and graph
    counts = 0
    for i in range(attr_num):
        attr = list()
        for j in range(pattern.vcount()):
            attr.append(random.randint(0,attr_range[i]))
        pattern.vs[attr_name[i]] = attr

    for i in range(attr_num):
        attr = list()
        for j in range(graph.vcount()):
            attr.append(random.randint(0,attr_range[i]))
        graph.vs[attr_name[i]] = attr
    
    #generator variable literals
    variable_literals = list()
    #chose two vertices of pattern
    left_attrs = [int(x) for x in range(attr_num)]
    for i in range(variables):
        #chose two vertices of pattern
        x = random.randint(0, pattern.vcount() - 1)
        y = random.randint(0, pattern.vcount() - 1)
        # guarantee x != y
        while y == x:
            y = random.randint(0, pattern.vcount() - 1)
        #chose two attributes, allow A == B
        #Warning: exactly if x or y changed, A and B can be same as other variable literals
        A = random.choice(left_attrs)
        B = random.choice(left_attrs)
        #each variable literal contain different attributes
        left_attrs.remove(A)
        if A != B:
            left_attrs.remove(B)
        variable_literals.append([x, A, y, B])
        #literal is x.A == y.B
        pattern.vs[x][attr_name[A]] = pattern.vs[y][attr_name[B]] = random.randint(0, min(attr_range[A], attr_range[B]))

    #generator constant literals
    constant_literals = list()
    for i in range(constants):
        #chose a vertex of pattern
        #Warning:attr shoude be different for each constant literal
        x = random.randint(0, pattern.vcount() - 1)
        A = random.randint(0, attr_num - 1)
        c = pattern.vs[x][attr_name[A]]
        constant_literals.append([x, A, c])

    #if has match, process matchs and compute counts
    literal_isos = list() 
    if meta["counts"] != 0:
        for subisomorphism in meta["subisomorphisms"]:
            satisfied = True
            #check variable literals
            for literal in variable_literals:
                x, A, y, B = literal
                x_2_g = subisomorphism[x]
                y_2_g = subisomorphism[y]
                if graph.vs[x_2_g][attr_name[A]] != graph.vs[y_2_g][attr_name[B]]:
                    satisfied = False
                    break
            #check constant literals
            for literal in constant_literals:
                x, A, c = literal
                if graph.vs[x_2_g][attr_name[A]] != c:
                    satisfied = False
                    break
            #if match already satisfied literals, add counts;else revise match to satisfied under a probability 
            if satisfied:
                counts += 1
                literal_isos.append(subisomorphism)
            elif random.random() > 0.5:
                for literal in variable_literals:
                    x, A, y, B = literal
                    x_2_g = subisomorphism[x]
                    y_2_g = subisomorphism[y]
                    graph.vs[x_2_g][attr_name[A]] = graph.vs[y_2_g][attr_name[B]] = random.randint(0, min(attr_range[A], attr_range[B]))
                    #resolve literals confict
                    for l in constant_literals:
                        if (x == l[0] and A == l[1]) or (y == l[0] and B == l[1]):
                            graph.vs[x_2_g][attr_name[A]] = graph.vs[y_2_g][attr_name[B]] = c
                            break

                for literal in constant_literals:
                    x, A, c = literal
                    x_2_g = subisomorphism[x]
                    graph.vs[x_2_g][attr_name[A]] = c
                counts += 1
                literal_isos.append(subisomorphism)
    #fixed a pair of vertices, and judge if has matches
    fixed_counts, mapping, fixed_isos = set_fixed_vertices(pattern, graph, literal_isos)
    #write to new data file
    os.makedirs(os.path.join(new_graph_dir, p), exist_ok=True)
    os.makedirs(os.path.join(new_metadata_dir, p), exist_ok=True)
    os.makedirs(os.path.join(new_metadata_dir + "_fixed", p), exist_ok=True)
    
    pattern.write(os.path.join(new_pattern_dir, p + ".gml"))
    graph.write(os.path.join(new_graph_dir, p, g + ".gml"))
    
    with open(os.path.join(new_metadata_dir,p ,g + ".meta"), "w") as f:
        json.dump({"counts": counts, "subisomorphisms": literal_isos}, f)
    
    with open(os.path.join(new_metadata_dir + "_fixed",p ,g +'.meta'), "w") as f:
        json.dump({"counts": fixed_counts, "subisomorphisms": fixed_isos, "mapping": mapping}, f)
    
    with open(os.path.join(new_pattern_dir, p + ".literals"), "w") as f:
        json.dump({"constant literals": constant_literals, "variable literals": variable_literals} , f)
    return 1

attr_name = ["A", "B", "C", "D", "E"]
def generate_attributes(graph_dir, pattern_dir, metadata_dir, new_pattern_dir, new_graph_dir, new_metadata_dir, attr_num, attr_range, constants, variables, num_workers=32):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)

    os.makedirs(new_pattern_dir, exist_ok=True)
    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
        for p, pattern in patterns.items():
            if p in graphs:
                for g, graph in graphs[p].items():
                    with Pool(num_workers if num_workers > 0 else os.cpu_count()) as pool:
                        state = pool.apply_async(process, args=(p, g, new_pattern_dir, new_graph_dir, new_metadata_dir, pattern, graph,meta[p][g], attr_num, attr_range, constants, variables))
                        state.get()
        pool.close()

               


config = {
    "pattern_dir": "../data/small/patterns",
    "graph_dir": "../data/small/graphs",
    "meta_dir": "../data/small/metadata",
    "new_pattern_dir": "../data/small_stage_1/patterns",
    "new_graph_dir": "../data/small_stage_1/graphs",
    "new_meta_dir": "../data/small_stage_1/metadata",
    "attr_num": 5,
    "attr_range": "8,8,8,8,8",
    "constants": 1,
    "variables": 2
}

if __name__ == "__main__":
    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i+1]
        
        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in config:
            print("Warning: %s is not surported now." % (arg))
            continue
        config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                config[arg] = value
        except:
            pass
    config["attr_num"] = int(config["attr_num"])
    config["attr_range"] = [int(x) for x in config["attr_range"].split(",")]
    config["constants"] = int(config["constants"])
    config["variables"] = int(config["variables"])
    generate_attributes(config["graph_dir"], config["pattern_dir"], config["meta_dir"], 
        config["new_pattern_dir"], config["new_graph_dir"], config["new_meta_dir"],
        config["attr_num"], config["attr_range"], config["constants"], config["variables"])