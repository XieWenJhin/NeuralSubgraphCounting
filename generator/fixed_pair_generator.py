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
        return counts, mapping
    else:
        #select corresponding vertices from isomorphisms
        if random.random() >= 0.5:
            counts = 1
            index = random.randint(0, len(subisomorphisms) - 1)
            ul = subisomorphisms[index][u]
            vl = subisomorphisms[index][v]
            mapping.append((u,ul))
            mapping.append((v,vl))
            return counts, mapping
        #select vertices from g - isomorphisms with same label
        else:
            counts = 0
            #select vertex sequence from graph
            #Warning: maybe return vertices are all in subisomorphsims!
            u_seq = graph.vs(label = u_label)
            v_seq = graph.vs(label = v_label)
            #get union of subisomorphsims
            union = set()
            for subisomorphism in subisomorphisms:
                union = union | set(subisomorphism)
            for ul in u_seq:
                if ul.index in union:
                    for vl in v_seq:
                        if vl.index in union:
                            continue
                        else:
                            mapping.append((u,ul.index))
                            mapping.append((v,vl.index))
                            return counts, mapping
                else:
                    mapping.append((u,ul.index))
                    mapping.append((v,v_seq[0].index))
                    return counts, mapping
            #if vertices are all in subisomorphsims
            counts = 1
            index = random.randint(0, len(subisomorphisms) - 1)
            ul = subisomorphisms[index][u]
            vl = subisomorphisms[index][v]
            mapping.append((u,ul))
            mapping.append((v,vl))
            return counts, mapping
                
def update_data(graph_dir, pattern_dir, metadata_dir, new_metadata_dir, num_workers=24):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    print("patterns read finished")
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    print("graphs read finished")
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)
    print("metadata read finished")
    train_data, dev_data, test_data = list(), list(), list()
    for p, pattern in patterns.items():
        os.makedirs(os.path.join(config["new_meta_dir"], p),exist_ok=True)
        if p in graphs:
            for g, graph in graphs[p].items():
                print("resolve " + str(p) + str(g))
                counts, mapping = set_fixed_vertices(pattern, graph, meta[p][g]["subisomorphisms"])
                with open(os.path.join(new_metadata_dir, p, g+".meta"), "w") as f:
                    json.dump({"counts": counts, "subisomorphisms": meta[p][g]["subisomorphisms"], "mapping": mapping}, f)


        elif len(graphs) == 1 and "raw" in graphs.keys():
            for g, graph in graphs["raw"].items():
                counts, mapping = set_fixed_vertices(pattern, graph, meta[p][g]["subisomorphisms"])
                with open(os.path.join(new_metadata_dir, p, g+".meta"), "w") as f:
                    json.dump({"counts": counts, "subisomorphisms": meta[p][g]["subisomorphisms"], "mapping": mapping}, f)
    


config = {
    "pattern_dir": "../data/small/patterns",
    "graph_dir": "../data/small/graphs",
    "meta_dir": "../data/small/metadata",
    "new_meta_dir": "../data/small/newmetadata",
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
update_data(config["graph_dir"], config["pattern_dir"], config["meta_dir"], config["new_meta_dir"])