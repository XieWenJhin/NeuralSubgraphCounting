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

def generate(pattern_dir, graph_dir, meta_dir):
    patterns = read_patterns_from_dir(pattern_dir, num_workers=num_workers)
    graphs = read_graphs_from_dir(graph_dir, num_workers=num_workers)
    meta = read_metadata_from_dir(metadata_dir, num_workers=num_workers)

    for p, pattern in patterns.items():
        if p in graphs:
            for g, graph in graphs[p].items():
                MAX_E_LABEL_VALUE = max(graph.es["label"])
                counts = -1
                if meta[p][g]["counts"] == 0:
                    g_v_count = graph.vcount()
                    u = random.randint(0, g_v_count - 1)
                    v = random.randint(0, g_v_count - 1)
                    # guarantee u != v
                    while u == v:
                        v = random.randint(0, g_v_count - 1)
                    graph.add_edge(u, v)
                elif random.random() > 0.5:
                    m1, m2 = meta[p][g]["mapping"]
                    p_v, g_v = m1
                    p_u, g_u = m2
                    for subisomorphism in meta[p][g]["subisomorphisms"]:
                        if g_v in subisomorphism and g_u in subisomorphism:
                            p_edges = p.get_edgelist()
                            index = random.randint(0, len(p_edges) - 1)
                            p_x, p_y = p_edges[index]
                            g_x, g_y = subisomorphism[p_x], subisomorphism[p_y] 
                            eid = graph.get_eid(g_x, g_y)
                            graph.delete_edges(eid)
                            counts = 0
                            subisomorphisms = list()  

                    
config = {
    "pattern_dir": "../data/small_multi_fixed/patterns",
    "graph_dir": "../data/small_multi_fixed/graphs",
    "meta_dir": "../data/small_multi_fixed/metadata_fixed"
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
    generate(config["pattern_dir"], config["graph_dir"], config["meta_dir"])
