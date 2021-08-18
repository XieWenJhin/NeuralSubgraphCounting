import torch
import numpy as np
import dgl
import os
import math
import pickle
import json
import copy
import torch.utils.data as data
from collections import defaultdict, Counter
from tqdm import tqdm
from utils import get_enc_len, int2onehot, \
    batch_convert_tensor_to_tensor, batch_convert_array_to_array

INF = float("inf")

##############################################
################ Sampler Part ################
##############################################
class Sampler(data.Sampler):
    _type_map = {
        int: np.int32,
        float: np.float32}

    def __init__(self, dataset, group_by, batch_size, shuffle, drop_last):
        super(Sampler, self).__init__(dataset)
        if isinstance(group_by, str):
            group_by = [group_by]
        for attr in group_by:
            setattr(self, attr, list())
        self.data_size = len(dataset.data)
        for x in dataset.data:
            for attr in group_by:
                value = x[attr]
                if isinstance(value, dgl.DGLGraph):
                    getattr(self, attr).append(value.number_of_nodes())
                elif hasattr(value, "__len__"):
                    getattr(self, attr).append(len(value))
                else:
                    getattr(self, attr).append(value)
        self.order = copy.copy(group_by)
        self.order.append("rand")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def make_array(self):
        self.rand = np.random.rand(self.data_size).astype(np.float32)
        if self.data_size == 0:
            types = [np.float32] * len(self.order)
        else:
            types = [type(getattr(self, attr)[0]) for attr in self.order]
            types = [Sampler._type_map.get(t, t) for t in types]
        dtype = list(zip(self.order, types))
        array = np.array(
            list(zip(*[getattr(self, attr) for attr in self.order])),
            dtype=dtype)
        return array

    def __iter__(self):
        array = self.make_array()
        indices = np.argsort(array, axis=0, order=self.order)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        batch_idx = 0
        while batch_idx < len(batches)-1:
            yield batches[batch_idx]
            batch_idx += 1
        if len(batches) > 0 and (len(batches[batch_idx]) == self.batch_size or not self.drop_last):
            yield batches[batch_idx]

    def __len__(self):
        if self.drop_last:
            return math.floor(self.data_size/self.batch_size)
        else:
            return math.ceil(self.data_size/self.batch_size)


##############################################
############# EdgeSeq Data Part ##############
##############################################
class EdgeSeq:
    def __init__(self, code):
        self.u = code[:,0]
        self.v = code[:,1]
        self.ul = code[:,2]
        self.el = code[:,3]
        self.vl = code[:,4]

    def __len__(self):
        if len(self.u.shape) == 2: # single code
            return self.u.shape[0]
        else: # batch code
            return self.u.shape[0] * self.u.shape[1]

    @staticmethod
    def batch(data):
        b = EdgeSeq(torch.empty((0,5), dtype=torch.long))
        b.u = batch_convert_tensor_to_tensor([x.u for x in data])
        b.v = batch_convert_tensor_to_tensor([x.v for x in data])
        b.ul = batch_convert_tensor_to_tensor([x.ul for x in data])
        b.el = batch_convert_tensor_to_tensor([x.el for x in data])
        b.vl = batch_convert_tensor_to_tensor([x.vl for x in data])
        return b
    
    def to(self, device):
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.ul = self.ul.to(device)
        self.el = self.el.to(device)
        self.vl = self.vl.to(device)


##############################################
############# EdgeSeq Data Part ##############
##############################################
class EdgeSeqDataset(data.Dataset):
    def __init__(self, data=None):
        super(EdgeSeqDataset, self).__init__()

        if data:
            self.data = EdgeSeqDataset.preprocess_batch(data, use_tqdm=True)
        else:
            self.data = list()
        self._to_tensor()
    
    def _to_tensor(self):
        for x in self.data:
            for k in ["pattern", "graph", "subisomorphisms"]:
                if isinstance(x[k], np.ndarray):
                    x[k] = torch.from_numpy(x[k])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            torch.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = torch.load(f)
        del self.data
        self.data = data

        return self

    @staticmethod
    def graph2edgeseq(graph):
        labels = graph.vs["label"]
        graph_code = list()

        for edge in graph.es:
            v, u = edge.tuple
            graph_code.append((v, u, labels[v], edge["label"], labels[u]))
        graph_code = np.array(graph_code, dtype=np.int64)
        graph_code.view(
            [("v", "int64"), ("u", "int64"), ("vl", "int64"), ("el", "int64"), ("ul", "int64")]).sort(
                axis=0, order=["v", "u", "el"])
        return graph_code

    @staticmethod
    def preprocess(x):
        pattern_code = EdgeSeqDataset.graph2edgeseq(x["pattern"])
        graph_code = EdgeSeqDataset.graph2edgeseq(x["graph"])
        subisomorphisms = np.array(x["subisomorphisms"], dtype=np.int32).reshape(-1, x["pattern"].vcount())

        x = {
            "id": x["id"],
            "pattern": pattern_code,
            "graph": graph_code,
            "counts": x["counts"],
            "subisomorphisms": subisomorphisms}
        return x
    
    @staticmethod
    def preprocess_batch(data, use_tqdm=False):
        d = list()
        if use_tqdm:
            data = tqdm(data)
        for x in data:
            d.append(EdgeSeqDataset.preprocess(x))
        return d

    @staticmethod
    def batchify(batch):
        _id = [x["id"] for x in batch]
        pattern = EdgeSeq.batch([EdgeSeq(x["pattern"]) for x in batch])
        pattern_len = torch.tensor([x["pattern"].shape[0] for x in batch], dtype=torch.int32).view(-1, 1)
        graph = EdgeSeq.batch([EdgeSeq(x["graph"]) for x in batch])
        graph_len = torch.tensor([x["graph"].shape[0] for x in batch], dtype=torch.int32).view(-1, 1)
        counts = torch.tensor([x["counts"] for x in batch], dtype=torch.float32).view(-1, 1)
        return _id, pattern, pattern_len, graph, graph_len, counts


##############################################
######### GraphAdj Data Part ###########
##############################################
class GraphAdjDataset(data.Dataset):
    def __init__(self, data=None):
        super(GraphAdjDataset, self).__init__()

        if data:
            self.data = GraphAdjDataset.preprocess_batch(data, use_tqdm=True)
        else:
            self.data = list()
        self._to_tensor()
    
    def _to_tensor(self):
        for x in self.data:
            for k in ["pattern", "graph"]:
                y = x[k]
                for k, v in y.ndata.items():
                    if isinstance(v, np.ndarray):
                        y.ndata[k] = torch.from_numpy(v)
                for k, v in y.edata.items():
                    if isinstance(v, np.ndarray):
                        y.edata[k] = torch.from_numpy(v)
            if isinstance(x["subisomorphisms"], np.ndarray):
                x["subisomorphisms"] = torch.from_numpy(x["subisomorphisms"])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, filename):
        cache = defaultdict(list)
        for x in self.data:
            for k in list(x.keys()):
                if k.startswith("_"):
                    cache[k].append(x.pop(k))
        with open(filename, "wb") as f:
            torch.save(self.data, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        if len(cache) > 0:
            keys = cache.keys()
            for i in range(len(self.data)):
                for k in keys:
                    self.data[i][k] = cache[k][i]

    def load(self, filename):
        with open(filename, "rb") as f:
            data = torch.load(f)
        del self.data
        self.data = data

        return self

    @staticmethod
    def comp_indeg_norm(graph):
        import igraph as ig
        if isinstance(graph, ig.Graph):
            # 10x faster  
            in_deg = np.array(graph.indegree(), dtype=np.float32)
        elif isinstance(graph, dgl.DGLGraph):
            in_deg = graph.in_degrees(range(graph.number_of_nodes())).float().numpy()
        else:
            raise NotImplementedError
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        return norm

    @staticmethod
    def graph2dglgraph(graph):
        dglgraph = dgl.DGLGraph(multigraph=True)
        dglgraph.add_nodes(graph.vcount())
        edges = graph.get_edgelist()
        dglgraph.add_edges([e[0] for e in edges], [e[1] for e in edges])
        dglgraph.readonly(True)
        return dglgraph

    @staticmethod
    def extend_graph(attr_name, attr_range, graph, variable_literals, constant_literals):
        MAX_V_LABEL_VALUE = max(graph.vs["label"])
        MAX_E_LABEL_VALUE = max(graph.es["label"])
        
        #for graph, add double range vertices for x.A = y.B = c and x.A = c
        add_v_count = max(attr_range) + 1
        o_v_count = graph.vcount()
        graph.add_vertices(add_v_count)
        for i in range(add_v_count):
            graph.vs[o_v_count + i]["label"] = MAX_V_LABEL_VALUE + 1
        graph.add_vertices(add_v_count)
        for i in range(add_v_count):
            graph.vs[o_v_count + add_v_count + i]["label"] = MAX_V_LABEL_VALUE + 2 + i
        for i in range(o_v_count):
            #extend graph with variable literals
            v_l_attrs = []
            for variable_literal in variable_literals:
                u, A, v, B = variable_literal
                A, B = int(A), int(B)
                v_l_attrs.extend([A, B])
            v_l_attrs = set(v_l_attrs)
            for A in v_l_attrs:
                A_value = graph.vs[i][attr_name[A]]
                graph.add_edge(i, o_v_count + A_value)
                i_2_t = graph.get_eid(i, o_v_count + A_value)
                graph.es[i_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A
            #extend graph with constant literals
            for constant_literal in constant_literals:
                u, A, c = constant_literal
                A = int(A)
                A_value = graph.vs[i][attr_name[A]]
                graph.add_edges([(i, o_v_count + add_v_count + A_value)])
                i_2_t = graph.get_eid(i, o_v_count + add_v_count + A_value)
                graph.es[i_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A

        return graph
    @staticmethod
    def preprocess(x):
        pattern = copy.copy(x["pattern"])
        graph = copy.copy(x["graph"])
        graph_ = copy.copy(x["graph_"])
        [p_u, g_u], [p_v, g_v] = x["mapping"]
        attr_name  = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        attr_range = [8, 8, 8, 8, 8]
        
        MAX_V_LABEL_VALUE = max(graph.vs["label"])
        MAX_E_LABEL_VALUE = max(graph.es["label"])
        
        #extend pattern with variable literals
        variable_literals = x["literals"]["variable literals"]
        constant_literals = x["literals"]["constant literals"]
        for variable_literal in variable_literals:
            u, A, v, B = variable_literal
            u, A, v, B = int(u), int(A), int(v), int(B)
            #target vertex: vlabel = MAX_V_LABEL_VALUE + 1, elabel = MAX_E_LABEL_VALUE + 1 + attribute_index
            
            o_v_count = pattern.vcount()
            pattern.add_vertices(1)
            tid = o_v_count
            pattern.vs[tid]["label"] = MAX_V_LABEL_VALUE + 1
            pattern.add_edges([(u, tid),(v, tid)])
        
            u_2_t, v_2_t = pattern.get_eid(u, tid), pattern.get_eid(v, tid)
            pattern.es[u_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A
            pattern.es[v_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + B
        
        #extend pattern with constant literals
        for constant_literal in constant_literals:
            u, A, c = constant_literal
            u, A, c = int(u), int(A), int(c)
            #target vertex: vlabel = MAX_V_LABEL_VALUE + 2 + attribute_value, elabel = MAX_E_LABEL_VALUE + 1 + attribute_index
            o_v_count = pattern.vcount()
            pattern.add_vertices(1)
            tid = o_v_count
            pattern.vs[tid]["label"] = MAX_V_LABEL_VALUE + 2 + c
            pattern.add_edges([(u, tid)])
            u_2_t = pattern.get_eid(u, tid)
            pattern.es[u_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A

        pattern_dglgraph = GraphAdjDataset.graph2dglgraph(pattern)
        pattern_dglgraph.ndata["indeg"] = np.array(pattern.indegree(), dtype=np.float32)
        pattern_dglgraph.ndata["label"] = np.array(pattern.vs["label"], dtype=np.int64)
        pattern_dglgraph.ndata["id"] = np.arange(0, pattern.vcount(), dtype=np.int64)
        pattern_dglgraph.edata["label"] = np.array(pattern.es["label"], dtype=np.int64)
        #mark fixed vertices
        k = 8
        pattern_weights = torch.zeros(pattern_dglgraph.number_of_nodes(), k)
        pattern_weights[p_u,] = 1
        pattern_weights[p_v,] = 1
        pattern_dglgraph.ndata["w"] = pattern_weights

        #exntend graph and graph_
        graph = extend_graph(attr_name, attr_range, graph, variable_literals, constant_literals)
        graph_ = extend_graph(attr_name, attr_range, graph, variable_literals, constant_literals)

        graph_dglgraph = GraphAdjDataset.graph2dglgraph(graph)
        graph_dglgraph.ndata["indeg"] = np.array(graph.indegree(), dtype=np.float32)
        graph_dglgraph.ndata["label"] = np.array(graph.vs["label"], dtype=np.int64)
        graph_dglgraph.ndata["id"] = np.arange(0, graph.vcount(), dtype=np.int64)
        graph_dglgraph.edata["label"] = np.array(graph.es["label"], dtype=np.int64)
        #mark fixed vertices
        graph_weights = torch.zeros(graph_dglgraph.number_of_nodes(), 8)
        graph_weights[g_u,] = 1
        graph_weights[g_v,] = 1
        graph_dglgraph.ndata["w"] = graph_weights
        subisomorphisms = np.array(x["subisomorphisms"], dtype=np.int32).reshape(-1, x["pattern"].vcount())

        graph_dglgraph_ = GraphAdjDataset.graph2dglgraph(graph_)
        graph_dglgraph_.ndata["indeg"] = np.array(graph_.indegree(), dtype=np.float32)
        graph_dglgraph_.ndata["label"] = np.array(graph_.vs["label"], dtype=np.int64)
        graph_dglgraph_.ndata["id"] = np.arange(0, graph_.vcount(), dtype=np.int64)
        graph_dglgraph_.edata["label"] = np.array(graph_.es["label"], dtype=np.int64)
        #mark fixed vertices
        graph_weights = torch.zeros(graph_dglgraph_.number_of_nodes(), 8)
        graph_weights[g_u,] = 1
        graph_weights[g_v,] = 1
        graph_dglgraph_.ndata["w"] = graph_weights

        x = {
            "id": x["id"],
            "pattern": pattern_dglgraph,
            "graph": graph_dglgraph,
            "graph_": graph_dglgraph_,
            "counts": x["counts"],
            "counts_": x["counts_"],
            "subisomorphisms": subisomorphisms}
        return x

    @staticmethod
    def preprocess_batch(data, use_tqdm=False):
        d = list()
        if use_tqdm:
            data = tqdm(data)
        for x in data:
            d.append(GraphAdjDataset.preprocess(x))
        return d

    @staticmethod
    def batchify(batch):
        _id = [x["id"] for x in batch]
        pattern = dgl.batch([x["pattern"] for x in batch])
        pattern_len = torch.tensor([x["pattern"].number_of_nodes() for x in batch], dtype=torch.int32).view(-1, 1)
        graph = dgl.batch([x["graph"] for x in batch])
        graph_len = torch.tensor([x["graph"].number_of_nodes() for x in batch], dtype=torch.int32).view(-1, 1)
        counts = torch.tensor([x["counts"] for x in batch], dtype=torch.float32).view(-1, 1)
        return _id, pattern, pattern_len, graph, graph_len, counts
            
