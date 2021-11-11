import torch
import os
import numpy as np
import dgl
import logging
import datetime
import math
import sys
import gc
import re
import subprocess
import json
import torch.nn.functional as F
import warnings
import shutil
from functools import partial
from collections import OrderedDict
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except BaseException as e:
    from tensorboardX import SummaryWriter
from dataset import Sampler, EdgeSeqDataset, GraphAdjDataset
from utils import anneal_fn, get_enc_len, load_data, get_best_epochs, get_linear_schedule_with_warmup
from cnn import CNN
from rnn import RNN
from txl import TXL
from rgcn import RGCN
from rgin import RGIN
from train import train, evaluate
from dgl.data.utils import save_graphs, load_graphs

warnings.filterwarnings("ignore")
INF = float("inf")

finetune_config = {
    "max_npv": 16, # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 16, # max_number_pattern_edges: 8, 16, 32
    "max_npvl": 16, # max_number_pattern_vertex_labels: 8, 16, 32
    "max_npel": 16, # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 512, # max_number_graph_vertices: 64, 512,4096
    "max_nge": 2048, # max_number_graph_edges: 256, 2048, 16384
    "max_ngvl": 64, # max_number_graph_vertex_labels: 16, 64, 256
    "max_ngel": 64, # max_number_graph_edge_labels: 16, 64, 256
    
    # "base": 2,

    "gpu_id": -1,
    "num_workers": 12,
    
    "epochs": 100,
    "batch_size": 128,
    "update_every": 4, # actual batch_size = batch_size * update_every
    "print_every": 100,
    "share_emb": True, # sharing embedding requires the same vector length
    "share_arch": True, # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,
    
    "predict_net": "SumPredictNet", # MeanPredictNet, SumPredictNet, MaxPredictNet,
                                    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
                                    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
                                    # DIAMNet
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
    "predict_net_mem_init": "mean", # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,

    "reg_loss": "MSE", # MAE, MSE, SMAE
    "bp_loss": "MSE", # MAE, MSE, SMAE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",    # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01, 
                                                # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
                                                # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "max_grad_norm": 8,
    
    "pattern_dir": "../data/middle/patterns",
    "graph_dir": "../data/middle/graphs",
    "metadata_dir": "../data/middle/metadata",
    "literal_dir": "../data/greate_large/patterns/literals",
    "save_data_dir": "../data/middle",
    "save_model_dir": "../dumps/middle",
    "load_model_dir": "../dumps/middle/XXXX"
}

def inference(model, pattern, graph, subisomorphisms, device, config, logger=None, writer=None):
    
    model.eval()
    with torch.no_grad():
        pattern_len = pattern.number_of_nodes()
        graph_len = graph.number_of_nodes()
        pattern.to(device)
        graph.to(device)
        pattern_len.to(device)
        graph.to(device)
        anchors = list()
        truth = list()
        for subisomorphism in subisomorphisms:
            anchors.append([subisomorphism[0], subisomorphism[1]])
            truth.append(1)
        for subisomorphism in subisomorphism:
            anchors.append([subisomorphism[0], random.choice((graph.ndata["label"] == pattern.ndata["label"][1]).nonzero()[:,0]).item()])
            truth.append(0)
        anchors.to(device)
        truth = torch.tensor(truth, dtype=torch.int)
        truth.to(device)
        pred = model(pattern, pattern_len, graph, graph_len, anchors, inference=True)
        print(pred)


def graph2dglgraph(graph):
    dglgraph = dgl.DGLGraph(multigraph=True)
    dglgraph.add_nodes(graph.vcount())
    edges = graph.get_edgelist()
    dglgraph.add_edges([e[0] for e in edges], [e[1] for e in edges])
    dglgraph.readonly(True)
    return dglgraph

def load_data(pattern_dir, graph_dir, meta_dir, literal_dir):
    import igraph as ig
    pattern = ig.rend(pattern_dir)
    graph = ig.read(graph_dir)
    meta = dict()
    literal = dict()
    with open(os.path.join(meta_dir), "r") as f:
        meta = json.load(f)
    with open(os.path.join(literal_dir), "r") as f:
        literal = json.load(f)
    #attributes: temporary static
    attr_name  = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    attr_range = [8, 8, 8, 8, 8, 8, 8]
    MAX_V_LABEL_VALUE = max(graph.vs["label"])
    MAX_E_LABEL_VALUE = max(graph.es["label"])

    #extend pattern with variable literals
    variable_literals = literal["variable literals"]
    constant_literals = literal["constant literals"]
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
        pattern.add_edge(u, tid)
        u_2_t = pattern.get_eid(u, tid)
        pattern.es[u_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A

    pattern_dglgraph = graph2dglgraph(pattern)
    pattern_dglgraph.ndata["indeg"] = np.array(pattern.indegree(), dtype=np.float32)
    pattern_dglgraph.ndata["label"] = np.array(pattern.vs["label"], dtype=np.int64)
    pattern_dglgraph.ndata["id"] = np.arange(0, pattern.vcount(), dtype=np.int64)
    pattern_dglgraph.edata["label"] = np.array(pattern.es["label"], dtype=np.int64)
    
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
            graph.add_edge(i, o_v_count + add_v_count + A_value)
            i_2_t = graph.get_eid(i, o_v_count + add_v_count + A_value)
            graph.es[i_2_t]["label"] = MAX_E_LABEL_VALUE + 1 + A

    graph_dglgraph = graph2dglgraph(graph)
    graph_dglgraph.ndata["indeg"] = np.array(graph.indegree(), dtype=np.float32)
    graph_dglgraph.ndata["label"] = np.array(graph.vs["label"], dtype=np.int64)
    graph_dglgraph.ndata["id"] = np.arange(0, graph.vcount(), dtype=np.int64)
    graph_dglgraph.edata["label"] = np.array(graph.es["label"], dtype=np.int64)

    subisomorphisms = meta["subisomorphisms"]
    save_graphs(os.path.join(save_data_dir, "pattern_dglgraph"), pattern_dglgraph)
    save_graphs(os.path.join(save_data_dir, "graph_dglgraph"), graph_dglgraph)
    return pattern_dglgraph, graph_dglgraph, subisomorphisms
    
    

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i+1]
        
        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in finetune_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        finetune_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                finetune_config[arg] = value
        except:
            pass

    # load config
    if os.path.exists(os.path.join(finetune_config["load_model_dir"], "train_config.json")):
        with open(os.path.join(finetune_config["load_model_dir"], "train_config.json"), "r") as f:
            train_config = json.load(f)
    elif os.path.exists(os.path.join(finetune_config["load_model_dir"], "finetune_config.json")):
        with open(os.path.join(finetune_config["load_model_dir"], "finetune_config.json"), "r") as f:
            train_config = json.load(f)
    else:
        raise FileNotFoundError("finetune_config.json and train_config.json cannot be found in %s" % (os.path.join(finetune_config["load_model_dir"])))
    
    for key in train_config:
        if key not in finetune_config:
            finetune_config[key] = train_config[key]

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = "%s_%s_%s" % (finetune_config["model"], finetune_config["predict_net"], ts)
    save_model_dir = finetune_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "finetune_config.json"), "w") as f:
        json.dump(finetune_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(save_model_dir, "finetune_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device("cuda:%d" % finetune_config["gpu_id"] if finetune_config["gpu_id"] != -1 else "cpu")
    if finetune_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

    # check model
    if finetune_config["model"] not in ["CNN", "RNN", "TXL", "RGCN", "RGIN"]:
        raise NotImplementedError("Currently, the %s model is not supported" % (finetune_config["model"]))

    # reset the pattern parameters
    if finetune_config["share_emb"]:
        finetune_config["max_npv"], finetune_config["max_npvl"], finetune_config["max_npe"], finetune_config["max_npel"] = \
            finetune_config["max_ngv"], finetune_config["max_ngvl"], finetune_config["max_nge"], finetune_config["max_ngel"]

    # get the best epoch
    if os.path.exists(os.path.join(finetune_config["load_model_dir"], "finetune_log.txt")):
        best_epochs = get_best_epochs(os.path.join(finetune_config["load_model_dir"], "finetune_log.txt"))
    elif os.path.exists(os.path.join(finetune_config["load_model_dir"], "train_log.txt")):
        best_epochs = get_best_epochs(os.path.join(finetune_config["load_model_dir"], "train_log.txt"))
    else:
        raise FileNotFoundError("finetune_log.txt and train_log.txt cannot be found in %s" % (os.path.join(finetune_config["load_model_dir"])))
    logger.info("retrieve the best epoch for training set ({:0>3d}), dev set ({:0>3d}), and test set ({:0>3d})".format(
        best_epochs["train"], best_epochs["dev"], best_epochs["test"]))

    # load the model
    for key in ["dropout", "dropatt"]:
        train_config[key] = finetune_config[key]
    
    if train_config["model"] == "CNN":
        model = CNN(train_config)
    elif train_config["model"] == "RNN":
        model = RNN(train_config)
    elif train_config["model"] == "TXL":
        model = TXL(train_config)
    elif train_config["model"] == "RGCN":
        model = RGCN(train_config)
    elif train_config["model"] == "RGIN":
        model = RGIN(train_config)
    else:
        raise NotImplementedError("Currently, the %s model is not supported" % (train_config["model"]))

    model.load_state_dict(torch.load(
        os.path.join(finetune_config["load_model_dir"], "epoch%d.pt" % (best_epochs["dev"])), map_location=torch.device("cpu")))
    model.increase_net(finetune_config)
    if not all([train_config[key] == finetune_config[key] for key in [
        "max_npv", "max_npe", "max_npvl", "max_npel", "max_ngv", "max_nge", "max_ngvl", "max_ngel",
        "share_emb", "share_arch"]]):
        model.increase_input_size(finetune_config)
    if not all([train_config[key] == finetune_config[key] for key in [
        "predict_net", "predict_net_hidden_dim",
        "predict_net_num_heads", "predict_net_mem_len", "predict_net_mem_init", "predict_net_recurrent_steps"]]):
        new_predict_net = model.create_predict_net(finetune_config["predict_net"],
            pattern_dim=model.predict_net.pattern_dim, graph_dim=model.predict_net.graph_dim,
            hidden_dim=finetune_config["predict_net_hidden_dim"],
            num_heads=finetune_config["predict_net_num_heads"], recurrent_steps=finetune_config["predict_net_recurrent_steps"], 
            mem_len=finetune_config["predict_net_mem_len"], mem_init=finetune_config["predict_net_mem_init"])
        del model.predict_net
        model.predict_net = new_predict_net
    model = model.to(device)
    torch.cuda.empty_cache()
    logger.info("load the model based on the dev set (epoch: {:0>3d})".format(best_epochs["dev"]))
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    os.makedirs(finetune_config["save_data_dir"], exist_ok=True)
    data_loaders = OrderedDict({"train": None, "dev": None, "test": None})
    pattern_dglgraph = None
    graph_dglgraph = None
    subisomorphisms = list()
    if(os.path.exists(os.path.join(finetune_config["save_data_dir"], "pattern_dglgraph")) and
        os.path.exists(os.path.join(finetune_config["save_data_dir"], "graph_dglgraph"))):
        pattern_dglgraph = load_graphs(os.path.join(finetune_config["save_data_dir"], "pattern_dglgraph"))[0]
        graph_dglgraph = load_graphs(os.path.join(finetune_config["save_data_dir"], "graph_dglgraph"))[0]
        with open(os.path.join(meta_dir), "r") as f:
            meta = json.load(f)
            subisomorphisms = meta["subisomorphisms"]
    else:
        pattern_dglgraph, graph_dglgraph, subisomorphisms = load_data(finetune_config["pattern_dir"], finetune_config["graph_dir"], finetune_config["metadata_dir"], finetune_config["literal_dir"], finetune_config["save_data_dir"])
    
    writer = SummaryWriter(save_model_dir)

    #inference
    inference(model, pattern_dglgraph, graph_dglgraph, subisomorphisms, device, finetune_config)