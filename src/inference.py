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
    "save_data_dir": "../data/middle",
    "save_model_dir": "../dumps/middle",
    "load_model_dir": "../dumps/middle/XXXX"
}

def evaluate(model, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    epoch_step = len(data_loader) #iteration
    total_step = config["epochs"] * epoch_step #itotal iteration
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6

    evaluate_results = {"data": {"id": list(), "counts": list(), "pred": list()},
        "error": {"mae": INF, "mse": INF},
        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target, reduce="none")
    elif config["reg_loss"] == "BCE": 
        reg_crit = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target, reduce="none")
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target, reduce="none")
    elif config["bp_loss"] == "BCE":
        bp_crit = lambda pred, target: F.binary_cross_entropy_with_logits(pred, target, reduce="none")
    else:
        raise NotImplementedError

    model.eval()

    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            ids, pattern, pattern_len, graph, graph_len, counts = batch
            cnt = counts.shape[0] #cnt meansing what??batch size?
            total_cnt += cnt

            evaluate_results["data"]["id"].extend(ids)
            evaluate_results["data"]["counts"].extend(counts.view(-1).tolist())

            pattern.to(device)
            graph.to(device)
            pattern_len, graph_len, counts = pattern_len.to(device), graph_len.to(device), counts.to(device)

            # d = torch.ones(counts.numel()).reshape(counts.shape)
            # d = d.to(device)
            # d = d - counts
            # label = torch.cat([d,counts],dim = 1)

            st = time.time()
            pred = model(pattern, pattern_len, graph, graph_len)
            #pred = torch.clamp(pred, min = -10, max = 10)
            et = time.time()
            evaluate_results["time"]["total"] += (et-st)
            avg_t = (et-st) / (cnt + 1e-8)
            evaluate_results["time"]["avg"].extend([avg_t]*cnt)
            evaluate_results["data"]["pred"].extend(pred.cpu().view(-1).tolist())

            # counts = counts.long()
            # counts = counts.reshape(1,-1).squeeze(dim=0)
            reg_loss = reg_crit(pred, counts)
            
            if isinstance(config["bp_loss_slp"], (int, float)):
                neg_slp = float(config["bp_loss_slp"])
            else:
                bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
                neg_slp = anneal_fn(bp_loss_slp, batch_id+epoch*epoch_step, T=total_step//4, lambda0=float(l0), lambda1=float(l1))
            bp_loss = bp_crit(pred, counts)
            
            reg_loss_item = reg_loss.mean().item()
            bp_loss_item = bp_loss.mean().item()
            total_reg_loss += reg_loss_item * cnt
            total_bp_loss += bp_loss_item * cnt
            
            #evaluate
            sigmoid = torch.nn.Sigmoid()
            res = (sigmoid(pred) > 0.5).int()
            P = counts.sum()
            N = counts.shape[0] - P
            TP = counts[res == counts].sum()
            TN = (res == counts).int().sum() - TP
            FP = N - TN
            FN = P - TP
            acc = (TP + TN) / counts.shape[0]
            precision = TP / (TP + FP)
            recall = TP / P
            F1 = 2 * precision * recall / (precision + recall)
            #print(ids)
            #print("pred: ", pred)
            #print("counts: ", counts)
            print("batch id: {:0>3d}\tacc: {:.3f}\tprecision: {:.3f}\trecall: {:.3f}\tF1: {:.3f}".format(batch_id, acc, precision, recall, F1))

            #evaluate_results["error"]["mae"] += F.l1_loss(F.relu(pred), counts, reduce="none").sum().item()
            #evaluate_results["error"]["mse"] += F.mse_loss(F.relu(pred), counts, reduce="none").sum().item()

            if writer:
                writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item, epoch*epoch_step+batch_id)
                writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch*epoch_step+batch_id)

            if logger and batch_id == epoch_step-1:
                logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}\tground: {:.3f}\tpredict: {:.3f}".format(
                    epoch, config["epochs"], data_type, batch_id, epoch_step,
                    reg_loss_item, bp_loss_item,
                    counts[0].item(), sigmoid(pred)[0].item()))
        mean_reg_loss = total_reg_loss/total_cnt
        mean_bp_loss = total_bp_loss/total_cnt
        if writer:
            writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
            writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
        if logger:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}".format(
                epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss))

        #evaluate_results["error"]["mae"] = evaluate_results["error"]["mae"] / total_cnt
        #evaluate_results["error"]["mse"] = evaluate_results["error"]["mse"] / total_cnt

    gc.collect()
    return mean_reg_loss, mean_bp_loss, evaluate_results

def graph2dglgraph(graph):
    dglgraph = dgl.DGLGraph(multigraph=True)
    dglgraph.add_nodes(graph.vcount())
    edges = graph.get_edgelist()
    dglgraph.add_edges([e[0] for e in edges], [e[1] for e in edges])
    dglgraph.readonly(True)
    return dglgraph

def load_data(pattern_dir, graph_dir, meta_dir):
    import igraph as ig
    pattern = ig.rend(pattern_dir)
    graph = ig.read(graph_dir)
    meta = dict()
    literal = dict()
    with open(os.path.join(dirpath, filename), "r") as f:
        meta = json.load(f)
    with open(os.path.join(dirpath, filename), "r") as f:
        literal = json.load(f)
    
    

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
    if all([os.path.exists(os.path.join(finetune_config["save_data_dir"],
        "%s_%s_dataset.pt" % (
            data_type, "dgl" if finetune_config["model"] in ["RGCN", "RGIN"] else "edgeseq"))) for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if finetune_config["model"] in ["RGCN", "RGIN"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                dataset = EdgeSeqDataset(list())
                dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), finetune_config["batch_size"]))
    else:
        data = load_data(finetune_config["graph_dir"], finetune_config["pattern_dir"], finetune_config["metadata_dir"], num_workers=finetune_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if finetune_config["model"] in ["RGCN", "RGIN", "RSIN"]:
                if os.path.exists(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    dataset = GraphAdjDataset(x)
                    dataset.save(os.path.join(finetune_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=GraphAdjDataset.batchify,
                    pin_memory=data_type=="train")
            else:
                if os.path.exists(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type))):
                    dataset = EdgeSeqDataset(list())
                    dataset.load(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                else:
                    dataset = EdgeSeqDataset(x)
                    dataset.save(os.path.join(finetune_config["save_data_dir"], "%s_edgeseq_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=finetune_config["batch_size"], shuffle=data_type=="train", drop_last=False)
                data_loader = DataLoader(dataset,
                    batch_sampler=sampler,
                    collate_fn=EdgeSeqDataset.batchify,
                    pin_memory=data_type=="train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info("data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader), finetune_config["batch_size"]))

    # optimizer and losses
    writer = SummaryWriter(save_model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config["lr"], weight_decay=finetune_config["weight_decay"], amsgrad=True)
    optimizer.zero_grad()
    scheduler = get_linear_schedule_with_warmup(optimizer,
        len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)

    best_reg_losses = {"train": INF, "dev": INF, "test": INF}
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}

    for epoch in range(finetune_config["epochs"]):
        for data_type, data_loader in data_loaders.items():

            if data_type == "train":
                mean_reg_loss, mean_bp_loss = train(model, optimizer, scheduler, data_type, data_loader, device,
                    finetune_config, epoch, logger=logger, writer=writer)
                torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            else:
                mean_reg_loss, mean_bp_loss, evaluate_results = evaluate(model, data_type, data_loader, device,
                    finetune_config, epoch, logger=logger, writer=writer)
                with open(os.path.join(save_model_dir, '%s%d.json' % (data_type, epoch)), "w") as f:
                    json.dump(evaluate_results, f)

            if mean_reg_loss <= best_reg_losses[data_type]:
                best_reg_losses[data_type] = mean_reg_loss
                best_reg_epochs[data_type] = epoch
                logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_reg_loss, epoch))
    for data_type in data_loaders.keys():
        logger.info("data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_reg_losses[data_type], best_reg_epochs[data_type]))
