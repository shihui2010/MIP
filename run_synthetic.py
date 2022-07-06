#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pprint
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modules.metrics import Metrics
from data.MixtureGaussian import Batcher
from sklearn.cluster import AgglomerativeClustering


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--n_dim', required=True, type=int)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--n_global_interest', default=1024, type=int)
parser.add_argument('--n_user_interest', default=8, type=int)
parser.add_argument('--global_query', action='store_true',
                    help='if the model uses global query')
parser.add_argument('--share_query', action='store_true',
                    help='if the model uses shared query')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()


def main():
    assert args.n_dim >= 16 or args.n_dim == 2, "undesired n_dim"
    seq_len = 20 if args.n_dim == 2 else 50
    if args.n_dim == 2:
        args.n_global_interest = 8
        args.n_user_interest = 4
        train_size = 50
        test_size = 10
        val_size = 10
        num_epochs = 500
        logdir = 'train_logs/synthetic/2d/'
    else:
        train_size = 10000
        test_size = 1000
        val_size = 1000
        num_epochs = 500
        logdir = "train_logs/synthetic/nd/"
    
    if args.global_query:
        if args.share_query:
            logdir += "share_query/"
        else:
            logdir += "global_query/"
    else:
        logdir += "self_attention/"
    
    logdir += f"{args.n_dim}D_{args.n_head}H"
    
    pprint.pprint(vars(args))
    print(f"log writes to : {logdir}")
    
    tb_logger = SummaryWriter(logdir)
    
    batcher = Batcher(n_dim=args.n_dim,
                      n_components=args.n_global_interest,
                      interests_per_user=args.n_user_interest,
                      train_size=train_size, 
                      test_size=test_size, 
                      val_size=val_size,
                      seq_len=seq_len
                      )
    
    global_query = True
    
    config = {"d_model": args.n_dim,
              "activation": None,
              "emb_dim": args.n_dim,
              "n_head": args.n_head,
              "att_on_tem": False,
              "att_on_pos": False,
              "tem_dim": 4,
              "dropout": 0.0,
              "loss_fn": "ce",
              "seq_len": seq_len,
              "loss_margin": None,
              "share_query": args.share_query,
              "lr": 5e-4}
    
    
    if args.global_query:
        from modules.named_models import GlobalQueryModel
        model = GlobalQueryModel(config)
        print("creating global query model")
        print(f"shared query: {args.share_query}")
    else:
        print("Creating Unweighted PinnerSage Model")
        from modules.named_models import StaticPinnerSagePlus
        model = StaticPinnerSagePlus(config)
    
    
    print("========model config========")
    pprint.pprint(config)
    
    
    def run_epoch(data_loader, epoch, tag, metric, tb_logger):
        train = tag == "Training"
        for step, data in enumerate(data_loader):
            seq, t, pos, neg = data
            for bid in range(seq.shape[0]):
                ward = AgglomerativeClustering(
                    n_clusters=5, linkage='ward')
                c_labels = ward.fit_predict(seq[bid].cpu().detach().numpy())
                # print(c_labels)
            # continue
    
            pos_logits, neg_logits, nll = model.supervised(
                seq, t, pos, neg, train=train)
            metric.update_states(pos_logits, neg_logits, nll)
            if step % 10 == 0:
                print(f"{tag} Epoch {epoch} | Step {step} | {metric}")
        metric.write_record(tb_logger, tag, epoch)
        print(f"[SUM {tag}] Epoch {epoch} | {metric}")
        metric.reset_states()
    
    
    train_dl = DataLoader(batcher.train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)
    
    test_dl = DataLoader(batcher.test_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=args.num_workers)
    val_dl = DataLoader(batcher.val_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers)
    
    metric = Metrics()
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"Total Params: {model.count_parameters()}")
    print('\n\n')
    
    for epoch in range(num_epochs):
        run_epoch(train_dl, epoch, "Training", metric, tb_logger)
        if epoch % 100 == 0:
            run_epoch(train_dl, epoch, "Test", metric, tb_logger)
            run_epoch(train_dl, epoch, "Val", metric, tb_logger)
            os.makedirs(logdir, exist_ok=True)
            
            torch.save(model, f"{logdir}/epoch_{epoch}")


if __name__ == "__main__":
    main()
