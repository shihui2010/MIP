import argparse
import os
import pprint
import torch
from modules.metrics import Metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.Batcher import CFDataset


torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser()
# dataset configs
parser.add_argument('--data', required=True, type=str,
                    help="dataset name")

# model architecture config
parser.add_argument('--global_query', action="store_true",
                    help="run global query model")
parser.add_argument('--share_query', action="store_true",
                    help="global query shared")
parser.add_argument('--att_on_tem', action="store_true",
                    help="Feed temporal encoding to attention")
parser.add_argument('--att_on_pos', action="store_true",
                    help="Feed positional encoding to attention")
parser.add_argument('--tem_enc', default=None, type=str,
                    help="temporal encoding type:"
                         "one-hot, sine"
                         "Left None for absolute values")

# model hyper-parameters
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--maxlen', default=50, type=int,
                    help="input sequence length")
parser.add_argument('--num_heads', default=8, type=int)
parser.add_argument('--emb_size', default=32, type=int,
                    help="embedding vector size of items")
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--cluster', default='ward', type=str)
parser.add_argument('--n_clusters', default=5, type=int)
parser.add_argument('--loss_fn', default='ce', type=str,
                    help="loss function: triplet / ce")
parser.add_argument('--loss_margin', default=0.2, type=float,
                    help="margin in triplet loss")

# training configs
parser.add_argument('--logdir', default="seq_model_train_log", type=str)
parser.add_argument('--weight', action="store_true",
                    help="modeling cluster weight")
parser.add_argument('--load_from', default=None, type=str,
                    help="loading model")
parser.add_argument('--start_epoch', default=0, type=int,
                    help='starting index of epochs')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--eval_only', action="store_true",
                    help="evaluate only")
parser.add_argument('--exp_decay', action="store_true",
                    help="evaluating using exponential decay weights")
parser.add_argument('--decay_rate', type=float, default=0.01,
                    help="exponential decay rates (lambda)")
parser.add_argument('--init_from', type=str,
                    help='initialize weighted model from pretrained'
                         ' unweighted model')
args = parser.parse_args()


print("=========args==========")
pprint.pprint(vars(args))
cuda = torch.cuda.is_available()

# set up dataset
if not args.eval_only:
    # multiprocessing not available for now
    train_dataset = DataLoader(CFDataset(args.data, "train", cuda=cuda),
                               batch_size=args.batch_size,
                               shuffle=True, num_workers=0)
    test_dataset = DataLoader(CFDataset(args.data, "test", cuda=cuda),
                              batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

val_dataset = DataLoader(CFDataset(args.data, "val", cuda=cuda),
                         batch_size=args.batch_size,
                         shuffle=False, num_workers=0)

config = {"d_model": args.emb_size,
          "activation": None,
          "item_count": val_dataset.dataset.item_count,
          "emb_dim": args.emb_size,
          "n_head": args.num_heads,
          "att_on_tem": args.att_on_tem,
          "att_on_pos": args.att_on_pos,
          "pos_dim": 8,
          "tem_dim": 8,
          "tem_enc": args.tem_enc,
          "dropout": args.dropout,
          "seq_len": args.maxlen,
          "loss_fn": args.loss_fn,
          "loss_margin": args.loss_margin,
          "share_query": args.share_query,
          "lr": 1e-3}

if args.exp_decay and args.decay_rate:
    config["lambda"] = args.decay_rate

if args.global_query:
    from modules.named_models import GlobalQueryModel
    model = GlobalQueryModel(config)
else:
    if args.weight:
        if args.exp_decay:
            from modules.named_models import ExpDecayWeightModel
            model = ExpDecayWeightModel(config)
        else:
            from modules.named_models import WeightedPinnerSagePlus
            model = WeightedPinnerSagePlus(config)
            
        if args.init_from is not None:
            args.load_from = None
            model.load_unweighted(args.init_from)
    else:
        from modules.named_models import StaticPinnerSagePlus
        model = StaticPinnerSagePlus(config)


def run_epoch(dataset, model, epoch, tag):
    train = tag == "Training"
    for step, data in enumerate(dataset):
        seq, t, pos, neg = data
        pos_logits, neg_logits, nll = model.supervised(
            seq, t, pos, neg, train=train)
        metric.update_states(pos_logits, neg_logits, nll)
        if step % 50:
            continue
        print(f"{tag} Epoch {epoch} | Step {step} | {metric}")
    print(f"[SUM {tag}] Epoch {epoch} | {metric}")
    metric.write_record(tb_logger, tag, epoch)
    metric.reset_states()


def run_eval(val_dataset, model):
    print(f"Run evaluation with {args.cluster}, n_clusters={args.n_clusters}")
    for step, data in enumerate(val_dataset):
        seq, t, pos, neg = data
        pos_logits, neg_logits, nll = model.clustered_inference(
            seq, t, pos, neg, n_clusters=args.n_clusters, selection="medoid",
            method=args.cluster)

        metric.update_states(pos_logits, neg_logits, nll)
    print(f"[SUM] Eval | {metric}")
    metric.reset_states()



print("========model config========")
pprint.pprint(config)

metric = Metrics()
if cuda:
    model = model.to(torch.device('cuda'))
if args.load_from is not None:
    model = torch.load(args.load_from)
    print(f"model loaded from {args.load_from}")
print(f"Total Params: {model.count_parameters()}")
print('\n\n')

if args.eval_only:
    run_eval(val_dataset, model)
    exit()


tb_logger = SummaryWriter(args.logdir)
print("created tb_logger")


print("enter training")
for epoch in range(args.start_epoch + 1, args.num_epochs + 1):
    print(f"epoch {epoch}")
    if epoch % 2 == 0:
        run_epoch(test_dataset, model, epoch, "Test")
        run_epoch(val_dataset, model, epoch, "Val")
    if epoch % 5 == 0:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        
        torch.save(model, f"{args.logdir}/epoch_{epoch}")
        #if epoch >= :
        #    os.remove(f"{args.logdir}/epoch_{epoch - 30}")

    run_epoch(train_dataset, model, epoch, "Training")
    if metric.early_stop(epoch):
        break

