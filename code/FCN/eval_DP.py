import os
import yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np

import torchvision
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer

from tensorboardX import SummaryWriter

defaultParams = {
    'activ': 'tanh',    # 'tanh' or 'selu'
    #'plastsize': 200,
    'rule': 'clip',     # 'hebb' or 'oja' or 'clip'
    'alpha': 'free',    # 'free' of 'yoked' (if the latter, alpha is a single scalar learned parameter, shared across all connection)
    'steplr': 1e6,      # How often should we change the learning rate?
    'nbclasses': 151,
    'gamma': .666,      # The annealing factor of learning rate decay for Adam
    'flare': 0,         # Whether or not the ConvNet has more features in higher channels
    'nbshots': 1,       # Number of 'shots' in the few-shots learning
    'prestime': 1,
    'nbf' : 64,         # Number of features. 128 is better (unsurprisingly) but we keep 64 for fair comparison with other reports
    'prestimetest': 1,
    'ipd': 0,           # Inter-presentation delay 
    'imgsize': 31,    
    'nbiter': 5000000,  
    'lr': 3e-5, 
    'test_every': 500,
    'save_every': 10000,
    'rngseed':0
}

def decode_segmap(temp, n_classes=151):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = 10 * (l % 10)
        g[temp == l] = l
        b[temp == l] = 0

    rgb = np.zeros((temp.shape[1], temp.shape[2], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb

def save_presentations(pres_results, num_pres, logdir):
    fig, axs = plt.subplots(nrows=num_pres, ncols=3, figsize=(16,16*num_pres//3))
    plt.subplots_adjust(top=0.93)
    print(len(pres_results))

    for idx, ax in enumerate(axs.flatten()):
       ax.imshow(pres_results[idx])

    plt.savefig(logdir + "/pres_results.png")

def eval(cfg, writer, logger, logdir):

    # Setup seeds
    #torch.manual_seed(cfg.get("seed", 1337))
    #torch.cuda.manual_seed(cfg.get("seed", 1337))
    #np.random.seed(cfg.get("seed", 1337))
    #random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataloader_type"])

    data_root = cfg["data"]["data_root"]
    presentation_root = cfg["data"]["presentation_root"]


    v_loader = data_loader(
        data_root= data_root,
        presentation_root= presentation_root,
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        augmentations=data_aug,
        test_mode=True
    )

    n_classes = v_loader.n_classes
  

    valloader = data.DataLoader(
        v_loader, 
        batch_size=cfg["training"]["batch_size"], 
        num_workers=cfg["training"]["n_workers"],
        shuffle=False
    )

    # Setup Metrics
    # running_metrics_train = runningScore(n_classes)
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model = get_model(cfg["model"], n_classes, defaultParams).to(device)

    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg) 
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    # train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = 0
    pres_results = []               # a final list of all <image, label, output> of all presentations

    while i < cfg["training"]["num_presentations"]:

        #                 #
        #  TESTING PHASE  #
        #                 #
        i += 1
        
        training_state_dict = model.state_dict()
        hebb = model.initialZeroHebb().to(device)
        valloader.dataset.random_select()
        start_ts = time.time()

        for idx, (images_val, labels_val) in enumerate(valloader, 1):  # get a single test presentation

            images_val = images_val.to(device)
            labels_val = labels_val.to(device)

            if idx <= 5:
                model.eval()
                with torch.no_grad():
                    outputs, hebb = model(images_val, labels_val, hebb, device, test_mode=False)
            else:
                model.train()
                optimizer.zero_grad()
                outputs, hebb = model(images_val, labels_val, hebb, device, test_mode=True)
                loss = loss_fn(input=outputs, target=labels_val)
                loss.backward()
                optimizer.step()

                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss_meter.update(loss.item())

                # Turning the image, label, and output into plottable formats
                img = torchvision.utils.make_grid(images_val.cpu()).numpy()
                img = np.transpose(img, (1, 2, 0))
                img = img[:, :, ::-1]
                print("img.shape",img.shape)
                print("gt.shape and type",gt.shape, gt.dtype)
                print("pred.shape and type",pred.shape, pred.dtype)

                cla, cnt = np.unique(pred, return_counts=True)
                print("Unique classes predicted = {}, counts = {}".format(cla, cnt))
                pres_results.append(img)
                pres_results.append(decode_segmap(gt))
                pres_results.append(decode_segmap(pred.astype('int64')))



        time_meter.update(time.time() - start_ts)            # -> time taken per presentation
        model.load_state_dict(training_state_dict)                          # revert back to training parameters

        # Display presentations stats
        fmt_str = "Pres [{:d}/{:d}]  Loss: {:.4f}  Time/Pres: {:.4f}"
        print_str = fmt_str.format(
            i + 1, 
            cfg["training"]["num_presentations"],
            loss.item(),
            time_meter.avg / cfg["training"]["batch_size"],
        )
        print(print_str)
        logger.info(print_str)
        writer.add_scalar("loss/test_loss", loss.item(), i + 1)
        time_meter.reset()

        writer.add_scalar("loss/val_loss", val_loss_meter.avg, i + 1)
        logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

        # Display presentation metrics
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print(k, v)
            logger.info("{}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, i + 1)

        #for k, v in class_iou.items():
        #    logger.info("{}: {}".format(k, v))
        #    writer.add_scalar("val_metrics/cls_{}".format(k), v, i + 1)

        val_loss_meter.reset()
        running_metrics_val.reset()

    # save presentations to a png image file
    save_presentations(pres_results=pres_results, num_pres=cfg["training"]["num_presentations"], logdir=logdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/oneShot_dp_fcn8s_ade20k.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    eval(cfg, writer, logger, logdir)
