import torch
import numpy as np
from tensorboardX import SummaryWriter

import os
import random
import time

######################################################################

def create_tensorboard(env):
    date = datetime.datetime.now().strftime("%d%m%y")
    tensorboard_dir = os.path.join("tb_logs", env + "_" + date)
    assert not os.path.exists(tensorboard_dir), "Tensorboard directory already exists !\nrm -r "+ tensorboard_dir
    log_string("Tensorboard experiment directory: {}".format(tensorboard_dir))
    return SummaryWriter(log_dir=tensorboard_dir)


def check_visdom(vis):
    if vis.check_connection():
        log_string('Visdom server ' + vis.server + ':' + str(vis.port))
    else:
        log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
        exit(1)


def set_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#####################################################################

def toCpu(array):
    return {k: v.cpu() for k, v in array.items()}


def save_model(model, env, epoch, train_mu, train_std, best_model_state, best_acc_train_loss, acc_train_loss, acc_test_loss):
    date = time.strftime("%d%m%y_%H%M", time.localtime())
    torch.save({
        "best": best_model_state,
        "last": toCpu(model.state_dict()),
        "args": model.args,
        "std": train_std,
        "mu": train_mu,
        "epoch": epoch,
        "best_acc_train_loss": best_acc_train_loss,
        "acc_train_loss": acc_train_loss,
        "acc_test_loss": acc_test_loss,
    }, "./trained_models/%s_ep%d_%s" % (env, epoch, date))


######################################################################

def log_string(s, color=None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())
    if color is not None:
        s = colored(s, color)
    print(t + s)
