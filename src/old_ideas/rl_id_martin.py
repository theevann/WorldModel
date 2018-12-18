#!/usr/bin/env python3

import visdom
from vizdoom import *
import random, time, sys

import torch
import numpy as np
import copy
from tqdm import tqdm

from torch import Tensor, nn

from termcolor import colored

######################################################################

from autoencoder import AutoEncoder, ConvAutoEncoder, ConvAutoEncoderDense
from world import World

######################################################################

# log_file = None
log_file = open('train.log', 'w')

def log_string(s, color = None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + '\n')
        log_file.flush()

    if color is not None:
        s = colored(s, color)

    print(t + s)
    sys.stdout.flush()

######################################################################

GPU = 0
torch.cuda.device(GPU)

random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

######################################################################

env = "Loss_Xt--Xt-1"
env = "main"
vis = visdom.Visdom(env=env, log_to_filename="log/" + env + ".log")

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

assert torch.cuda.is_available(), 'We need a GPU to run this.'

######################################################################

world = World(5)

# FOR NORMAL CONVAUTOENCODER
layer_specs_enc = [
    {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
    {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
    {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
]

layer_specs_dec = [
    {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
    {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
    {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
]
#
# layer_specs_enc = [
#     {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
#     {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
#     {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
#     {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
# ]
#
# layer_specs_dec = [
#     {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4, 'stride': 2},
#     {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (0, 1)},
#     {'in_channels':  64, 'out_channels':  32, 'kernel_size': 6, 'stride': 3, 'output_padding': (2, 1)},
#     {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
# ]

# FOR CONVAUTOENCODER WITH DENSE
# layer_specs_enc = [
#     {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
#     {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
#     {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
#     # {'in_channels': 128, 'out_channels': 256, 'kernel_size': 4, 'stride': 2},
# ]
#
# layer_specs_dec = [
#     {'in_channels': 0, 'out_channels': 256, 'kernel_size': (3, 4), 'stride': 3},
#     {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
#     {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
#     {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
#     {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
# ]

layer_specs_dense = [1000, 100, 1000]

train_images, train_actions = world.generate_batch(1)
image_shape = train_images.shape[1:]

print(train_images.shape)
# model = ConvAutoEncoderDense(layer_specs_enc, layer_specs_dec, layer_specs_dense, image_shape)
model = ConvAutoEncoder(layer_specs_enc, layer_specs_dec)


log_string(str(model.encoder))
# log_string(str(model.dense))
log_string(str(model.decoder))

embed_shape = model.get_embed_shape(train_images.shape[1:])
log_string('Embedding dimension is ' + str(embed_shape))

criterion = torch.nn.MSELoss()
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.cuda(GPU)

# nb_frames, nb_epochs = 1000, 15
# nb_frames, nb_epochs = 2500, 30
nb_frames, nb_epochs = 50000, 100
batch_size = 30
shift = 2

best_acc_train_loss = None

log_string('Generating %d train images' % nb_frames)
train_images, train_actions = world.generate_batch(nb_frames)
log_string('Generating %d test images' % int(nb_frames / 5))
test_images, test_actions = world.generate_batch(int(nb_frames / 5))

train_mu, train_std = train_images.mean(), train_images.std()

train_images = (train_images - train_mu) / train_std
test_images = (test_images - train_mu) / train_std

vis.images(train_images.cpu()[torch.arange(shift, 1000 + shift, 100).long()] * train_std + train_mu)

log_string('Start training')

for e in range(nb_epochs):
    acc_train_loss = 0.0

    if (e+1) % 5 == 0:
        batch_train_images_t = train_images[torch.arange(0, 1000, 100).long()].cuda(GPU)
        batch_train_images_t_p1 = train_images[torch.arange(shift, 1000 + shift, 100).long()].cuda(GPU)
        result = (batch_train_images_t + model(batch_train_images_t_p1).detach()).cpu() * train_std + train_mu
        vis.images(result.detach().clamp(min=0, max=255))

    for b in range(0, train_images.size(0), batch_size):
        real_batch_size = min(batch_size, train_images.size(0) - b)
        batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)
        output = model(batch_train_images[shift:])
        loss = 100 * criterion(output, batch_train_images[shift:] - batch_train_images[:-shift])
        acc_train_loss += loss.item() * (real_batch_size - 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_test_loss = 0.0

    for b in range(0, test_images.size(0), batch_size):
        real_batch_size = min(batch_size, test_images.size(0) - b)
        batch_test_images = test_images.narrow(0, b, real_batch_size).cuda(GPU)
        output = model(batch_test_images[shift:])
        loss = criterion(output, batch_test_images[shift:] - batch_test_images[:-shift])
        acc_test_loss += loss.item() * (real_batch_size - 1)

    log_string('Loss epoch {:d} | train {:f} | test {:f}'.format(e+1, acc_train_loss / train_images.size(0), acc_test_loss / test_images.size(0)))

    if best_acc_train_loss is None or acc_train_loss < best_acc_train_loss:
        best_model_state = copy.deepcopy(model.state_dict())
        best_acc_train_loss = acc_train_loss

######################################################################

model.load_state_dict(best_model_state)

batch_train_images_t = train_images[torch.arange(10000, 11000, 100).long()].cuda(GPU)
batch_train_images_t_p1 = train_images[torch.arange(10000 + shift, 11000 + shift, 100).long()].cuda(GPU)
result = (batch_train_images_t + model(batch_train_images_t_p1).detach()).cpu() * train_std + train_mu
vis.images(batch_train_images_t_p1.cpu() * train_std + train_mu)
vis.images(result.detach().clamp(min=0, max=255))

batch_test_images_t = test_images[torch.arange(7000, 8000, 100).long()].cuda(GPU)
batch_test_images_t_p1 = test_images[torch.arange(7000 + shift, 8000 + shift, 100).long()].cuda(GPU)
result = (batch_test_images_t + model(batch_test_images_t_p1).detach()).cpu() * train_std + train_mu
vis.images(batch_test_images_t_p1.cpu() * train_std + train_mu)
vis.images(result.detach().clamp(min=0, max=255))

batch_test_images_t = test_images[torch.arange(5000, 6000, 100).long()].cuda(GPU)
batch_test_images_t_p1 = test_images[torch.arange(5000 + shift, 6000 + shift, 100).long()].cuda(GPU)
result = (batch_test_images_t + model(batch_test_images_t_p1).detach()).cpu() * train_std + train_mu
vis.images(batch_test_images_t_p1.cpu() * train_std + train_mu)
vis.images(result.detach().clamp(min=0, max=255))


######################################################################
