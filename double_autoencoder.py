import visdom
from vizdoom import *
import random, time, sys

import torch
import torch.nn as nn
import numpy as np
import copy

from termcolor import colored

######################################################################

from autoencoder import AutoEncoder, ConvAutoEncoder
from world import World
from spec import *

######################################################################

log_file = None
# log_file = open('train.log', 'w')

def log_string(s, color=None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())

    if log_file is not None:
        log_file.write(t + s + '\n')
        log_file.flush()

    if color is not None:
        s = colored(s, color)

    print(t + s)
    sys.stdout.flush()

######################################################################

show_progress = True
GPU = 0
torch.cuda.device(GPU)

random.seed(0)
torch.cuda.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

######################################################################

env = "double_spec-l-h-k"
vis = visdom.Visdom(env=env, log_to_filename="log/" + env + ".log")

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

assert torch.cuda.is_available(), 'We need a GPU to run this.'

######################################################################

world = World(nbots=5)

train_images, train_actions = world.generate_batch(1)
print('Image shape', train_images.shape)

spec = spec_4
model_low = ConvAutoEncoder(spec['layer_specs_enc_low'], spec['layer_specs_dec_low'])
model_high = ConvAutoEncoder(spec['layer_specs_enc_high'], spec['layer_specs_dec_high'])

def apply_models(batch, shift=0):
    return model_low(batch[shift:]) + model_high(batch[:((-shift-1) % len(batch))+1])

log_string(str(model_low.encoder))
log_string(str(model_low.decoder))
log_string(str(model_high.encoder))
log_string(str(model_high.decoder))

embed_shape = model_low.get_embed_shape(train_images.shape[1:])
log_string('Low embedding dimension is ' + str(embed_shape[0]) + ' x ' + str(embed_shape[1]) + ' x ' + str(embed_shape[2]))
embed_shape = model_high.get_embed_shape(train_images.shape[1:])
log_string('High embedding dimension is ' + str(embed_shape[0]) + ' x ' + str(embed_shape[1]) + ' x ' + str(embed_shape[2]))

criterion = torch.nn.MSELoss()
lr = 1e-3
parameters = list(model_low.parameters()) + list(model_high.parameters())
optimizer = torch.optim.Adam(parameters, lr=lr)

model_low.cuda(GPU)
model_high.cuda(GPU)

# nb_frames, nb_epochs = 1000, 15
# nb_frames, nb_epochs = 2500, 50
nb_frames, nb_epochs = 5000, 100
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


log_string('Start training')

for e in range(nb_epochs):
    acc_train_loss = 0.0

    if show_progress and (e+1) % 5 == 0:
        # batch_images = train_images[torch.arange(100, 800, 100).long()].cuda(GPU)
        batch_images = train_images[torch.randint(0, nb_frames, (8,)).long()].cuda(GPU)

        vis.images(batch_images.cpu() * train_std + train_mu)
        result = apply_models(batch_images, shift=0).detach().cpu() * train_std + train_mu
        vis.images(result.detach().clamp(min=0, max=255))
        result = (model_high(batch_images)).detach().cpu() * train_std + train_mu
        vis.images(result.detach().clamp(min=0, max=255))
        result = (model_low(batch_images)).detach().cpu() * train_std + train_mu
        vis.images(result.detach().clamp(min=0, max=255))

    for b in range(0, train_images.size(0), batch_size):
        real_batch_size = min(batch_size, train_images.size(0) - b)
        batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)
        output = apply_models(batch_train_images, shift=shift)
        loss = criterion(output, batch_train_images[shift:])
        acc_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_test_loss = 0.0

    for b in range(0, test_images.size(0), batch_size):
        real_batch_size = min(batch_size, test_images.size(0) - b)
        batch_test_images = test_images.narrow(0, b, real_batch_size).cuda(GPU)
        output = apply_models(batch_test_images, shift=shift)
        loss = criterion(output, batch_test_images[shift:])
        acc_test_loss += loss.item()

    log_string('train_loss {:d} {:f} {:f}'.format(e, acc_train_loss, acc_test_loss))

    if best_acc_train_loss is None or acc_train_loss < best_acc_train_loss:
        best_model_state = (copy.deepcopy(model_low.state_dict()), copy.deepcopy(model_high.state_dict()))
        best_acc_train_loss = acc_train_loss
    # else:
    #     model_low.load_state_dict(best_model_state[0])
    #     model_high.load_state_dict(best_model_state[1])
    #     lr *= 0.9
    #     log_string('set_lr {:f}'.format(lr))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


######################################################################

model_low.load_state_dict(best_model_state[0])
model_high.load_state_dict(best_model_state[1])

batch_train_images = train_images[torch.randperm(train_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
result = apply_models(batch_train_images, shift=0).detach().cpu() * train_std + train_mu
vis.images(batch_train_images.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

batch_test_images = test_images[torch.randperm(test_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
result = apply_models(batch_test_images, shift=0).detach().cpu() * train_std + train_mu
vis.images(batch_test_images.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

######################################################################
