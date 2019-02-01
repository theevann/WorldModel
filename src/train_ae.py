#!/usr/bin/env python3

import copy
import visdom
from vizdoom import *

import torch
from torch import nn

######################################################################

from utils import *
from autoencoder import *
from world import World

######################################################################

TEST = True
# server = "http://10.90.45.11"  # "http://iccluster139.iccluster.epfl.ch"
server = "http://localhost"  # "http://iccluster139.iccluster.epfl.ch"
env = "ae_NoDense(3_3)_10x5x5_1600d_50k"
GPU = 0

layer_specs = 10

shift = 1
nbots = 8
nb_frames = 50000
nb_epochs = 300
batch_size = 50
lr = 0.5e-3
clip = 1e-2


if TEST:
    nb_frames = 500
    nb_epochs = 5
    env = "test"


nb_frames_test = int(nb_frames / 5)


######## CUDA & SEEDS ##############################################################

assert torch.cuda.is_available(), 'We need a GPU to run this.'
torch.cuda.device(GPU)
set_seeds()

######## VISUALISATIONS ##############################################################

tensorboard = None if TEST else create_tensorboard(env)
vis = visdom.Visdom(server=server, port=8097, env=env, log_to_filename="log/" + env + ".log")
check_visdom(vis)

######## BUILD WORLD ##############################################################

world = World(nbots=nbots, skip=True)

######## MODEL DEFINITION ###############################################################

### Simple AE

if False:
    model = ConvAutoEncoder3_3(world.image_shape, dim=layer_specs).cuda(GPU)


### Double Way

if True:
    model = DoubleConvAutoEncoder(
        class_1=ConvAutoEncoder3_1,
        args_1={"img_dim": world.image_shape, "dim": layer_specs},
        class_2=ConvAutoEncoder3,
        args_2={"img_dim": world.image_shape},
        combine=combine_sum,
        shift=shift,
    ).cuda(GPU)

###

embed_shape = model.get_embed_shape()

log_string(str(model))
log_string("Images shape: %s" % str(world.image_shape))
log_string('Embedding dimension is %s' % str(embed_shape))


### BUILD TRAINING AND VALIDATION SET

log_string('Generating %d train images' % nb_frames)
train_images, train_actions = world.generate_batch(nb_frames)
log_string('Generating %d test images' % nb_frames_test)
test_images, test_actions = world.generate_batch(nb_frames_test)

train_mu, train_std = train_images.mean(), train_images.std()

train_images.add_(-train_mu).div_(train_std)   # train_images = (train_images - train_mu) / train_std
test_images.add_(-train_mu).div_(train_std)    # test_images = (test_images - train_mu) / train_std


##########################################################################


def process_epoch(images, batch_size, nb_frames, is_train):
    acc_loss = 0.0
    with torch.set_grad_enabled(is_train):

        for b in range(0, nb_frames, batch_size):
            real_batch_size = min(batch_size, nb_frames - b)
            batch_images = images[b:b + real_batch_size].cuda(GPU)
            output = model(batch_images)
            loss = criterion(output, batch_images[shift:])
            acc_loss += loss.item() * real_batch_size

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

    return acc_loss


def visualise(n_images):
    perm = torch.randint(0, nb_frames_test - shift, (n_images,), dtype=torch.long)
    all_cat = []
    for i in range(n_images):
        batch_images = test_images[perm[i]:perm[i]+shift+1].cuda(GPU)
        result = model(batch_images).detach()
        all_cat.append(torch.cat((batch_images[shift:], result), 2))
    all_cat = torch.cat(all_cat, 0).cpu() * train_std + train_mu
    vis.images(all_cat.clamp(min=0, max=255), padding=1)


######## TRAIN ##############################################################

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc_train_loss = None
best_model_state = None

log_string('Start training')

for e in range(1, nb_epochs + 1):

    model.train()
    acc_train_loss = process_epoch(train_images, nb_frames, batch_size, is_train=True)

    model.train()
    acc_test_loss = process_epoch(test_images, nb_frames_test, batch_size, is_train=False)

    log_string('Loss epoch {:d} | train {:f} | test {:f}'.format(e, acc_train_loss / nb_frames, acc_test_loss / nb_frames_test))

    # Plot train/test loss
    if tensorboard is not None:
        tensorboard.add_scalar('train-loss', acc_train_loss / nb_frames, global_step=e)
        tensorboard.add_scalar('test-loss', acc_test_loss / nb_frames_test, global_step=e)

    # Visualize training
    if e % 5 == 0 and e > 0:
        visualise(n_images=6)

    # Remember current model state if this is the best so far
    if best_acc_train_loss is None or acc_train_loss < best_acc_train_loss:
        best_model_state = copy.deepcopy(toCpu(model.state_dict()))
        best_acc_train_loss = acc_train_loss

    # Save model every 50 epoch
    if e % 50 == 0 and e > 0:
        save_model(model, env, epoch, train_mu, train_std, best_model_state, best_acc_train_loss, acc_train_loss, acc_test_loss)

save_model(model, env, nb_epochs, train_mu, train_std, best_model_state, best_acc_train_loss, acc_train_loss, acc_test_loss)


######################################################################

model.load_state_dict(best_model_state)

# batch_test_images = test_images[torch.randperm(test_images.size(0)).narrow(0, 0, 8).long()].cuda(GPU)
# result = model(batch_test_images).detach().cpu() * train_std + train_mu
# vis.images(torch.cat([batch_test_images.cpu() * train_std + train_mu, result.clamp(min=0, max=255)], 0))




### END
