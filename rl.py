#!/usr/bin/env python3

# @XREMOTE_HOST: elk.fleuret.org
# @XREMOTE_EXEC: ${HOME}/conda/bin/python
# @XREMOTE_PRE: ln -s ${HOME}/misc/git/ViZDoom/bin/freedoom2.wad
# @XREMOTE_PRE: ln -s ${HOME}/misc/git/ViZDoom/bin/vizdoom
# @XREMOTE_PRE: ln -s ${HOME}/misc/git/ViZDoom/bin/python3.6/vizdoom.cpython-36m-x86_64-linux-gnu.so
# @XREMOTE_SEND: autoencoder.py
# @XREMOTE_GET: *.log

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

class World:
    def __init__(self):
        self.game = DoomGame()

        self.game.set_window_visible(False)

        # self.game.set_doom_scenario_path("freedoom2.wad")
        self.game.set_doom_map("map04")

        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.CRCGCB) # This gives 3xHxW tensor
        # self.game.set_depth_buffer_enabled(True)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_messages(False)
        self.game.set_render_screen_flashes(False)
        self.game.set_render_weapon(False)
        self.game.set_render_effects_sprites(False)

        self.game.set_mode(Mode.PLAYER)

        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.MOVE_FORWARD)

        self.game.set_seed(0)  # DETERMINISTIC GAME !
        self.game.init()

        self.game.new_episode()

        self.actions = [
            ('turn_left',    [ True, False, False ]),
            ('turn_right',   [ False, True, False ]),
            ('move_forward', [ False, False, True ]),
            ('stay_put',     [ False, False, False]),
        ]

        for i in range(5):
            self.game.send_game_command("addbot")

    def generate_batch(self, nb):
        batch_images = Tensor(nb, self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        batch_actions = torch.LongTensor(nb)

        for t in tqdm(range(nb)):
            if t == 0 or random.random() < 0.1:
                a = random.randrange(len(self.actions))
            reward = self.game.make_action(self.actions[a][1])

            state = self.game.get_state()
            # import ipdb;ipdb.set_trace()

            if self.game.is_episode_finished() or self.game.is_player_dead():
                self.game.new_episode()
                state = self.game.get_state()

            # misc = state.game_variables
            frame = torch.from_numpy(state.screen_buffer).float()
            batch_images[t] = frame
            batch_actions[t] = a

        return batch_images, batch_actions

######################################################################

vis = visdom.Visdom()

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

assert torch.cuda.is_available(), 'We need a GPU to run this.'

######################################################################

world = World()

# FOR NORMAL CONVAUTOENCODER
# layer_specs_enc = [
#     {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
#     {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
#     {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
#     {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
# ]
#
# layer_specs_dec = [
#     {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
#     {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
#     {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
#     {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
# ]

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
layer_specs_enc = [
    {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
    {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
    {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
    # {'in_channels': 128, 'out_channels': 256, 'kernel_size': 4, 'stride': 2},
]

layer_specs_dec = [
    {'in_channels': 0, 'out_channels': 256, 'kernel_size': (3, 4), 'stride': 3},
    {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
    {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
    {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
    {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
]

layer_specs_dense = [1000, 100, 1000]

train_images, train_actions = world.generate_batch(1)
image_shape = train_images.shape[1:]

print(train_images.shape)
model = ConvAutoEncoderDense(layer_specs_enc, layer_specs_dec, layer_specs_dense, image_shape)
# model = ConvAutoEncoder(layer_specs_enc, layer_specs_dec)


log_string(str(model.encoder))
log_string(str(model.dense))
log_string(str(model.decoder))

embed_shape = model.get_embed_shape(train_images.shape[1:])
log_string('Embedding dimension is ' + str(embed_shape))

criterion = torch.nn.MSELoss()
lr = .5e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.cuda(GPU)

# nb_frames, nb_epochs = 1000, 15
# nb_frames, nb_epochs = 2500, 300
nb_frames, nb_epochs = 50000, 100
batch_size = 30

best_acc_train_loss = None

log_string('Generating %d train images' % nb_frames)
train_images, train_actions = world.generate_batch(nb_frames)
log_string('Generating %d test images' % int(nb_frames / 5))
test_images, test_actions = world.generate_batch(int(nb_frames / 5))

train_mu, train_std = train_images.mean(), train_images.std()

train_images = (train_images - train_mu) / train_std
test_images = (test_images - train_mu) / train_std

vis.images(train_images.cpu()[torch.arange(0, 1000, 100).long()] * train_std + train_mu)

log_string('Start training')

for e in range(nb_epochs):
    acc_train_loss = 0.0

    if (e+1) % 5 == 0:
        batch_train_images = train_images[torch.arange(0, 1000, 100).long()].cuda(GPU)
        result = model(batch_train_images).detach().cpu() * train_std + train_mu
        vis.images(result.detach().clamp(min=0, max=255))

    for b in range(0, train_images.size(0), batch_size):
        real_batch_size = min(batch_size, train_images.size(0) - b)
        batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)
        output = model(batch_train_images)
        # print(output.shape)
        # print(batch_train_images.shape)
        # import ipdb; ipdb.set_trace()
        loss = criterion(output, batch_train_images)
        acc_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_test_loss = 0.0

    for b in range(0, test_images.size(0), batch_size):
        real_batch_size = min(batch_size, test_images.size(0) - b)
        batch_test_images = test_images.narrow(0, b, real_batch_size).cuda(GPU)
        output = model(batch_test_images)
        loss = criterion(output, batch_test_images)
        acc_test_loss += loss.item()

    log_string('train_loss {:d} {:f} {:f}'.format(e, acc_train_loss, acc_test_loss))

    if best_acc_train_loss is None or acc_train_loss < best_acc_train_loss:
        best_model_state = copy.deepcopy(model.state_dict())
        best_acc_train_loss = acc_train_loss

######################################################################

model.load_state_dict(best_model_state)

batch_train_images = train_images[torch.randperm(train_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
result = model(batch_train_images).detach().cpu() * train_std + train_mu
vis.images(batch_train_images.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

batch_test_images = test_images[torch.randperm(test_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
result = model(batch_test_images).detach().cpu() * train_std + train_mu
vis.images(batch_test_images.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

batch_test_images = test_images[torch.randperm(test_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
result = model(batch_test_images).detach().cpu() * train_std + train_mu
vis.images(batch_test_images.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

######################################################################
