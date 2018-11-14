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

        for t in range(nb):
            if t == 0 or random.random() < 0.1:
                if random.random() < 0.3:
                    a = 3
                else:
                    a = random.randrange(len(self.actions))
            reward = self.game.make_action(self.actions[a][1])

            state = self.game.get_state()
            if state is None:
                self.game.new_episode()
                state = self.game.get_state()

            frame = torch.from_numpy(state.screen_buffer).float()
            batch_images[t] = frame
            batch_actions[t] = a

        return batch_images, batch_actions

######################################################################

vis = None

vis = visdom.Visdom()

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

######################################################################

nb_frames = 50000
world = World()
train_images, train_actions = world.generate_batch(1)
print(train_images.shape)

log_string('Generating %d train images' % nb_frames)
train_images, train_actions = world.generate_batch(nb_frames)

train_mu, train_std = train_images.mean(), train_images.std()
train_images = (train_images - train_mu) / train_std

# batch_train_images = train_images[torch.arange(100, 500, 1).long()] * train_std + train_mu
batch_train_images = train_images * train_std + train_mu
np.save('out_images', batch_train_images.numpy())
# vis.images(batch_train_images.cpu())
# vis.images((batch_train_images[1:] - batch_train_images[:-1]).cpu())
# vis.images((batch_train_images[5:] - batch_train_images[:-5]).cpu())


# batch_train_images = train_images[torch.arange(0, 1000, 100).long()]
# vis.images(batch_train_images.cpu() * train_std + train_mu)
#
# batch_train_images = train_images[torch.randperm(train_images.size(0)).narrow(0, 0, 16).long()].cuda(GPU)
# vis.images(batch_train_images.cpu() * train_std + train_mu)
