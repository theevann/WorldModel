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
import random, time

import torch
import numpy as np
import copy

from torch import Tensor, nn

from termcolor import colored

######################################################################

def log_string(s, color = None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())
    if color is not None:
        s = colored(s, color)
    print(t + s)

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

    def test_inertia(self, nb, move_at, action=2):
        batch_images = Tensor(nb, self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        batch_actions = torch.LongTensor(nb)

        for t in range(300):
            self.game.make_action(self.actions[3][1])

        for t in range(nb):
            if t == move_at:
                a = action
            else:
                a = 3
            reward = self.game.make_action(self.actions[a][1])

            state = self.game.get_state()
            if state is None:
                self.game.new_episode()
                state = self.game.get_state()

            frame = torch.from_numpy(state.screen_buffer).float()
            batch_images[t] = frame
            batch_actions[t] = a

        return batch_images, batch_actions

    def test_inertia_2_moves(self, nb, move_at_1, move_at_2, action=2):
        batch_images = Tensor(nb, self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        batch_actions = torch.LongTensor(nb)

        for t in range(300):
            self.game.make_action(self.actions[3][1])

        for t in range(nb):
            if t == move_at_1 or t == move_at_2:
                a = action
            else:
                a = 3
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

vis = visdom.Visdom(env='test_inertia')

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

######################################################################

# PICK AN ACTION TO TEST
a = 0  # Head Rotation
a = 2  # Forward


### FIRST TEST

world = World()
nb_frames = 2 + 29
log_string('Testing inertia - Forward move at frame t=+2 - End of vis at t=+31')
train_images, train_actions = world.test_inertia(nb_frames, move_at=2, action=a)
grad = train_images[1:] - train_images[:-1]

vis.images(train_images.cpu())
vis.images(grad.cpu())


### SECOND TEST

world = World()
nb_frames = 3 + 35
log_string('Testing inertia - Two forward move')
train_images, train_actions = world.test_inertia_2_moves(nb_frames, move_at_1=2, move_at_2=3, action=a)
grad = train_images[1:] - train_images[:-1]

vis.images(train_images.cpu())
vis.images(grad.cpu())


### CONCLUSION
# No inertia with head rotation
# Inertia ~30 frame with forward move
