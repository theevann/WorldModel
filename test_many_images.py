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
from tqdm import tqdm

import torch
import numpy as np
import copy

from torch import Tensor, nn

from termcolor import colored

######################################################################

def log_string(s, color=None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())
    if color is not None:
        s = colored(s, color)
    print(t + s)

######################################################################

class World:
    def __init__(self):
        self.game = DoomGame()

        self.game.set_window_visible(False)
        self.game.set_doom_map("map04")

        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.CRCGCB)  # This gives 3xHxW tensor
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

        random.seed(0)
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

    def test_images(self, nb, print_every):
        for t in tqdm(range(nb)):
            if t == 0 or random.random() < 0.1:
                a = random.randrange(len(self.actions))
            reward = self.game.make_action(self.actions[a][1])

            state = self.game.get_state()
            # import ipdb; ipdb.set_trace()
            if self.game.is_episode_finished() or self.game.is_player_dead():
                self.game.new_episode()
                state = self.game.get_state()

            frame = torch.from_numpy(state.screen_buffer).float()
            if t % print_every == 0:
                vis.images(frame.cpu())

######################################################################

vis = visdom.Visdom(env='test_images')

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)

######################################################################

nb_frames = 25000
world = World()

log_string('Generating %d train images' % nb_frames)
world.test_images(nb_frames, print_every=250)
