import random
from tqdm import tqdm
import torch
from vizdoom import *


class World:
    def __init__(self, nbots, skip=False):
        self.skip = skip
        self.game = DoomGame()

        self.game.set_window_visible(False)

        # self.game.set_doom_scenario_path("freedoom2.wad")
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

        self.game.set_seed(0)  # DETERMINISTIC GAME !
        self.game.init()

        self.game.new_episode()

        self.actions = [
            ('turn_left',    [ True, False, False ]),
            ('turn_right',   [ False, True, False ]),
            ('move_forward', [ False, False, True ]),
            ('stay_put',     [ False, False, False]),
        ]

        # Add bots
        for i in range(nbots):
            self.game.send_game_command("addbot")

    def generate_batch(self, nb):
        batch_images = torch.Tensor(nb, self.game.get_screen_channels(), self.game.get_screen_height(), self.game.get_screen_width())
        batch_actions = torch.LongTensor(nb)
        c = 0

        for t in tqdm(range(nb)):
            while True:
                if t == 0 or random.random() < 0.1:
                    a = random.randrange(len(self.actions))
                reward = self.game.make_action(self.actions[a][1])

                state = self.game.get_state()

                if self.game.is_episode_finished() or self.game.is_player_dead():
                    self.game.new_episode()
                    state = self.game.get_state()

                # misc = state.game_variables
                frame = torch.from_numpy(state.screen_buffer).float()
                batch_images[t] = frame
                batch_actions[t] = a

                if not self.skip or (t == 0) or ((batch_images[t] - batch_images[t-1]).sum() != 0):
                    break
                c += 1
                # print("Skipped image %d" % c, end='\n')

        print("Skipped %d images" % c, end='\n')

        return batch_images, batch_actions
