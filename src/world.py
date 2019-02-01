import torch
import random
from vizdoom import *
from tqdm import tqdm


class World:
    def __init__(self, nbots, skip=False, seed=0, label=False):
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
        self.game.set_labels_buffer_enabled(label)

        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.MOVE_FORWARD)

        self.game.set_seed(seed)  # DETERMINISTIC GAME !
        self.game.init()

        self.image_shape = [self.game.get_screen_channels(),
                            self.game.get_screen_height(),
                            self.game.get_screen_width()]

        self.actions = [
            ('turn_left',    [ True, False, False ]),
            ('turn_right',   [ False, True, False ]),
            ('move_forward', [ False, False, True ]),
            ('stay_put',     [ False, False, False]),
        ]

        # Add bots
        for i in range(nbots):
            self.game.send_game_command("addbot")

    def generate_batch(self, nb, countEnemy=False):
        batch_images = torch.Tensor(nb, *self.image_shape)
        batch_actions = torch.LongTensor(nb)
        skipped = 0
        nbFramesEnemy = 0

        for t in tqdm(range(nb)):
            while True:
                if t == 0 or random.random() < 0.1:
                    a = random.randrange(len(self.actions))
                reward = self.game.make_action(self.actions[a][1])

                state = self.game.get_state()

                # Check enemy presence in image (object needs to be player and big enough)
                if countEnemy and state.labels is not None:
                    nbFramesEnemy += any((obj.object_name == "DoomPlayer" and obj.height >= 60) for obj in state.labels)
                    # for obj in state.labels:
                    #     if obj.object_name == "DoomPlayer" and obj.height >= 60:
                    #         print(t, obj.height)

                if self.game.is_episode_finished() or self.game.is_player_dead():
                    self.game.new_episode()
                    state = self.game.get_state()

                frame = torch.from_numpy(state.screen_buffer).float()
                batch_images[t] = frame
                batch_actions[t] = a

                if not self.skip or (t == 0) or ((batch_images[t] - batch_images[t-1]).sum() != 0):
                    break
                skipped += 1

        print("Skipped %d images" % skipped, end='\n')
        if countEnemy:
            print("Enemy Presence %d %%" % (nbFramesEnemy * 100. / nb), end='\n')

        return batch_images, batch_actions

    def generate_batch_better(self):
        # TODO
        pass
