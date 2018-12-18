import visdom
import random, time, sys

import torch
import torch.nn as nn
import numpy as np
import copy

from termcolor import colored

######################################################################

from autoencoder import ConvAutoEncoderDense_v2
from world import World

######################################################################

def log_string(s, color=None):
    t = time.strftime("%Y-%m-%d_%H:%M:%S - ", time.localtime())

    if color is not None:
        s = colored(s, color)

    print(t + s)
    sys.stdout.flush()

######################################################################

show_progress = True
GPU = 3
DATAPARALLEL = False

assert torch.cuda.is_available(), 'We need a GPU to run this.'
torch.cuda.device(GPU)

random.seed(0)
torch.cuda.manual_seed_all(0)
torch.manual_seed(0)
np.random.seed(0)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

######################################################################

env = "main"
# env = "double_spec5-l100d-h6912-50k"
vis = visdom.Visdom(env=env, log_to_filename="log/" + env + ".log")

if vis.check_connection():
    log_string('Visdom server ' + vis.server + ':' + str(vis.port))
else:
    log_string('Cannot connect to the visdom server. Does it run? (\'python -m visdom.server\')')
    exit(1)


class Plot(object):
    def __init__(self, vis):
        self.viz = vis
        self.windows = {}
        self.viz.register_event_handler(self.callback, "images")
        self.callbackhit = False

    def register(self, name, xlabel="Step", ylabel=None):
        ylabel = ylabel if ylabel is not None else name
        win = self.viz.line(
            X=np.array([0]),
            Y=np.array([1]),
            opts=dict(title=name, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update(self, name, x, y):
        self.viz.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=self.windows[name],
            update="append"
        )

    def sendImages(self, images):
        self.viz.images(images, win="images")
        self.callbackhit = False

    def callback(self, event):
        self.callbackhit = True


vislogger = Plot(vis)
vislogger.register('Loss')

######################################################################

world = World(nbots=5, skip=True)

train_images, train_actions = world.generate_batch(1)
image_shape = train_images.shape[1:]
print('Image shape', train_images.shape)

model = ConvAutoEncoderDense_v2(image_shape, 300, 100, 200)
log_string(str(model))

embed_shape = model.get_embed_shape()
log_string('Embedding dimensions are ' + str(embed_shape))

criterion = torch.nn.MSELoss()
lr = 1e-3
clip = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.cuda(GPU)
m = model
if DATAPARALLEL:
    model = nn.DataParallel(model)

nb_frames, nb_epochs = 50000, 1000
batch_size = 100
shift = 2

wu_nb_frames, wu_nb_epochs = 1000, 100
wu_batch_size = 200

best_acc_train_loss = None

log_string('Generating %d train images' % nb_frames)
# train_images, train_actions = world.generate_batch(nb_frames)
# torch.save((train_images, train_actions), "data/train")
(train_images, train_actions) = torch.load("data/train")

log_string('Generating %d test images' % int(nb_frames / 5))
# test_images, test_actions = world.generate_batch(int(nb_frames / 5))
# torch.save((test_images, test_actions), "data/test")
(test_images, test_actions) = torch.load("data/test")

train_mu, train_std = train_images.mean(), train_images.std()

train_images = (train_images - train_mu) / train_std
test_images = (test_images - train_mu) / train_std

#############################################################

if False:
    log_string('Start warmup')

    for e in range(wu_nb_epochs):
        acc_train_loss = 0.0

        if show_progress and (e+1) % 10 == 0:
            perm = torch.randint(0, wu_nb_frames-shift, (8,)).long()
            batch_images_0 = train_images[perm].cuda(GPU)
            batch_images_1 = train_images[perm+shift].cuda(GPU)

            with torch.no_grad():
                vis.images(batch_images_0.cpu() * train_std + train_mu)
                vis.images(batch_images_1.cpu() * train_std + train_mu)
                result = model(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))
                result = m.forward_1(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))
                result = m.forward_2(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))

        dnf = int(wu_nb_frames + e * (nb_frames - wu_nb_frames) / wu_nb_epochs)
        dnf = wu_nb_frames
        for b in range(0, dnf, wu_batch_size):
            real_batch_size = min(wu_batch_size, dnf - b)
            batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)

            if(real_batch_size <= 1):
                continue

            output = model(batch_train_images[:-shift], batch_train_images[shift:])
            loss = criterion(output, batch_train_images[shift:])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            acc_train_loss += (loss.item()) * real_batch_size

        log_string('Loss warmup epoch {:d} | train {:f} | dnf {:d}'.format(e+1, acc_train_loss / dnf, dnf))

    #############################################################

    log_string('Start transition')

    for e in range(wu_nb_epochs * 2):
        acc_train_loss = 0.0

        if show_progress and (e+1) % 10 == 0:
            perm = torch.randint(0, wu_nb_frames-shift, (8,)).long()
            batch_images_0 = train_images[perm].cuda(GPU)
            batch_images_1 = train_images[perm+shift].cuda(GPU)

            with torch.no_grad():
                vis.images(batch_images_0.cpu() * train_std + train_mu)
                vis.images(batch_images_1.cpu() * train_std + train_mu)
                result = model(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))
                result = m.forward_1(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))
                result = m.forward_2(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu
                vis.images(result.clamp(min=0, max=255))

        dnf = int(wu_nb_frames + e * (nb_frames - wu_nb_frames) / wu_nb_epochs / 2)
        for b in range(0, dnf, wu_batch_size):
            real_batch_size = min(wu_batch_size, dnf - b)
            batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)

            if(real_batch_size <= 1):
                continue

            output = model(batch_train_images[:-shift], batch_train_images[shift:])
            loss = criterion(output, batch_train_images[shift:])

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            acc_train_loss += (loss.item()) * real_batch_size

        log_string('Loss transition epoch {:d} | train {:f} | dnf {:d}'.format(e+1, acc_train_loss / dnf, dnf))

###############################################################################################


print('\n')
log_string('Start training')

gc = 0
for e in range(nb_epochs):
    acc_train_loss = 0.0

    for b in range(0, train_images.size(0), batch_size):
        real_batch_size = min(batch_size, train_images.size(0) - b)
        batch_train_images = train_images[b:b + real_batch_size].cuda(GPU)
        output = model(batch_train_images[:-shift], batch_train_images[shift:])
        loss = criterion(output, batch_train_images[shift:])
        acc_train_loss += loss.item() * real_batch_size

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        gc += 1
        if (gc+1) % 10 == 0:
            vislogger.update("Loss", gc, acc_train_loss / b)

        if b == 0 or vislogger.callbackhit:
            perm = torch.randint(0, int(nb_frames / 5) - shift, (8,)).long()
            batch_images_0 = test_images[perm].cuda(GPU)
            batch_images_1 = test_images[perm+shift].cuda(GPU)
            result = [batch_images_1.cpu() * train_std + train_mu]
            with torch.no_grad():
                result.append((model(batch_images_0, batch_images_1).data.cpu() * train_std + train_mu).clamp(0,255))
                result.append((model(batch_images_0, batch_images_1, out=1).data.cpu() * train_std + train_mu).clamp(0,255))
                result.append((model(batch_images_0, batch_images_1, out=2).data.cpu() * train_std + train_mu).clamp(0,255))
            vislogger.sendImages(torch.cat(result, 0))

    acc_test_loss = 0.0

    for b in range(0, test_images.size(0), batch_size):
        real_batch_size = min(batch_size, test_images.size(0) - b)
        batch_test_images = test_images.narrow(0, b, real_batch_size).cuda(GPU)
        output = model(batch_test_images[:-shift], batch_test_images[shift:])
        loss = criterion(output, batch_test_images[shift:])
        acc_test_loss += loss.item() * real_batch_size

    log_string('Loss epoch {:d} | train {:f} | test {:f}'.format(e+1, acc_train_loss / train_images.size(0), acc_test_loss / test_images.size(0)))

    if best_acc_train_loss is None or acc_train_loss < best_acc_train_loss:
        best_model_state = copy.deepcopy(model.state_dict())
        best_acc_train_loss = acc_train_loss


######################################################################

model.load_state_dict(best_model_state)

perm = torch.randperm(train_images.size(0) - shift).narrow(0, 0, 16).long()
batch_train_images_0 = train_images[perm].cuda(GPU)
batch_train_images_1 = train_images[perm + shift].cuda(GPU)
result = model(batch_train_images_0, batch_train_images_1).data.cpu() * train_std + train_mu
vis.images(batch_train_images_0.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

perm = torch.randperm(test_images.size(0) - shift).narrow(0, 0, 16).long()
batch_test_images_0 = test_images[perm].cuda(GPU)
batch_test_images_1 = test_images[perm + shift].cuda(GPU)
result = model(batch_test_images_0, batch_test_images_1).data.cpu() * train_std + train_mu
vis.images(batch_test_images_0.cpu() * train_std + train_mu)
vis.images(result.clamp(min=0, max=255))

######################################################################
