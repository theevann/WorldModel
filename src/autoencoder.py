import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from clstm import Generator

h_dim = 1000
z_dim = 400


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * 240 * 320, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.ReLU()
        )

    def forward(self, X):
        return self.encoder(X.reshape(-1, 3 * 240 * 320))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 240 * 320 * 3),
        )

    def forward(self, X):
        return self.decoder(X).reshape(-1, 3, 240, 320)


class AutoEncoder(nn.Module):
    def __init__(self, ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X):
        Y = self.encoder(X)
        Z = self.decoder(Y)
        return Z

    def process(self, img_in):
        return self.forward(img_in)


########################################

class ConvEncoder(nn.Module):
    def __init__(self, layer_specs, z_dim=0):
        super(ConvEncoder, self).__init__()

        modules = []
        for spec in layer_specs:
            modules.append(nn.Conv2d(**spec))
            modules.append(nn.ReLU(True))

        self._encoder = nn.Sequential(*modules)
        self.init_weights()

    def init_weights(self):
        for mod in self._encoder.children():
            if type(mod) is nn.Conv2d:
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')

    def forward(self, X):
        return self._encoder(X)


class ConvDecoder(nn.Module):
    def __init__(self, layer_specs):
        super(ConvDecoder, self).__init__()
        modules = []
        for spec in layer_specs:
            modules.append(nn.ConvTranspose2d(**spec))
            modules.append(nn.ReLU(True))
        modules.pop()

        self._decoder = nn.Sequential(*modules)
        self.init_weights()

    def init_weights(self):
        for mod in self._decoder.children():
            if type(mod) is nn.ConvTranspose2d:
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')

    def forward(self, X):
        return self._decoder(X)


class ConvAutoEncoder(nn.Module):
    def __init__(self, layer_specs_enc, layer_specs_dec):
        super(ConvAutoEncoder, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoder(layer_specs_enc)
        self.decoder = ConvDecoder(layer_specs_dec)

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def get_embed_shape(self, img_dim):
        return self.encoder(torch.rand(1, *img_dim)).shape[1:]

#############################################################


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        return x


class Dense(nn.Module):
    def __init__(self, layer_dims, dim_in):
        super(Dense, self).__init__()
        modules = []
        previous_dim = dim_in
        layer_dims.append(dim_in)

        modules.append(Flatten())
        for i, dim in enumerate(layer_dims):
            modules.append(nn.Linear(previous_dim, dim))
            modules.append(nn.ReLU(True))
            previous_dim = dim
        modules.append(Unflatten())

        self._dense = nn.Sequential(*modules)
        self.z_dim = min(layer_dims)

    def forward(self, X):
        return self._dense(X)


class ConvEncoderDense(ConvEncoder):
    def __init__(self, layer_specs, img_dim):
        super(ConvEncoderDense, self).__init__(layer_specs)
        self.z_dim = self._encoder(torch.rand(1, *img_dim)).numel()


class ConvDecoderDense(ConvDecoder):
    def __init__(self, layer_specs, z_dim):
        layer_specs[0]['in_channels'] = z_dim
        super(ConvDecoderDense, self).__init__(layer_specs)


class ConvAutoEncoderDense(nn.Module):
    def __init__(self, layer_specs_enc, layer_specs_dec, layer_specs_dense, img_dim):
        super(ConvAutoEncoderDense, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoderDense(layer_specs_enc, img_dim)
        self.dense = Dense(layer_specs_dense, self.encoder.z_dim)
        self.decoder = ConvDecoderDense(layer_specs_dec, self.encoder.z_dim)

    def forward(self, X):
        Y = self.encoder(X)
        Z = self.dense(Y)
        X = self.decoder(Z)
        return X

    def encode(self, X):
        print("Warning : Output of Convolution without Dense")
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def get_embed_shape(self, img_dim):
        return self.dense.z_dim


#######################################################################################


class ConvAutoEncoderDense_v2(nn.Module):
    def __init__(self, img_dim, dim1, dim2, dim3):
        super(ConvAutoEncoderDense_v2, self).__init__()
        self.args = copy.deepcopy(locals())
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.relu = nn.ReLU(True)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(8,8), stride=4),
            self.relu,
            nn.Conv2d(32, 64, kernel_size=(6,6), stride=3),
            self.relu,
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=2),
            self.relu,
            Flatten()
        )

        for mod in self.encoder1.children():
            if type(mod) is nn.Conv2d:
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')

        self.encoder2 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=(8,8), stride=4),
            self.relu,
            nn.Conv2d(32, 64, kernel_size=(6,6), stride=3),
            self.relu,
            nn.Conv2d(64, 128, kernel_size=(4,4), stride=2),
            self.relu,
            Flatten()
        )

        for mod in self.encoder2.children():
            if type(mod) is nn.Conv2d:
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')

        image = torch.rand(1, *img_dim)

        self.d1 = self.encoder1(image).numel()
        self.dense1 = nn.Linear(self.d1, self.dim1)

        self.d2 = self.encoder2(torch.cat((image, image), 1)).numel()
        self.dense2 = nn.Linear(self.d2, self.dim2)

        self.dense3 = nn.Linear(self.dim1 + self.dim2, self.dim3)

        self.decoder = nn.Sequential(
            Unflatten(),
            nn.ConvTranspose2d(self.dim3, 256, kernel_size=(3,4), stride=3),
            self.relu,
            nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=3),
            self.relu,
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=3, output_padding=(1,2)),
            self.relu,
            nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=2, output_padding=1),
            self.relu,
            nn.ConvTranspose2d(32, 3, kernel_size=(8,8), stride=4)
        )

        for mod in self.decoder.children():
            if type(mod) is nn.ConvTranspose2d:
                nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')


    def forward(self, X_0, X_1, out=0):
        X_2 = torch.cat((X_0, X_1), 1)
        Y_1 = self.relu(self.dense1(self.encoder1(X_0))) * (0 if (out == 2) else 1)  # try tanh
        Y_2 = self.relu(self.dense2(self.encoder2(X_2))) * (0 if (out == 1) else 1)
        Z = self.dense3(torch.cat((Y_1, Y_2), 1))
        X = self.decoder(Z)
        return X

    def forward_1(self, X_0, X_1):
        X_2 = torch.cat((X_0, X_1), 1)
        Y_1 = self.relu(self.dense1(self.encoder1(X_0)))  # try tanh
        Y_2 = self.relu(self.dense2(self.encoder2(X_2))).zero_()
        Z = self.dense3(torch.cat((Y_1, Y_2), 1))
        X = self.decoder(Z)
        return X


    def forward_2(self, X_0, X_1):
        X_2 = torch.cat((X_0, X_1), 1)
        Y_1 = self.relu(self.dense1(self.encoder1(X_0))).zero_()  # try tanh
        Y_2 = self.relu(self.dense2(self.encoder2(X_2)))
        Z = self.dense3(torch.cat((Y_1, Y_2), 1))
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return (self.dim1, self.dim2, self.dim3)


#######################################################################################

# TODO: make it residual
class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.relu = nn.ReLU(True)
        self.flatten = Flatten()

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
        )

        self.b2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)

    def forward(self, X):
        return self.flatten(self.b2(self.b1(X)))


class ResEncoderDeep(nn.Module):
    def __init__(self):
        super(ResEncoderDeep, self).__init__()
        self.relu = nn.ReLU(True)
        self.flatten = Flatten()

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
        )

        self.b2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
        )

        self.b3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)
        init_weights(self.b3)

    def forward(self, X):
        return self.flatten(self.b3(self.b2(self.b1(X))))


class ResEncoderRes(nn.Module):
    def __init__(self):
        super(ResEncoderRes, self).__init__()
        self.relu = nn.ReLU(True)
        self.flatten = Flatten()

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=4),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)

    def forward(self, X):
        Y = self.b1(X)
        return self.flatten(self.b2(Y) + Y)


class ResEncoderBN(nn.Module):
    def __init__(self):
        super(ResEncoderBN, self).__init__()
        self.relu = nn.ReLU(True)
        self.flatten = Flatten()

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
        )

        self.b2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)

    def forward(self, X):
        return self.flatten(self.b2(self.b1(X)))


class ResEncoderAll(nn.Module):
    def __init__(self):
        super(ResEncoderAll, self).__init__()
        self.relu = nn.ReLU(True)
        self.flatten = Flatten()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)
        init_weights(self.b3)

    def forward(self, X):
        Y = self.maxpool(self.b1(X))
        Z = self.maxpool(self.b2(Y) + Y)
        return self.flatten(self.b3(Z) + Z)


class ConvAutoEncoderDense_B(nn.Module):
    def __init__(self, img_dim, dim_latent, dim_dense_out, encoder):
        super(ConvAutoEncoderDense_B, self).__init__()
        self.args = copy.deepcopy(locals())
        self.dim_latent = dim_latent
        self.dim_dense_out = dim_dense_out
        self.relu = nn.ReLU(True)

        self.encoder = encoder()

        image = torch.rand(1, *img_dim)

        self.dim_enc_out = self.encoder(image).numel()
        self.dense1 = nn.Linear(self.dim_enc_out, self.dim_latent)
        self.dense2 = nn.Linear(self.dim_latent, self.dim_dense_out)

        self.decoder = nn.Sequential(
            Unflatten(),
            nn.ConvTranspose2d(self.dim_dense_out, 256, kernel_size=(3,4), stride=3),
            self.relu,
            nn.ConvTranspose2d(256, 128, kernel_size=(3,3), stride=3),
            self.relu,
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=3),
            self.relu,
            nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=4),
            self.relu,
            nn.ConvTranspose2d(32, 3, kernel_size=(2,2), stride=2)
        )

        init_weights(self.decoder)

    def forward(self, X):
        Y = self.encoder(X)
        Z = self.dense1(Y)
        Y = self.relu(self.dense2(Z))
        X = F.interpolate(self.decoder(Y), size=X.shape[-2:])
        return X

    def get_embed_shape(self):
        return self.dim_latent


####

class ResEncoderNoFlatten(nn.Module):
    def __init__(self):
        super(ResEncoderNoFlatten, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
        )

        self.b2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
        )

        init_weights(self.b1)
        init_weights(self.b2)

    def forward(self, X):
        return self.b2(self.b1(X))

class ConvAutoEncoderDense_C(nn.Module):
    def __init__(self, img_dim, encoder):
        super(ConvAutoEncoderDense_C, self).__init__()
        self.args = copy.deepcopy(locals())

        self.encoder = encoder()

        image = torch.rand(1, *img_dim)
        output = self.encoder(image)
        self.dim_latent = output.numel()

        self.decoder = Generator(input_size=output.shape[-2:],
                                 input_dim=output.size(1),
                                 hidden_dim=output.size(1),
                                 kernel_size=(5, 5),
                                 genstep=10)

    def forward(self, X):
        Y = self.encoder(X)
        # X = F.interpolate(self.decoder(Y), size=X.shape[-2:])
        X = self.decoder(Y)
        return X

    def get_embed_shape(self):
        return self.dim_latent


##########################

class ConvDecoder2(nn.Module):
    def __init__(self):
        super(ConvDecoder2, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(4), stride=4),
            self.relu,
            nn.ConvTranspose2d(64, 3, kernel_size=(4), stride=4),
            self.relu,
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvAutoEncoder2(nn.Module):
    def __init__(self, img_dim):
        super(ConvAutoEncoder2, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ResEncoderNoFlatten()
        self.decoder = ConvDecoder2()
        self.img_dim = img_dim

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return self.encoder(torch.rand(1, *self.img_dim).cuda()).shape[1:]

#######


class ConvEncoder3(nn.Module):
    def __init__(self):
        super(ConvEncoder3, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.AvgPool2d(kernel_size=(3, 4)),
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvDecoder3(nn.Module):
    def __init__(self):
        super(ConvDecoder3, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(3,4), stride=(3,4)),
            self.relu,
            nn.ConvTranspose2d(64, 64, kernel_size=(4), stride=4),
            self.relu,
            nn.ConvTranspose2d(64, 3, kernel_size=(4), stride=4),
            self.relu,
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)

class ConvAutoEncoder3(nn.Module):
    def __init__(self, img_dim):
        super(ConvAutoEncoder3, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoder3()
        self.decoder = ConvDecoder3()
        self.img_dim = img_dim

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return self.encoder(torch.rand(1, *self.img_dim).to(next(self.parameters()).device)).shape[1:]


######


class ConvEncoder3_1(nn.Module):
    def __init__(self, dim):
        super(ConvEncoder3_1, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, dim, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(dim, dim, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.AvgPool2d(kernel_size=(15, 20)),
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvDecoder3_1(nn.Module):
    def __init__(self, dim):
        super(ConvDecoder3_1, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=(15,20), stride=(15,20)),
            self.relu,
            nn.ConvTranspose2d(dim, 64, kernel_size=(4), stride=4),
            self.relu,
            nn.ConvTranspose2d(64, 3, kernel_size=(4), stride=4),
            self.relu,
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvAutoEncoder3_1(nn.Module):
    def __init__(self, img_dim, dim):
        super(ConvAutoEncoder3_1, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoder3_1(dim)
        self.decoder = ConvDecoder3_1(dim)
        self.img_dim = img_dim

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return self.encoder(torch.rand(1, *self.img_dim).to(next(self.parameters()).device)).shape[1:]


######

class ConvEncoder3_2(nn.Module):
    def __init__(self):
        super(ConvEncoder3_2, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=(3, 4)),
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvAutoEncoder3_2(nn.Module):
    def __init__(self, img_dim):
        super(ConvAutoEncoder3_2, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoder3_2()
        self.decoder = ConvDecoder3()
        self.img_dim = img_dim

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return self.encoder(torch.rand(1, *self.img_dim).to(next(self.parameters()).device)).shape[1:]


####



class ConvEncoder3_3(nn.Module):
    def __init__(self, dim):
        super(ConvEncoder3_3, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(3, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 64, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, dim, kernel_size=(3), stride=1, padding=1),
            self.relu,
            nn.AvgPool2d(kernel_size=(3, 4)),
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvDecoder3_3(nn.Module):
    def __init__(self, dim):
        super(ConvDecoder3_3, self).__init__()
        self.relu = nn.ReLU(True)

        self.b1 = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, kernel_size=(3,4), stride=(3,4)),
            self.relu,
            nn.ConvTranspose2d(64, 64, kernel_size=(4), stride=4),
            self.relu,
            nn.ConvTranspose2d(64, 3, kernel_size=(4), stride=4),
            self.relu,
        )

        init_weights(self.b1)

    def forward(self, X):
        return self.b1(X)


class ConvAutoEncoder3_3(nn.Module):
    def __init__(self, img_dim, dim):
        super(ConvAutoEncoder3_3, self).__init__()
        self.args = copy.deepcopy(locals())
        self.encoder = ConvEncoder3_3(dim)
        self.decoder = ConvDecoder3_3(dim)
        self.img_dim = img_dim

    def forward(self, X):
        Z = self.encoder(X)
        X = self.decoder(Z)
        return X

    def get_embed_shape(self):
        return self.encoder(torch.rand(1, *self.img_dim).to(next(self.parameters()).device)).shape[1:]


####

class DoubleConvAutoEncoder(nn.Module):
    def __init__(self, class_1, args_1, class_2, args_2, combine, shift=0):
        super(DoubleConvAutoEncoder, self).__init__()
        self.args = copy.deepcopy(locals())
        self.ae_1 = class_1(**args_1)
        self.ae_2 = class_2(**args_2)
        self.combine = combine
        self.shift = shift

    def forward(self, X):
        out_1 = self.ae_1(X[self.shift:])
        out_2 = self.ae_2(X[:((-self.shift-1) % len(X)) + 1])
        return self.combine(out_1, out_2)

    def get_embed_shape(self):
        return (self.ae_1.get_embed_shape(), self.ae_2.get_embed_shape())


def combine_sum(a, b):
    return a + b

def init_weights(obj):
    for mod in obj.children():
        if (type(mod) is nn.ConvTranspose2d) or (type(mod) is nn.Conv2d):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
