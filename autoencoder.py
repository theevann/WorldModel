import torch
import torch.nn as nn


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

    def forward(self, X):
        return self._decoder(X)


class ConvAutoEncoder(nn.Module):
    def __init__(self, layer_specs_enc, layer_specs_dec):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = ConvEncoder(layer_specs_enc)
        self.decoder = ConvDecoder(layer_specs_dec)

    def forward(self, X):
        Z = self.encoder(X)
        # print(Z.shape)
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
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Unflatten(nn.Module):
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
