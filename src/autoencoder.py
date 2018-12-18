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


#######################################################################################


class ConvAutoEncoderDense_v2(nn.Module):
    def __init__(self, img_dim, dim1, dim2, dim3):
        super(ConvAutoEncoderDense_v2, self).__init__()
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
