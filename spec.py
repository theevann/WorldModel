

spec_1 = {
    "layer_specs_enc_high": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_high": [
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_enc_low": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
    ],
    "layer_specs_dec_low": [
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (0, 1)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 6, 'stride': 3, 'output_padding': (2, 1)},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ]
}


spec_2 = {
    "layer_specs_enc_high": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_high": [
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_enc_low": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
    ],
    "layer_specs_dec_low": [
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (0, 1)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 6, 'stride': 3, 'output_padding': (2, 1)},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ]
}


spec_3 = {
    "layer_specs_enc_high": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_high": [
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_enc_low": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_low": [
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
}


spec_4 = {
    "layer_specs_enc_high": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_high": [
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_enc_low": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 2},
    ],
    "layer_specs_dec_low": [
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (0, 1)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 6, 'stride': 3, 'output_padding': (2,1)},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ]
}

spec_5 = {
    "layer_specs_enc_high": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 4, 'stride': 2},
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3},
    ],
    "layer_specs_dec_high": [
        {'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_enc_low": [
        {'in_channels':  3, 'out_channels':  32, 'kernel_size': 8, 'stride': 4},
        {'in_channels':  32, 'out_channels':  64, 'kernel_size': 6, 'stride': 3},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 4, 'stride': 2},
    ],
    "layer_specs_dec_low": [
        {'in_channels': 0, 'out_channels': 256, 'kernel_size': (3, 4), 'stride': 3},
        {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 3},
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 3, 'output_padding': (1, 2)},
        {'in_channels':  64, 'out_channels':  32, 'kernel_size': 4, 'stride': 2, 'output_padding': 1},
        {'in_channels':  32, 'out_channels':  3, 'kernel_size': 8, 'stride': 4},
    ],
    "layer_specs_dense": [1000, 100, 1000],
}


# ++data (trying for dense)
# PCA ok
# smarter policy => put only in training couple of image where we didnt move
# L2 ?
