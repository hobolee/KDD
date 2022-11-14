from collections import OrderedDict
from ConvRNN import CGRU_cell, CLSTM_cell, CGRU_cell_wrf


# build model
# in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4]
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 32, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        CLSTM_cell(shape=(64,64), input_channels=8, filter_size=5, num_features=32),
        CLSTM_cell(shape=(32,32), input_channels=32, filter_size=5, num_features=64),
        CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64)
    ]
]

convlstm_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [64, 64, 4, 2, 1]}),
        OrderedDict({
            'conv3_leaky_1': [32, 8, 3, 1, 1],
            'conv4_leaky_1': [8, 1, 1, 1, 0]
        }),
    ],

    [
        CLSTM_cell(shape=(16,16), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=64),
        CLSTM_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
    ]
]


convgru_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 4, 3, 1, 1]}),
        OrderedDict({'conv2_leaky_1': [8, 8, 3, 2, 1]}),
        OrderedDict({'conv3_leaky_1': [16, 16, 3, 2, 1]}),
    ],

    [
        # CGRU_cell(shape=(137), input_channels=4, filter_size=5, num_features=8),
        # CGRU_cell(shape=(69), input_channels=8, filter_size=5, num_features=16),
        # CGRU_cell(shape=(35), input_channels=16, filter_size=5, num_features=32)
        CGRU_cell(shape=(26), input_channels=4, filter_size=5, num_features=8),
        CGRU_cell(shape=(13), input_channels=8, filter_size=5, num_features=16),
        CGRU_cell(shape=(7), input_channels=16, filter_size=5, num_features=32)
    ]
]

convgru_decoder_params = [
    [
        OrderedDict({'deconv1_leaky_1': [32, 32, 4, 2, 2]}),
        OrderedDict({'deconv2_leaky_1': [16, 16, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [8, 8, 3, 1, 1],
            'conv4_no_1': [8, 1, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(shape=(7), input_channels=32, filter_size=5, num_features=32),
        CGRU_cell(shape=(13), input_channels=32, filter_size=5, num_features=16),
        CGRU_cell(shape=(26), input_channels=16, filter_size=5, num_features=8),
    ]
]


