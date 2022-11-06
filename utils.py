from torch import nn
from collections import OrderedDict


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool1d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv3' in layer_name or 'deconv2' in layer_name:
            transposeConv1d = nn.ConvTranspose1d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv1d))
            if 'relu' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('leaky_' + layer_name,
                               # nn.Tanh()))
                               # nn.Sigmoid()))
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'deconv1' in layer_name:
            transposeConv1d = nn.ConvTranspose1d(in_channels=v[0],
                                                 out_channels=v[1],
                                                 kernel_size=v[2],
                                                 stride=v[3],
                                                 padding=v[4],
                                                 output_padding=1)
            layers.append((layer_name, transposeConv1d))
            if 'relu' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('leaky_' + layer_name,
                               # nn.Tanh()))
                               # nn.Sigmoid()))
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv1d = nn.Conv1d(in_channels=v[0],
                               out_channels=v[1],
                               kernel_size=v[2],
                               stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv1d))
            if 'relu' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                # layers.append(('dropout', nn.Dropout(0.5)))
                layers.append(('leaky_' + layer_name,
                               # nn.Tanh()))
                               # nn.Sigmoid()))
                               nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))
