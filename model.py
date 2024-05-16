from torch import nn
import torch

import random
import numpy as np
import torch.nn as nn
def get_rampup_ratio(i, start, end, mode = "linear"):
    """
    Obtain the rampup ratio.
    :param i: (int) The current iteration.
    :param start: (int) The start iteration.
    :param end: (int) The end itertation.
    :param mode: (str) Valid values are {`linear`, `sigmoid`, `cosine`}.
    """
    i = np.clip(i, start, end)
    if(mode == "linear"):
        rampup = (i - start) / (end - start)
    elif(mode == "sigmoid"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(np.exp(-5.0 * phase * phase))
    elif(mode == "cosine"):
        phase  = 1.0 - (i - start) / (end - start)
        rampup = float(.5 * (np.cos(np.pi * phase) + 1))
    else:
        raise ValueError("Undefined rampup mode {0:}".format(mode))
    return rampup

def initialize_weights(param, p):

    class_name = param.__class__.__name__
    if class_name.startswith('Conv2d') and random.random() <= p:
        # Initialization according to original Unet paper
        # See https://arxiv.org/pdf/1505.04597.pdf
        _, in_maps, k, _ = param.weight.shape
        n = k * k * in_maps
        std = np.sqrt(2 / n)
        nn.init.normal_(param.weight.data, mean=0.0, std=std)



def get_acti_func(acti_func, params):
    acti_func = acti_func.lower()
    if (acti_func == 'relu'):
        inplace = params.get('relu_inplace', False)
        return nn.ReLU(inplace)

    elif (acti_func == 'leakyrelu'):
        slope = params.get('leakyrelu_negative_slope', 1e-2)
        inplace = params.get('leakyrelu_inplace', False)
        return nn.LeakyReLU(slope, inplace)

    elif (acti_func == 'prelu'):
        num_params = params.get('prelu_num_parameters', 1)
        init_value = params.get('prelu_init', 0.25)
        return nn.PReLU(num_params, init_value)

    elif (acti_func == 'rrelu'):
        lower = params.get('rrelu_lower', 1.0 / 8)
        upper = params.get('rrelu_upper', 1.0 / 3)
        inplace = params.get('rrelu_inplace', False)
        return nn.RReLU(lower, upper, inplace)

    elif (acti_func == 'elu'):
        alpha = params.get('elu_alpha', 1.0)
        inplace = params.get('elu_inplace', False)
        return nn.ELU(alpha, inplace)

    elif (acti_func == 'celu'):
        alpha = params.get('celu_alpha', 1.0)
        inplace = params.get('celu_inplace', False)
        return nn.CELU(alpha, inplace)

    elif (acti_func == 'selu'):
        inplace = params.get('selu_inplace', False)
        return nn.SELU(inplace)

    elif (acti_func == 'glu'):
        dim = params.get('glu_dim', -1)
        return nn.GLU(dim)

    elif (acti_func == 'sigmoid'):
        return nn.Sigmoid()

    elif (acti_func == 'logsigmoid'):
        return nn.LogSigmoid()

    elif (acti_func == 'tanh'):
        return nn.Tanh()

    elif (acti_func == 'hardtanh'):
        min_val = params.get('hardtanh_min_val', -1.0)
        max_val = params.get('hardtanh_max_val', 1.0)
        inplace = params.get('hardtanh_inplace', False)
        return nn.Hardtanh(min_val, max_val, inplace)

    elif (acti_func == 'softplus'):
        beta = params.get('softplus_beta', 1.0)
        threshold = params.get('softplus_threshold', 20)
        return nn.Softplus(beta, threshold)

    elif (acti_func == 'softshrink'):
        lambd = params.get('softshrink_lambda', 0.5)
        return nn.Softshrink(lambd)

    elif (acti_func == 'softsign'):
        return nn.Softsign()

    else:
        raise ValueError("Not implemented: {0:}".format(acti_func))


def interleaved_concate(f1, f2, shuffle=False):
    f1_shape = list(f1.shape)
    f2_shape = list(f2.shape)
    c1 = f1_shape[1]
    c2 = f2_shape[1]

    f1_shape_new = f1_shape[:1] + [c1, 1] + f1_shape[2:]
    f2_shape_new = f2_shape[:1] + [c2, 1] + f2_shape[2:]

    f1_reshape = torch.reshape(f1, f1_shape_new)
    f2_reshape = torch.reshape(f2, f2_shape_new)
    output = torch.cat((f1_reshape, f2_reshape), dim=2)
    if shuffle:
        seq = torch.randperm(c1)
        output = output[:, seq]
    out_shape = f1_shape[:1] + [c1 + c2] + f1_shape[2:]
    output = torch.reshape(output, out_shape)
    return output


class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number.
    :param kernel_size: The size of convolution kernel. It can be either a single
        int or a tupe of two or three ints.
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution.
    :param padding: (int) Padding size.
    :param dilation: (int) Dilation rate.
    :param conv_group: (int) The groupt number of convolution.
    :param bias: (bool) Add bias or not for convolution.
    :param norm_type: (str or None) Normalization type, can be `batch_norm`, 'group_norm'.
    :param norm_group: (int) The number of group for group normalization.
    :param acti_func: (str or None) Activation funtion.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dim=3,
                 stride=1, padding=0, dilation=1, conv_group=1, bias=True,
                 norm_type='batch_norm', norm_group=1, acti_func=None):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns = in_channels
        self.n_out_chns = out_channels
        self.norm_type = norm_type
        self.norm_group = norm_group
        self.acti_func = acti_func

        assert (dim == 2 or dim == 3)
        if (dim == 2):
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, conv_group, bias)
            if (self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm2d(out_channels)
            elif (self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif (self.norm_type == 'instance_norm'):
                self.bn = nn.InstanceNorm2d(out_channels)
            elif (self.norm_type is not None):
                raise ValueError(
                    "unsupported normalization method {0:}".format(norm_type))
        else:
            self.conv = nn.Conv3d(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, conv_group, bias)
            if (self.norm_type == 'batch_norm'):
                self.bn = nn.BatchNorm3d(out_channels)
            elif (self.norm_type == 'group_norm'):
                self.bn = nn.GroupNorm(self.norm_group, out_channels)
            elif (self.norm_type == 'instance_norm'):
                self.bn = nn.InstanceNorm3d(out_channels)
            elif (self.norm_type is not None):
                raise ValueError(
                    "unsupported normalization method {0:}".format(norm_type))

    def forward(self, x):
        f = self.conv(x)
        if (self.norm_type is not None):
            f = self.bn(f)
        if (self.acti_func is not None):
            f = self.acti_func(f)
        return f


class DeconvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    deconvolution -> (batch_norm / layer_norm / group_norm / instance_norm) -> (activation) -> (dropout)
    Batch norm and activation are optional.

    :param in_channels: (int) The input channel number.
    :param out_channels: (int) The output channel number.
    :param kernel_size: The size of convolution kernel. It can be either a single
        int or a tupe of two or three ints.
    :param dim: (int) The dimention of convolution (2 or 3).
    :param stride: (int) The stride of convolution.
    :param padding: (int) Padding size.
    :param dilation: (int) Dilation rate.
    :param groups: (int) The groupt number of convolution.
    :param bias: (bool) Add bias or not for convolution.
    :param batch_norm: (bool) Use batch norm or not.
    :param acti_func: (str or None) Activation funtion.
    """

    def __init__(self, in_channels, out_channels, kernel_size,mode="transconv",
                 dim=3, stride=1, padding=0, output_padding=0,
                 dilation=1, groups=1, bias=True, 
                 norm_type="batch_norm", acti_func=None):
        super(DeconvolutionLayer, self).__init__()
        self.n_in_chns = in_channels
        self.n_out_chns = out_channels
        self.norm_type = norm_type
        self.acti_func = acti_func
        self.mode = mode.lower()

        assert (dim == 2 or dim == 3)
        if (dim == 2):
            if self.mode == "transconv":
                self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size, stride, padding, output_padding,
                                           groups, bias, dilation)
            else:
                self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
                if self.mode == "nearest":
                     self.conv = nn.Upsample(scale_factor=2, mode=self.mode)
            
                else:
                    self.conv = nn.Upsample(scale_factor=2, mode=self.mode, align_corners=True)
            
            if (self.norm_type == "group_norm"):
                self.bn = nn.GroupNorm(groups, out_channels)
            elif (self.norm_type == "batch_norm"):
                self.bn = nn.BatchNorm2d(out_channels)
            elif (self.norm_type == "instance_norm"):
                self.bn = nn.InstanceNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                                           kernel_size, stride, padding, output_padding,
                                           groups, bias, dilation)
            if (self.norm_type == "group_norm"):
                self.bn = nn.GroupNorm(groups, out_channels)
            elif (self.norm_type == "batch_norm"):
                self.bn = nn.BatchNorm3d(out_channels)
            elif (self.norm_type == "instance_norm"):
                self.bn = nn.InstanceNorm3d(out_channels)
    
    def forward(self, x):
        if self.mode != "transconv":
            x = self.conv1x1(x)
        
        f = self.conv(x)
        if (self.norm_type is not None):
            f = self.bn(f)
        if (self.acti_func is not None):
            f = self.acti_func(f)
        return f


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type, groups, acti_func):
        super(UNetBlock, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channels
        self.acti_func = acti_func

        group1 = 1 if (in_channels < 8) else groups
        self.conv1 = ConvolutionLayer(in_channels, out_channels, 3,
                                      dim=2, padding=1, conv_group=group1, norm_type=norm_type, norm_group=group1,
                                      acti_func=acti_func)
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3,
                                      dim=2, padding=1, conv_group=groups, norm_type=norm_type, norm_group=groups,
                                      acti_func=acti_func)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        return f2


class MGUNet(nn.Module):
    def __init__(self, params):
        super(MGUNet, self).__init__()
        self.params = params
        self.ft_chns = [self.params['ndf'] *
                        int(pow(2, i)) for i in range(0, 5)]
        self.in_chns = self.params['in_chns']
        self.ft_groups = self.params['feature_grps']
        self.norm_type = self.params['norm_type']
        self.n_class = self.params['class_num']
        self.acti_func = get_acti_func(self.params['acti_func'], self.params)
        self.dropout = self.params['dropout']
        self.decoder_ratio = self.params.get("decoder_ratio", 1)
        self.shuffle_channel = self.params.get("shuffle_channel", False)

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], self.norm_type[0], self.ft_groups[0],
                                self.acti_func)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], self.norm_type[0], self.ft_groups[0],
                                self.acti_func)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], self.norm_type[0], self.ft_groups[0],
                                self.acti_func)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], self.norm_type[0], self.ft_groups[0],
                                self.acti_func)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], self.norm_type[0], self.ft_groups[0],
                                self.acti_func)

        self.block6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3] * self.decoder_ratio, self.norm_type[0],
                                self.ft_groups[1],
                                self.acti_func)

        self.block7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2] * self.decoder_ratio,
                                self.norm_type[0],
                                self.ft_groups[1],
                                self.acti_func)

        self.block8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1] * self.decoder_ratio,
                                self.norm_type[0],
                                self.ft_groups[1],
                                self.acti_func)

        self.block9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0] * self.decoder_ratio,
                                self.norm_type[0],
                                self.ft_groups[1],
                                self.acti_func)

        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.up1 = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups[1],
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])

        self.up2 = DeconvolutionLayer(self.ft_chns[3] * self.decoder_ratio,
                                      self.ft_chns[2],
                                      kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups[1],
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])

        self.up3 = DeconvolutionLayer(self.ft_chns[2] * self.decoder_ratio, self.ft_chns[1],
                                      kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups[1],
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])
        self.up4 = DeconvolutionLayer(self.ft_chns[1] * self.decoder_ratio, self.ft_chns[0],
                                      kernel_size=2,
                                      dim=2, stride=2, groups=self.ft_groups[1],
                                      acti_func=self.acti_func, norm_type=self.norm_type[1])

        if (self.dropout):
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.2)
            self.drop3 = nn.Dropout(p=0.3)
            self.drop4 = nn.Dropout(p=0.4)
            self.drop5 = nn.Dropout(p=0.5)

        self.conv9 = nn.Conv2d(self.ft_chns[0] * self.decoder_ratio, self.n_class * self.ft_groups[1],
                               kernel_size=1, groups=self.ft_groups[1])

        self.out_conv1 = nn.Conv2d(
            self.ft_chns[3] * 2, self.n_class, kernel_size=1)
        self.out_conv2 = nn.Conv2d(
            self.ft_chns[2] * 2, self.n_class, kernel_size=1)
        self.out_conv3 = nn.Conv2d(
            self.ft_chns[1] * 2, self.n_class, kernel_size=1)

    def param(self):
        encoder = nn.Sequential(self.block1, self.drop1, self.down1, self.block2, self.drop2, self.down2, self.block3,
                                self.drop3, self.down3, self.block4, self.drop4, self.down4, self.block5, self.drop5)
        decoder = nn.Sequential(self.up1, self.block6, self.up2, self.block7, self.up3, self.block8, self.up4,
                                self.block9, self.conv9, self.out_conv1, self.out_conv2, self.out_conv3)
        encoder_param = sum(p.numel() for p in encoder.parameters())
        decoder_param = sum(p.numel() for p in decoder.parameters())
        return encoder_param, decoder_param

    def reshape_back(self, out, N, D):
        for i in range(len(out)):
            new_shape = [N, D] + list(out[i].shape)[1:]
            out[i] = torch.transpose(torch.reshape(out[i], new_shape), 1, 2)
        return out

    def forward(self, x):

        f1 = self.block1(x)
        if (self.dropout):
            f1 = self.drop1(f1)
        d1 = self.down1(f1)  # 96x96xft[0]

        f2 = self.block2(d1)  # 96x96xft[1]
        if (self.dropout):
            f2 = self.drop2(f2)
        d2 = self.down2(f2)  # 48x48xft[1]

        f3 = self.block3(d2)  # 48x48xft[2]
        if (self.dropout):
            f3 = self.drop3(f3)
        d3 = self.down3(f3)  # 24x24xft[2]

        f4 = self.block4(d3)  # 24x24xft[3]
        if (self.dropout):
            f4 = self.drop4(f4)
        d4 = self.down4(f4)  # 12x12xft[3]

        f5 = self.block5(d4)  # 12x12xft[4]
        if (self.dropout):
            f5 = self.drop5(f5)
            
            
        f5up = self.up1(f5)  # 24x24xft[3]

        f4cat = interleaved_concate(f4, f5up, self.shuffle_channel)
        f6 = self.block6(f4cat)
        f6up = self.up2(f6)
        f3cat = interleaved_concate(f3, f6up, self.shuffle_channel)

        f7 = self.block7(f3cat)
        f7up = self.up3(f7)

        f2cat = interleaved_concate(f2, f7up, self.shuffle_channel)
        f8 = self.block8(f2cat)
        f8up = self.up4(f8)

        f1cat = interleaved_concate(f1, f8up, self.shuffle_channel)
        f9 = self.block9(f1cat)

        output = self.conv9(f9)
        grouped_pred = torch.chunk(output, self.ft_groups[1], dim=1)
        level_pred = [self.out_conv1(f4cat),
                      self.out_conv2(f3cat),
                      self.out_conv3(f2cat)]
        return grouped_pred, level_pred

class BiNet(nn.Module):
    def __init__(self, net1,net2):
        super(BiNet, self).__init__()
        assert net1.ft_groups[0] == 1 and net1.ft_groups[1] == 1
        assert net2.ft_groups[0] == 1 and net2.ft_groups[1] == 1
        self.net1 = net1
        self.net2 = net2 
          

    def forward(self, x):
        out1,_ = self.net1(x)
        out2,_ = self.net2(x)
        out1,out2 = out1[0],out2[0]
        if(self.training):
          return out1, out2
        else:
          return [[(out1+out2)/2,],None]


class UnetEncoder(nn.Module):
    def __init__(self,in_chns, ft_chns, norm_type, ft_groups, acti_func, dropout=True) -> None:        
        super().__init__()
        #for conv block
        self.block1 = UNetBlock(in_chns, ft_chns[0], norm_type[0], ft_groups[0],
                                acti_func)
        self.block2 = UNetBlock(ft_chns[0], ft_chns[1], norm_type[0], ft_groups[0],
                                acti_func)
        self.block3 = UNetBlock(ft_chns[1], ft_chns[2], norm_type[0], ft_groups[0],
                                acti_func)
        self.block4 = UNetBlock(ft_chns[2], ft_chns[3], norm_type[0], ft_groups[0],
                                acti_func)
        self.block5 = UNetBlock(ft_chns[3], ft_chns[4], norm_type[0], ft_groups[0],
                                acti_func)
        
        #for dropout
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        
        # for downsample
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.down3 = nn.MaxPool2d(kernel_size=2)

        self.down4 = nn.MaxPool2d(kernel_size=2)
        
        
        self.forward = self.forward_dropout if dropout else self.forward_no_dropout
 
 
    def forward_dropout(self, x):
        f1 = self.drop1(self.block1(x))
        d1 = self.down1(f1)  # 96x96xft[0]

        f2 = self.drop2(self.block2(d1))  # 96x96xft[1]
        d2 = self.down2(f2)  # 48x48xft[1]

        f3 = self.drop3(self.block3(d2))  # 48x48xft[2]
        d3 = self.down3(f3)  # 24x24xft[2]

        f4 = self.drop4(self.block4(d3))  # 24x24xft[3]
        d4 = self.down4(f4)  # 12x12xft[3]

        f5 = self.drop5(self.block5(d4))  # 12x12xft[4]
        return [f1, f2, f3, f4, f5]
        
    def forward_no_dropout(self, x):
        f1 = self.block1(x)
        d1 = self.down1(f1)  # 96x96xft[0]

        f2 = self.block2(d1)  # 96x96xft[1]
        d2 = self.down2(f2)  # 48x48xft[1]

        f3 = self.block3(d2)  # 48x48xft[2]
        d3 = self.down3(f3)  # 24x24xft[2]

        f4 = self.block4(d3)  # 24x24xft[3]
        d4 = self.down4(f4)  # 12x12xft[3]

        f5 = self.block5(d4)  # 12x12xft[4]
        return [f1, f2, f3, f4, f5]
        
        
class UnetDecoder(nn.Module):
    def __init__(self, ft_chns, norm_type, ft_groups, acti_func, decoder_ratio, mode="transconv" ) -> None:
        super().__init__()
        self.up1 = DeconvolutionLayer(ft_chns[4], ft_chns[3], kernel_size=2,
                                      dim=2, stride=2, groups=ft_groups[1],mode = mode,
                                      acti_func=acti_func, norm_type=norm_type[1])

        self.up2 = DeconvolutionLayer(ft_chns[3] * decoder_ratio,
                                      ft_chns[2],mode = mode,
                                      kernel_size=2,
                                      dim=2, stride=2, groups=ft_groups[1],
                                      acti_func=acti_func, norm_type=norm_type[1])

        self.up3 = DeconvolutionLayer(ft_chns[2] * decoder_ratio,ft_chns[1],
                                      kernel_size=2,mode = mode,
                                      dim=2, stride=2, groups=ft_groups[1],
                                      acti_func=acti_func, norm_type=norm_type[1])
        self.up4 = DeconvolutionLayer(ft_chns[1] * decoder_ratio,ft_chns[0],
                                      kernel_size=2,mode = mode,
                                      dim=2, stride=2, groups=ft_groups[1],
                                      acti_func=acti_func, norm_type=norm_type[1])
        self.block6 = UNetBlock(ft_chns[3] * 2, ft_chns[3] * decoder_ratio, norm_type[0],
                                ft_groups[1],
                                acti_func)

        self.block7 = UNetBlock(ft_chns[2] * 2, ft_chns[2] * decoder_ratio,
                                norm_type[0],
                                ft_groups[1],
                                acti_func)

        self.block8 = UNetBlock(ft_chns[1] * 2, ft_chns[1] * decoder_ratio,
                                norm_type[0],
                                ft_groups[1],
                                acti_func)

        self.block9 = UNetBlock(ft_chns[0] * 2, ft_chns[0] * decoder_ratio,
                                norm_type[0],
                                ft_groups[1],
                                acti_func)

    def interleaved_concate(self ,f1, f2):
        f1_shape = list(f1.shape)
        f2_shape = list(f2.shape)
        c1 = f1_shape[1]
        c2 = f2_shape[1]

        f1_shape_new = f1_shape[:1] + [c1, 1] + f1_shape[2:]
        f2_shape_new = f2_shape[:1] + [c2, 1] + f2_shape[2:]

        f1_reshape = torch.reshape(f1, f1_shape_new)
        f2_reshape = torch.reshape(f2, f2_shape_new)
        output = torch.cat((f1_reshape, f2_reshape), dim=2)

        out_shape = f1_shape[:1] + [c1 + c2] + f1_shape[2:]
        output = torch.reshape(output, out_shape)
        return output
    def forward(self, x):
        f1, f2, f3, f4, f5 = x
        f5up = self.up1(f5)  # 24x24xft[3]

        f4cat = self.interleaved_concate(f4, f5up)
        f6 = self.block6(f4cat)
        f6up = self.up2(f6)
        
        f3cat = self.interleaved_concate(f3, f6up)
        f7 = self.block7(f3cat)
        f7up = self.up3(f7)

        f2cat = self.interleaved_concate(f2, f7up)
        f8 = self.block8(f2cat)
        f8up = self.up4(f8)

        f1cat = self.interleaved_concate(f1, f8up)
        f9 = self.block9(f1cat)

        return f9, [f4cat, f3cat, f2cat]

class Unet(nn.Module):     
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.ft_chns = [self.params['ndf'] *
                        int(pow(2, i)) for i in range(0, 5)]
        self.in_chns = self.params['in_chns']
        self.ft_groups = self.params['feature_grps']
        self.norm_type = self.params['norm_type']
        self.n_class = self.params['class_num']
        self.acti_func = get_acti_func(self.params['acti_func'], self.params)
        self.dropout = self.params['dropout']
        self.decoder_ratio = self.params.get("decoder_ratio", 1)
        self.up_mode = self.params.get("up_mode", "transconv") 
        
        
        self.encoder = UnetEncoder(self.in_chns, self.ft_chns, self.norm_type, self.ft_groups, self.acti_func)
        self.decoder = UnetDecoder(self.ft_chns, self.norm_type, self.ft_groups, self.acti_func,self.decoder_ratio,mode=self.up_mode)
        
        self.conv9 = nn.Conv2d(self.ft_chns[0] * self.decoder_ratio, self.n_class * self.ft_groups[1],
                               kernel_size=1, groups=self.ft_groups[1])
        self.out_conv1 = nn.Conv2d(
            self.ft_chns[3] * 2, self.n_class, kernel_size=1)
        self.out_conv2 = nn.Conv2d(
            self.ft_chns[2] * 2, self.n_class, kernel_size=1)
        self.out_conv3 = nn.Conv2d(
            self.ft_chns[1] * 2, self.n_class, kernel_size=1)
        
    def count_params(self):
        #encoder params
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        #decoder params
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total = encoder_params + decoder_params +self.conv9.bias.numel()+ self.conv9.weight.numel()+self.out_conv1.weight.numel()+self.out_conv2.weight.numel()+self.out_conv3.weight.numel() + self.out_conv1.bias.numel() + self.out_conv2.bias.numel() + self.out_conv3.bias.numel()
        return total
    def forward(self, x):
        feature = self.encoder(x)
        output, dfeature = self.decoder(feature)
        pred = self.conv9(output)
        grouped_pred = torch.chunk(pred, self.ft_groups[1], dim=1)
        
        f4cat, f3cat, f2cat = dfeature
        level_pred = [self.out_conv1(f4cat),
                      self.out_conv2(f3cat),
                      self.out_conv3(f2cat)]
        return grouped_pred, level_pred
         
        
class MCNet(Unet):
    def __init__(self, params):
        super().__init__(params)
        self.decoder2 = UnetDecoder(self.ft_chns, self.norm_type, self.ft_groups, self.acti_func,self.decoder_ratio,mode="nearest")
        self.decoder3 = UnetDecoder(self.ft_chns, self.norm_type, self.ft_groups, self.acti_func,self.decoder_ratio,mode="bilinear")
        self.conv10 = nn.Conv2d(self.ft_chns[0] * self.decoder_ratio, self.n_class * self.ft_groups[1],
                               kernel_size=1, groups=self.ft_groups[1])
        self.conv11 = nn.Conv2d(self.ft_chns[0] * self.decoder_ratio, self.n_class * self.ft_groups[1],
                               kernel_size=1, groups=self.ft_groups[1]) 
       
    def count_params(self):
        ed = super().count_params()
        d2 = sum(p.numel() for p in self.decoder2.parameters())
        d3 = sum(p.numel() for p in self.decoder3.parameters())
        total = ed + d2+d3+self.conv10.bias.numel()+ self.conv10.weight.numel()+self.conv11.weight.numel()+self.conv11.bias.numel()
        return total
        
    def forward(self, x):
        feature = self.encoder(x)
        output, _ = self.decoder(feature)
        pred1 = self.conv9(output)
        if not self.training:
            return pred1
        o2,_ = self.decoder2(feature)
        o3,_ = self.decoder3(feature)
        pred2 = self.conv10(o2)
        pred3 = self.conv11(o3)
        
        return pred1, pred2, pred3
        
        
        
if __name__ == '__main__':
    import sys
    import os
    import yaml

    def read_yml(filepath):
        assert os.path.exists(filepath), "file not exist"
        with open(filepath, "r", encoding="utf8") as fp:
            config = yaml.load(fp, yaml.FullLoader)
        return config

    assert len(sys.argv) - 1 == 1, f"{sys.argv}"

    cfg = read_yml(sys.argv[1])
    model = MGUNet(cfg["Network"])
    print(model.param())
    print(model)
    
    model = Unet(cfg["Network"])