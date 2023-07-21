"parts of the code were adapted from https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MST_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list, dropout=0.5, w_max=0, offline_mode=False):
        super(MST_TCN2, self).__init__()
        self.w_max = w_max
        self.offline_mode = offline_mode
        self.num_R = num_R
        self.PG = MT_Prediction_Generation(
            num_layers_PG, num_f_maps, dim, num_classes_list, dropout)

        if num_R > 0:
            self.Rs = nn.ModuleList([copy.deepcopy(MT_Refinement(num_layers_R, num_f_maps, sum(
                num_classes_list), num_classes_list, dropout)) for s in range(num_R)])

    def forward(self, x, *args):
        outputs = []
        outs, _ = self.PG(x, self.w_max, self.offline_mode)
        for out in outs:
            outputs.append(out.unsqueeze(0))
        out = torch.cat(outs, 1)

        if self.num_R > 0:
            for R in self.Rs:
                outs = R(F.softmax(out, dim=1),
                         self.w_max, self.offline_mode)
                out = torch.cat(outs, 1)
                for i, output in enumerate(outputs):
                    outputs[i] = torch.cat(
                        (output, outs[i].unsqueeze(0)), dim=0)

        return outputs


class MT_Prediction_Generation_many_heads(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout=0.5):
        super(MT_Prediction_Generation_many_heads, self).__init__()

        self.layers_heads = self.heads_config(num_layers)
        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([copy.deepcopy(
            nn.ModuleList([copy.deepcopy(
                nn.Conv1d(num_f_maps, num_classes_list[s], 1))
                for s in range(len(num_classes_list))]))
            for i in range(len(self.layers_heads))])

    def heads_config(self, num_of_layers):
        heads = []
        for i in ([4, 7, 10]):
            if i < num_of_layers - 1:
                heads.append(i)
        heads.append(num_of_layers-1)
        return heads

    def forward(self, x, w_max, offline_mod):
        optputs = []
        outs = []
        featutes = []

        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](
                f, w_max, offline_mod), self.conv_dilated_2[i](f, w_max, offline_mod)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
            if i in self.layers_heads:
                featutes.append(f)
                index = self.layers_heads.index(i)
                for conv_out in self.heads[index]:
                    outs.append(conv_out(f))
                optputs.append(outs)
                outs = []

        return optputs, featutes


class MT_Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout):
        super(MT_Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout(dropout)

        self.conv_outs = nn.ModuleList([copy.deepcopy(
            nn.Conv1d(num_f_maps, num_classes_list[s], 1))
            for s in range(len(num_classes_list))])

    def forward(self, x, w_max, offline_mod):
        outs = []
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](
                f, w_max, offline_mod), self.conv_dilated_2[i](f, w_max, offline_mod)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in
        for conv_out in self.conv_outs:
            outs.append(conv_out(f))

        return outs, f


class MT_Refinement(nn.Module):  # refinement stage
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout=0.5):
        super(MT_Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(
            2**i, num_f_maps, num_f_maps, dropout=dropout)) for i in range(num_layers)])
        self.conv_outs = nn.ModuleList([copy.deepcopy(
            nn.Conv1d(num_f_maps, num_classes_list[s], 1))
            for s in range(len(num_classes_list))])

    def forward(self, x, w_max, offline_mode):
        outs = []
        f = self.conv_1x1(x)
        for layer in self.layers:
            f = layer(f, w_max, offline_mode)
        for conv_out in self.conv_outs:
            outs.append(conv_out(f))
        return outs


class Dilated_conv(nn.Module):
    def __init__(self, num_f_maps, karnel_size, dilation):
        super(Dilated_conv, self).__init__()
        self.dilation = dilation
        self.Dilated_conv = nn.Conv1d(
            num_f_maps, num_f_maps, karnel_size, dilation=dilation)
        # the dilation seperates the frames

    def forward(self, x, w_max, offline):
        if offline:
            out = self.Acausal_padding(x, self.dilation)
        else:
            out = self.window_padding(x, self.dilation, w_max)
        out = self.Dilated_conv(out)
        return out

    def Acausal_padding(self, input, padding_dim):
        padding = torch.zeros(
            input.shape[0], input.shape[1], padding_dim).to(input.device)
        return torch.cat((padding, input, padding), 2)

    def window_padding(self, input, padding_dim, w_max=0):
        if w_max > padding_dim:
            w_max = padding_dim
        padding_left = torch.zeros(
            input.shape[0], input.shape[1], 2*padding_dim - w_max).to(input.device)
        padding_right = torch.zeros(
            input.shape[0], input.shape[1], w_max).to(input.device)
        return torch.cat((padding_left, input, padding_right), 2)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout=0.5):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        # In the original code padding=dilation
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w_max, offline):
        if offline:
            out = self.Acausal_padding(x, self.dilation)
        else:
            out = self.window_padding(x, self.dilation, w_max)

        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

    def Acausal_padding(self, input, padding_dim):
        padding = torch.zeros(
            input.shape[0], input.shape[1], padding_dim).to(input.device)
        return torch.cat((padding, input, padding), 2)

    def window_padding(self, input, padding_dim, w_max):
        if w_max > padding_dim:
            w_max = padding_dim
        padding_left = torch.zeros(
            input.shape[0], input.shape[1], 2*padding_dim - w_max).to(input.device)
        padding_right = torch.zeros(
            input.shape[0], input.shape[1], w_max).to(input.device)
        return torch.cat((padding_left, input, padding_right), 2)
