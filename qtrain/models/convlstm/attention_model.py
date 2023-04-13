import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):

    def __init__(self, device, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMCell, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_i_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_f_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_c_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        self.conv_o_xx = nn.Conv2d(input_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_hh = nn.Conv2d(hidden_size, hidden_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False)

        torch.nn.init.xavier_normal_(self.conv_i_xx.weight)
        torch.nn.init.constant_(self.conv_i_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_f_xx.weight)
        torch.nn.init.constant_(self.conv_f_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_c_xx.weight)
        torch.nn.init.constant_(self.conv_c_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_o_xx.weight)
        torch.nn.init.constant_(self.conv_o_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_hh.weight)

    def forward(self, x, state):
        if state is None:
            state = (torch.Tensor(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3))).to(self.device),
                     torch.Tensor(torch.randn(x.size(0), x.size(1), x.size(2), x.size(3))).to(self.device))
        ht_1, ct_1 = state
        it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct


class attentionModel(nn.Module):
    def __init__(self, device, resnet_model, num_classes=14, mem_size=512, train=True):
        """
        resnet_model: resnet model from resnet.py
        """
        super(attentionModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.resNet = resnet_model
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = ConvLSTMCell(device, 512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(mem_size, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        if train:
            self.dropout = nn.Dropout(0.4)
            self.classifier = nn.Sequential(self.dropout, self.fc1, self.fc2)
        else:
            self.classifier = nn.Sequential(self.fc1, self.fc2)

    def forward(self, inputVariable):
        state = (torch.Tensor(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7))).to(self.device),
                 torch.Tensor(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7))).to(self.device))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state = self.lstm_cell(attentionFeat, state)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1