import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


class fcn8s_zhou(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=True):
        super(fcn8s_zhou, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d, size_average=False)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, self.n_classes, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore4 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 4, stride=2, bias=False
            )
            self.upscore8 = nn.ConvTranspose2d(
                self.n_classes, self.n_classes, 16, stride=8, bias=False
            )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        if self.learned_billinear:
            upscore2 = self.upscore2(score)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5: 5 + upscore2.size()[2], 5: 5 + upscore2.size()[3]
            ]
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)

            score_pool3c = self.score_pool3(conv3)[
                :, :, 9: 9 + upscore_pool4.size()[2], 9: 9 + upscore_pool4.size()[3]
            ]

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31: 31 + x.size()[2], 31: 31 + x.size()[3]
            ]
            return out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            score_pool3 = self.score_pool3(conv3)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3
            out = F.upsample(score, x.size()[2:])

        return out

    def init_pretrained_params(self, pretrained):

        def find_conv_layer(model, b):
            for j in range(b+1, len(model)):
                if model[j]['type'] == 'Convolution' or model[j]['type'] == 'Deconvolution':
                    return j, model[j]

        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        index = 0
        for conv_block in blocks:
            for l1 in conv_block:
                if isinstance(l1, nn.Conv2d):
                    index, l2 = find_conv_layer(pretrained, index)
                    assert l1.weight.size() == l2['weights'][0].shape
                    assert l1.bias.size() == l2['weights'][1].shape
                    l1.weight.data = torch.from_numpy(l2['weights'][0])
                    l1.bias.data = torch.from_numpy(l2['weights'][1])

        for l1 in self.classifier:
            if isinstance(l1, nn.Conv2d):
                index, l2 = find_conv_layer(pretrained, index)
                assert l1.weight.size() == l2['weights'][0].shape
                assert l1.bias.size() == l2['weights'][1].shape
                l1.weight.data = torch.from_numpy(l2['weights'][0])
                l1.bias.data = torch.from_numpy(l2['weights'][1])

        index, ln = find_conv_layer(pretrained, index)
        assert self.upscore2.weight.size() == ln['weights'][0].shape
        self.upscore2.weight.data = torch.from_numpy(ln['weights'][0])

        index, ln = find_conv_layer(pretrained, index)
        assert self.score_pool4.weight.size() == ln['weights'][0].shape
        assert self.score_pool4.bias.size() == ln['weights'][1].shape
        self.score_pool4.weight.data = torch.from_numpy(ln['weights'][0])
        self.score_pool4.bias.data = torch.from_numpy(ln['weights'][1])

        index, ln = find_conv_layer(pretrained, index)
        assert self.upscore4.weight.size() == ln['weights'][0].shape
        self.upscore4.weight.data = torch.from_numpy(ln['weights'][0])

        index, ln = find_conv_layer(pretrained, index)
        assert self.score_pool3.weight.size() == ln['weights'][0].shape
        assert self.score_pool3.bias.size() == ln['weights'][1].shape
        self.score_pool3.weight.data = torch.from_numpy(ln['weights'][0])
        self.score_pool3.bias.data = torch.from_numpy(ln['weights'][1])

        index, ln = find_conv_layer(pretrained, index)
        assert self.upscore8.weight.size() == ln['weights'][0].shape
        self.upscore8.weight.data = torch.from_numpy(ln['weights'][0])


pretrained = np.array(np.load('FCN_weights.npy'))

model = fcn8s_zhou(n_classes=151)
model.init_pretrained_params(pretrained)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\n", model.state_dict()[param_tensor].size())

torch.save(model.state_dict(), "pretrained_FCN8_state.pt")
