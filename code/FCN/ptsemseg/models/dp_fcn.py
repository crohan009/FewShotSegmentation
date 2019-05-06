import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.misc as m

from ptsemseg.models.utils import get_upsampling_weight
from ptsemseg.loss import cross_entropy2d

ttype = torch.cuda.FloatTensor;

# FCN8s
class dp_fcn8s(nn.Module):
    def __init__(self, params, n_classes=6, learned_billinear=True):
        super(dp_fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.rule = params['rule']       # --rule hebb
        self.n_classes = n_classes       # n_classes == 151
        self.params = params
        self.eta = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)  # Everyone has the same eta
        self.w = torch.nn.Parameter((.01 * torch.randn(200, params['nbclasses'])).cuda(), requires_grad=True)
        #self.w = torch.nn.Parameter((.01 * torch.randn(params['nbclasses'], params['nbclasses'])) )
        if params['alpha'] == 'free':     # --alpha free 
            self.alpha = torch.nn.Parameter((.01 * torch.rand(200, params['nbclasses'])).cuda(), requires_grad=True) # Note: rand rather than randn (all positive)
        elif params['alpha'] == 'yoked':
            self.alpha = torch.nn.Parameter((.01 * torch.ones(1)).cuda(), requires_grad=True)
        else :
            raise ValueError("Must select a value for alpha ('free' or 'yoked')")
        #self.loss = functools.partial(cross_entropy2d, size_average=False)

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

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 200, 1),
        )

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2, bias=False)
            self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4, stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16, stride=8, bias=False)

        self.softmax = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(
                    get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                )   
      
      
    def scale_and_make_one_hot(self, device, labels, dim=(17, 17), C=151): 
        ''' 
        Converts an integer label torch.autograd.Variable to a one-hot Variable. 

        Parameters 
        ---------- 
        labels : torch.autograd.Variable of torch.cuda.LongTensor 
           N x 1 x H x W, where N is batch size.  
           Each value is an integer representing correct classification. 
        C : integer.  
           number of classes in labels. 

        Returns 
        ------- 
        target : torch.autograd.Variable of torch.cuda.FloatTensor 
           N x C x H x W, where C is class number. One-hot encoded. 
        ''' 

        lbl = labels.cpu().numpy().squeeze().astype(float)
        lbl = m.imresize(lbl, dim, "nearest", mode="F")
        lbl = lbl.astype(int)
        labels = torch.from_numpy(lbl).long()
        labels = labels.view(1,1, dim[0], dim[1]).to(device)

        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_() 
        target = one_hot.scatter_(1, labels.data, 1) 
        target = Variable(target) 
        return target 
      
      
      
    def forward(self, x, y, hebb, device, test_mode=False):
        #print("x.shape",x.shape)
        conv1 = self.conv_block1(x)
        #print("conv1.shape",conv1.shape)
        conv2 = self.conv_block2(conv1)
        #print("conv2.shape",conv2.shape)
        conv3 = self.conv_block3(conv2)
        #print("conv3.shape",conv3.shape)
        conv4 = self.conv_block4(conv3)
        #print("conv4.shape",conv4.shape)
        conv5 = self.conv_block5(conv4)
        #print("conv5.shape",conv5.shape)

        conv6 = self.conv_block6(conv5)
        
        conv6_shape = conv6.shape
        #print("conv6_shape = ", conv6_shape)
        
        activin = conv6.view(-1, conv6_shape[1])
        #print("---> activin.shape = ", activin.shape)
        
        #print("---(DP)---")
        
        #----------- PLASTIC LAYER --------
        
        if not test_mode:
          lbl = self.scale_and_make_one_hot(device, y, dim=(conv6_shape[2], conv6_shape[3]), C=self.n_classes)
          #print("lbl.shape = ", lbl.shape)
          
        
        if self.params['alpha'] == 'free':
            activ = activin.mm( self.w + torch.mul(self.alpha, hebb)) 
            if not test_mode:
                activ = activ + 1000.0 * lbl.view(-1, self.n_classes) 
        elif self.params['alpha'] == 'yoked':
            activ = activin.mm( self.w + self.alpha * hebb) 
            if not test_mode:
                activ = activ + 1000.0 * lbl.view(-1, self.n_classes)
                
        #print("---> activ.shape = ", activ.shape)
        
        #----------------------------------
        
        activout = activ.view(conv6_shape[0], -1, conv6_shape[2], conv6_shape[3])
        activout_soft = self.softmax(activout).view(-1, self.n_classes) 
        
        #print("activout.shape = ", activout.shape)
        #print("activout_soft.shape = ", activout_soft.shape)
        #print("---------")
        
        #-------- HEBB WEIGHT UPDATE -------
        
        if self.rule == 'hebb':
            x_in_x_out = torch.bmm(activin.unsqueeze(2), activout_soft.unsqueeze(1))
            #print("x_in_x_out.shape = ", x_in_x_out.shape)
            #print("x_in_x_out[0].shape = ", x_in_x_out[0].shape)
            for i in range(activin.shape[0]):
                hebb = (1 - self.eta) * hebb + self.eta * x_in_x_out[i] 
                
        elif self.rule == 'clip':
            x_in_x_out = torch.bmm(activin.unsqueeze(2), activout_soft.unsqueeze(1))
            for i in range(activin.shape[0]):
                hebb =  hebb + self.eta * x_in_x_out[i]
                torch.clamp(hebb, min=-1.0, max=1.0)
        
        elif self.rule == 'oja':
            for i in range(activin.shape[0]):
                hebb = hebb + self.eta * torch.mul((activin[i].unsqueeze(1) - torch.mul(hebb ,
                                                                                        activout_soft[i].unsqueeze(0))) ,
                                                   activout_soft[i].unsqueeze(0))  
        else:
           raise ValueError("Rule '{}' not implemented. Please select one learning rule ('hebb' or 'oja')".format(self.rule))
            
        #----------------------------------

        if self.learned_billinear:
            upscore2 = self.upscore2(activout)
            #print("upscore2.shape = ", upscore2.shape)
            score_pool4c = self.score_pool4(conv4)[
                :, :, 5 : 5 + upscore2.size()[2], 5 : 5 + upscore2.size()[3]
            ]
            #print("score_pool4c.shape = ", score_pool4c.shape)
            upscore_pool4 = self.upscore4(upscore2 + score_pool4c)
            #print("upscore_pool4.shape = ", upscore_pool4.shape)
            score_pool3c = self.score_pool3(conv3)[
                :, :, 9 : 9 + upscore_pool4.size()[2], 9 : 9 + upscore_pool4.size()[3]
            ]
            #print("score_pool3c.shape = ", score_pool3c.shape)

            out = self.upscore8(score_pool3c + upscore_pool4)[
                :, :, 31 : 31 + x.size()[2], 31 : 31 + x.size()[3]
            ]
            out = out.contiguous()

        else:
            score_pool4 = self.score_pool4(conv4)
            #print("score_pool4.shape = ", score_pool4.shape)
            score_pool3 = self.score_pool3(conv3)
            #print("score_pool3.shape = ", score_pool3.shape)
            score = F.upsample(score, score_pool4.size()[2:])
            #print("score.shape = ", score.shape)
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            #print("score.shape = ", score.shape)
            score += score_pool3
            out = F.upsample(score, x.size()[2:])
            
        return out, hebb

    def load_pretrained_weights(self, weight_path):
        print("Loading weights at {}".format(weight_path))

        weights = torch.load(weight_path)
        net_params = self.state_dict()

        mismatch_lst = ["conv_block6.6.weight", "conv_block6.6.bias", "score_pool4.weight", "score_pool4.bias",
                        "score_pool3.weight", "score_pool3.bias", "upscore2.weight", "upscore4.weight", "upscore8.weight", 
                        "classifier.6.weight", "classifier.6.bias"]
      
        for param in weights['model_state'].keys():

            if (param[7:] in mismatch_lst):
                print("--Skipped param: '{}'--".format(param))
                continue

            elif (param[7:] in net_params.keys()):
                net_params[param[7:]] = weights['model_state'][param]
                print("net_params[{}] <--- weights['model_state'][{}]".format(param[7:], param))

            elif param[7:].split('.')[0] == "classifier":
                s = param[7:].split('.')
                s[0] = "conv_block6"
                s = ".".join(s)
                net_params[s] = weights['model_state'][param]
                print("net_params[{}] <--- weights['model_state'][{}]".format(s, param))

        self.load_state_dict(net_params) 
        print("Done")

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.conv_block6[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.conv_block6[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.conv_block6[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

    def initialZeroHebb(self):
        #return Variable(torch.zeros(self.params['plastsize'], self.params['nbclasses']).type(ttype))
        return Variable(torch.zeros(200, self.params['nbclasses']).type(ttype))



if __name__ == "__main__":

    defaultParams = {
        'activ': 'tanh',    # 'tanh' or 'selu'
        #'plastsize': 200,
        'rule': 'clip',     # 'hebb' or 'oja' or 'clip'
        'alpha': 'free',    # 'free' of 'yoked' (if the latter, alpha is a single scalar learned parameter, shared across all connection)
        'steplr': 1e6,      # How often should we change the learning rate?
        'nbclasses': 151,
        'gamma': .666,      # The annealing factor of learning rate decay for Adam
        'flare': 0,         # Whether or not the ConvNet has more features in higher channels
        'nbshots': 1,       # Number of 'shots' in the few-shots learning
        'prestime': 1,
        'nbf' : 64,         # Number of features. 128 is better (unsurprisingly) but we keep 64 for fair comparison with other reports
        'prestimetest': 1,
        'ipd': 0,           # Inter-presentation delay 
        'imgsize': 31,    
        'nbiter': 5000000,  
        'lr': 3e-5, 
        'test_every': 500,
        'save_every': 10000,
        'rngseed':0
    }

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    net = fcn8s(params=defaultParams, n_classes=151).to(device)
    #print(net)
    hebb = net.initialZeroHebb().to(device)

    img_tensor = torch.randn(1,3,512,512).float().to(device)
    target_tensor = torch.randint(0,151, size=(1,1,512,512)).long().to(device)

    #print("------")
    #print("img_tensor.shape = ", img_tensor.shape)
    #print("target_tensor.shape = ", target_tensor.shape)
    #print("------")
    #print("hebb.shape = ", hebb.shape)
    #print("------")

    out = net(img_tensor, target_tensor, hebb, device, test_mode=True)

    #print("out.shape = {}\n\n\n".format(out.shape))

