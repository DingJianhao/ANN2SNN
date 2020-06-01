import torch
import torch.nn as nn
from DeepANN2SNN.modules import *
import matplotlib.pyplot as plt
import numpy as np
import functools
from DeepANN2SNN.utils import *

class VGG16(nn.Module):
    def __init__(self, time_step=50):
        super(VGG16, self).__init__()
        self.spk_gen = SpkGenerator(1000)
        self.features = self._make_features()
        self.resize1d = Resize1d(512)
        self.classifier1 = self._make_classifier()
        self.classifier2 = Linear(512,10)
        self.mode = "ANN"
        self.set_working_mode(self.mode)
        self.time_step = time_step


    def _make_features(self):
        layers = []
        layers += [Conv2dwithBN(3,64,3,1,1)]
        layers += [Dropout(0.3)]
        layers += [Conv2dwithBN(64,64,3,1,1)]
        layers += [MaxPool2d(2,2)]

        layers += [Conv2dwithBN(64, 128, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(128, 128, 3, 1, 1)]
        layers += [MaxPool2d(2, 2)]

        layers += [Conv2dwithBN(128, 256, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(256, 256, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(256, 256, 3, 1, 1)]
        layers += [MaxPool2d(2, 2)]

        layers += [Conv2dwithBN(256, 512, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(512, 512, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(512, 512, 3, 1, 1)]
        layers += [MaxPool2d(2, 2)]

        layers += [Conv2dwithBN(512, 512, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(512, 512, 3, 1, 1)]
        layers += [Dropout(0.4)]
        layers += [Conv2dwithBN(512, 512, 3, 1, 1)]
        layers += [MaxPool2d(2, 2)]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        layers += [Dropout(0.5)]
        layers += [LinearwithBN(512,512)]
        layers += [Dropout(0.5)]
        return nn.Sequential(*layers)

    def forward(self,x):
        if self.mode == "ANN":
            out = self.spk_gen(x)
            out = self.features(out)
            out = self.resize1d(out)
            out = self.classifier1(out)
            out = self.classifier2(out)
        else:
            self.reset_snn()
            for t in range(self.time_step):
                spk_input = self.spk_gen(x)
                spk_out = self.features(spk_input)
                spk_out = self.resize1d(spk_out)
                spk_out = self.classifier1(spk_out)
                spk_out = self.classifier2(spk_out)
            out = self.classifier2.spk_cnt / self.time_step
        return out
    def set_working_mode(self,mode):
        self.mode = mode
        for n,m in self.named_modules():
            if hasattr(m,"mode"):
                m.mode = self.mode

    def to(self, device):
        self.device = device
        super().to(device)
        for n, m in self.named_modules():
            m.device = device
        return self

    def reset_snn(self):
        for n,m in self.named_modules():
            if hasattr(m,"snn_reset"):
                m.snn_reset()

    def visualize_forward(self, x, target):
        u1 = [32, 32, 32,32, 16, 16, 16,16, 8, 8,8,8, 8, 8, 4,4,4, 4, 4, 4, 2,2,2, 2, 2, 2, 1, 32,32,32,32, 2]
        u2 = [32, 32, 32,32, 16, 16, 16,16, 8, 8,8,8, 8, 8, 4,4,4, 4, 4, 4, 2,2,2, 2, 2, 2, 1, 16,16,16,16, 5]

        self.eval()
        tmp_mode = self.mode
        self.set_working_mode('ANN')
        x = x.view(1, 3, 32, 32)
        out = x
        idx = 0
        for n, m in self.named_modules():
            if m.__class__.__name__ in forward_list:
                idx += 1
                # print(n, m.__class__.__name__)
                # out = m(out)
                if hasattr(m, 'ann_forward_nobn'):
                    out = F.relu(m.ann_forward_nobn(out))
                elif hasattr(m, 'ann_forward_scale'):
                    out = m.ann_forward_scale(out)
                elif hasattr(m, 'ann_forward'):
                    out = F.relu(m.ann_forward(out))
                else:
                    out = m(out)
                # print(out.size(),idx-1)
                if hasattr(m, 'spk_cnt'):
                    if len(out.size()) == 4:
                        show(out[:, 0].view(u1[idx - 1], u2[idx - 1]).detach().cpu().numpy(),
                             'VGG16_%d_%s%d_ANN.jpg' % (idx, m.__class__.__name__, idx))
                    else:
                        show(out.view(u1[idx - 1], u2[idx - 1]).detach().cpu().numpy(),
                             'VGG16_%d_layer%d_ANN.jpg' % (idx, idx))

        self.set_working_mode('SNN')
        print('SNN test:')
        self.reset_snn()
        out = x
        lastmodule = None
        for t in range(self.time_step):
            for n, m in self.named_modules():
                if m.__class__.__name__ == 'SpkGenerator':
                    out = m(x)
                elif m.__class__.__name__ in forward_list:
                    lastmodule = m
                    # print(n, m.__class__.__name__)
                    out = m(out)
        out = lastmodule.spk_cnt / self.time_step
        idx = 0
        for n, m in self.named_modules():
            if m.__class__.__name__ in forward_list:
                idx += 1
                if hasattr(m, 'spk_cnt'):
                    if len(m.spk_cnt.size()) == 4:
                        show(m.spk_cnt[:, 0].view(u1[idx - 1], u2[idx - 1]).detach().cpu().numpy(),
                             'VGG16_%d_%s%d_SNN.jpg' % (idx, m.__class__.__name__, idx))
                    else:
                        show(m.spk_cnt.view(u1[idx - 1], u2[idx - 1]).detach().cpu().numpy(),
                             'VGG16_%d_layer%d_SNN.jpg' % (idx, idx))
        self.set_working_mode(tmp_mode)
        pass

    def finegrained_tune(self,x,target):
        self.eval()
        self.reset_snn()
        tmp_mode = self.mode
        x = x.view(1, 3, 32, 32)
        history_activation = []
        history_spk = []
        now_activation = []
        now_spk = []

        idx = 0
        for n, m in self.named_modules():
            if m.__class__.__name__ == 'SpkGenerator':
                print(m.__class__.__name__)
                idx += 1
                self.set_working_mode('ANN')
                out = x
                out = m(out)
                history_activation = [out.clone()]
                self.set_working_mode('SNN')
                for t in range(self.time_step):
                    out = m(out)
                    history_spk.append(out.clone())
            elif m.__class__.__name__ in forward_list:
                idx += 1
                print(m.__class__.__name__)

                Epochs = 10
                for e in range(Epochs):
                    print('e',e)

                    now_activation = []
                    now_spk = []

                    self.set_working_mode('ANN')
                    out = m(history_activation[0])
                    now_activation.append(out.clone())
                    self.set_working_mode('SNN')
                    for t in range(self.time_step):
                        out = m(history_spk[t])
                        now_spk.append(out.clone())
                    a1 = torch.sum(torch.cat(now_spk), dim=0) / self.time_step
                    a2 = now_activation[0].view(a1.size())
                    mask = (a1>=0.05).float() * (a2>=1e-5).float() * (a1<=0.9).float() * (a2<=1.0).float()

                    scale = torch.sum(mask*a2/(a1+1e-5))/torch.sum(mask).item()
                    manual_scale_v2(m, scale)
                    print('scale',scale)

                now_activation = []
                now_spk = []
                self.set_working_mode('ANN')
                out = m(history_activation[0])
                now_activation.append(out.clone())
                self.set_working_mode('SNN')
                for t in range(self.time_step):
                    out = m(history_spk[t])
                    now_spk.append(out.clone())
                    # if hasattr(m,'forward_scale'):
                    #     a1 = torch.sum(torch.cat(now_spk), dim=0) / self.time_step
                    #     a2 = now_activation[0].view(a1.size())
                    #     loss = torch.pow(torch.norm(m.forward_scale*a1-a2,p=2),2)/ \
                    #            functools.reduce(lambda mm, nn: mm * mm, a1.size())
                    #     print(loss)
                    #     grad = 1.0/ functools.reduce(lambda mm, nn: mm * mm, a1.size())\
                    #            *2*m.forward_scale*torch.sum(a1*(m.forward_scale*a1-a2))
                    #     print('now',m.forward_scale)
                    #     manual_scale_v2(m, m.forward_scale-10000*grad.item())
                    #     print('after', m.forward_scale)
                    #
                    #     print('a1',torch.min(a1),'-',torch.max(a1))
                    #     print('a2',torch.min(a2),'-',torch.max(a2))
                history_activation = now_activation
                history_spk = now_spk

                if m.__class__.__name__ in forward_list:
                    print(m.__class__.__name__)
                    sum_spk = torch.cat(history_spk)
                    print((torch.sum(sum_spk,dim=0)/self.time_step)[1,20:25,20:25])
                    print(torch.max(torch.sum(sum_spk,dim=0)/self.time_step),torch.min(torch.sum(sum_spk,dim=0)/self.time_step))
                    print(history_activation[0][0,1,20:25,20:25])
                    print(torch.max(history_activation[0]),torch.min(history_activation[0]))
                    plt.imshow((torch.sum(sum_spk,dim=0)/self.time_step)[1,20:25,20:25].cpu().numpy())
                    plt.show()

                    plt.imshow(history_activation[0][0,1,20:25,20:25].cpu().numpy())
                    plt.show()
                if idx==4:
                    break




        self.set_working_mode(tmp_mode)
    '''
    SpkGenerator torch.Size([100, 3, 32, 32])
Conv2dwithBN torch.Size([100, 64, 32, 32])
Dropout torch.Size([100, 64, 32, 32])
Conv2dwithBN torch.Size([100, 64, 32, 32])
MaxPool2d torch.Size([100, 64, 16, 16])
Conv2dwithBN torch.Size([100, 128, 16, 16])
Dropout torch.Size([100, 128, 16, 16])
Conv2dwithBN torch.Size([100, 128, 16, 16])
MaxPool2d torch.Size([100, 128, 8, 8])
Conv2dwithBN torch.Size([100, 256, 8, 8])
Dropout torch.Size([100, 256, 8, 8])
Conv2dwithBN torch.Size([100, 256, 8, 8])
Dropout torch.Size([100, 256, 8, 8])
Conv2dwithBN torch.Size([100, 256, 8, 8])
MaxPool2d torch.Size([100, 256, 4, 4])
Conv2dwithBN torch.Size([100, 512, 4, 4])
Dropout torch.Size([100, 512, 4, 4])
Conv2dwithBN torch.Size([100, 512, 4, 4])
Dropout torch.Size([100, 512, 4, 4])
Conv2dwithBN torch.Size([100, 512, 4, 4])
MaxPool2d torch.Size([100, 512, 2, 2])
Conv2dwithBN torch.Size([100, 512, 2, 2])
Dropout torch.Size([100, 512, 2, 2])
Conv2dwithBN torch.Size([100, 512, 2, 2])
Dropout torch.Size([100, 512, 2, 2])
Conv2dwithBN torch.Size([100, 512, 2, 2])
MaxPool2d torch.Size([100, 512, 1, 1])
Resize1d torch.Size([100, 512])
Dropout torch.Size([100, 512])
LinearwithBN torch.Size([100, 512])
Dropout torch.Size([100, 512])
Linear torch.Size([100, 10])
    '''
