import torch
import torch.nn as nn
import torch.nn.functional as F


def mem_update(operator, mem, input_spk, cfg, leak=0):
    impulse = operator(input_spk)
    mem = mem + impulse + leak
    # +1 spike
    output_spk = (mem >= cfg['threshold']).float()
    # -1 spike
    # if not ifrelu:
    #     output_spk += -1 * (mem <= -1 * self.threshold).float()
    # Reset
    mem = mem - output_spk * cfg['threshold']
    # Ban updates until....
    # if refrac_end is not None:
    #     refrac_end = torch.abs(output_spk) * (t + self.refractory_t)
    return mem, output_spk


class SpkGenerator(nn.Module):
    def __init__(self,max_rate):
        super(SpkGenerator, self).__init__()
        self.mode = 'ANN'  # or "SNN"
        self.dt = 0.001  # second
        self.refractory_t = 0
        self.max_rate = max_rate
        self.device = None
        assert(max_rate<=1000)
        self.rescale_factor = 1.0 / (self.dt * self.max_rate)

        self.type = 'Analog'
        self.spk_cnt = None

        self.forward_scale = 1.0

    def ann_forward(self, x):
        return x

    def ann_forward_scale(self, x):
        return self.forward_scale * self.ann_forward(x)

    def forward(self,x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o

    def snn_reset(self):
        self.spk_cnt = None

    def snn_forward(self,x):
        if self.spk_cnt is None:
            self.spk_cnt = torch.zeros(x.size(),device=self.device)
        spk = (torch.rand(x.size(),device=self.device) * self.rescale_factor <= self.ann_forward_scale(x)).float()
        #print(spk.size(),self.spk_cnt.size())

        if self.type == 'Analog':
            self.spk_cnt += self.ann_forward_scale(x)
            return x
        else:
            self.spk_cnt += spk
            return spk


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.mode = 'ANN' # or "SNN"

        self.device = None
        self.cfg = {'threshold':1.0}
        self.mem_post = None
        self.spk_cnt = None

        self.forward_scale = 1.0

    def ann_forward(self,x):
        return F.linear(x, self.weight, self.bias)

    def ann_forward_scale(self,x):
        return self.forward_scale * self.ann_forward(x)

    def snn_reset(self):
        self.mem_post = None
        self.spk_cnt = None

    def snn_forward(self,input_spk):
        assert(len(input_spk.size())==2)
        if self.mem_post is None or self.spk_cnt is None:
            self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
            self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
        self.mem_post, output_spk = mem_update(self.ann_forward_scale,self.mem_post,input_spk,self.cfg)
        self.spk_cnt += output_spk
        return output_spk

    def forward(self,x):
        if self.mode == "ANN":
            o = F.relu(self.ann_forward(x))
        else:
            o = self.snn_forward(x)
        return o


class LinearwithBN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearwithBN, self).__init__(in_features, out_features, bias)
        self.mode = 'ANN' # or "SNN"
        self.BN = nn.BatchNorm1d(out_features)

        self.device = None
        self.cfg = {'threshold':1.0}
        self.mem_post = None
        self.spk_cnt = None

        self.weight_nobn = None
        self.bias_nobn = None
        self.nobn = False

        self.forward_scale = 1.0

    def transBatchNorm(self):
        bn_std = torch.sqrt(self.BN.running_var)
        if self.bias is not None:
            self.weight_nobn = self.weight * self.BN.weight.view(-1, 1) / bn_std.view(-1, 1)
            self.bias_nobn = (self.bias - self.BN.running_mean.view(-1)) * self.BN.weight.view(-1) / bn_std.view(-1) + self.BN.bias.view(-1)
        else:
            self.weight_nobn = self.weight * self.BN.weight.view(-1, 1) / bn_std.view(-1, 1)
            self.bias_nobn = (torch.zeros_like(self.BN.running_mean.view(-1)) - self.BN.running_mean.view(-1)) * self.BN.weight.view(-1) / bn_std.view(-1) + self.BN.bias.view(-1)
        self.nobn = True

    def ann_forward(self,x):
        x = F.linear(x, self.weight, self.bias)
        x = self.BN(x)
        return x

    def ann_forward_nobn(self,x):
        x = self.forward_scale * F.linear(x, self.weight_nobn, self.bias_nobn)
        return x

    def snn_reset(self):
        self.mem_post = None
        self.spk_cnt = None

    def snn_forward(self,input_spk):
        if not self.nobn:
            assert(len(input_spk.size())==2)
            if self.mem_post is None or self.spk_cnt is None:
                self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
                self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
            self.mem_post, output_spk = mem_update(self.ann_forward,self.mem_post,input_spk,self.cfg)
            self.spk_cnt += output_spk
        else:
            if self.mem_post is None or self.spk_cnt is None:
                self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
                self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), device=self.device)
            self.mem_post, output_spk = mem_update(self.ann_forward_nobn,self.mem_post,input_spk,self.cfg)
            self.spk_cnt += output_spk
        return output_spk

    def forward(self,x):
        if self.mode == "ANN":
            o = F.relu(self.ann_forward(x))
        else:
            o = self.snn_forward(x)
        return o


class Conv2d(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.mode = 'ANN' # or "SNN"

        self.device = None
        self.cfg = {'threshold':1.0}
        self.mem_post = None
        self.spk_cnt = None

        self.forward_scale = 1.0

    def ann_forward(self,x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def ann_forward_scale(self,x):
        return self.forward_scale * self.ann_forward(x)

    def snn_reset(self):
        self.mem_post = None
        self.spk_cnt = None

    def snn_forward(self,input_spk):
        assert(len(input_spk.size())==4)
        if self.mem_post is None or self.spk_cnt is None:
            out_h = (input_spk.size(2) - self.kernel_size[0] + 2*self.padding[0]) // self.stride[0]+1
            out_w = (input_spk.size(3) - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
            self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0),out_h,out_w,device=self.device)
            self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), out_h, out_w, device=self.device)
        self.mem_post, output_spk = mem_update(self.ann_forward_scale,self.mem_post,input_spk,self.cfg)
        self.spk_cnt += output_spk
        return output_spk

    def forward(self,x):
        if self.mode == "ANN":
            o = F.relu(self.ann_forward(x))
        else:
            o = self.snn_forward(x)
        return o


class Conv2dwithBN(nn.Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2dwithBN, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.BN = nn.BatchNorm2d(out_channels)
        self.mode = 'ANN' # or "SNN"

        self.device = None
        self.cfg = {'threshold':1.0}
        self.mem_post = None
        self.spk_cnt = None

        self.weight_nobn = None
        self.bias_nobn = None
        self.nobn = False

        self.forward_scale = 1.0

    def transBatchNorm(self):
        bn_std = torch.sqrt(self.BN.running_var)
        if self.bias is not None:
            self.weight_nobn = self.weight * self.BN.weight.view(-1,1,1,1) / bn_std.view(-1,1,1,1)
            self.bias_nobn = (self.bias - self.BN.running_mean.view(-1)) * self.BN.weight.view(-1) / bn_std.view(-1) + self.BN.bias.view(-1)
        else:
            self.weight_nobn = self.weight * self.BN.weight.view(-1,1,1,1) / bn_std.view(-1,1,1,1)
            self.bias_nobn = (torch.zeros_like(self.BN.running_mean.view(-1)) - self.BN.running_mean.view(-1)) * self.BN.weight.view(-1) / bn_std.view(-1) + self.BN.bias.view(-1)
        self.nobn = True

    def ann_forward(self,x):
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        x = self.BN(x)
        return x

    def ann_forward_nobn(self,x):
        x = self.forward_scale * F.conv2d(x, self.weight_nobn, self.bias_nobn, self.stride,
                     self.padding, self.dilation, self.groups)
        return x

    def snn_reset(self):
        self.mem_post = None
        self.spk_cnt = None

    def snn_forward(self,input_spk):
        if not self.nobn:
            assert(len(input_spk.size())==4)
            if self.mem_post is None or self.spk_cnt is None:
                out_h = (input_spk.size(2) - self.kernel_size[0] + 2*self.padding[0]) // self.stride[0]+1
                out_w = (input_spk.size(3) - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
                self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0),out_h,out_w,device=self.device)
                self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), out_h, out_w, device=self.device)
            self.mem_post, output_spk = mem_update(self.ann_forward,self.mem_post,input_spk,self.cfg)
            self.spk_cnt += output_spk
        else:
            if self.mem_post is None or self.spk_cnt is None:
                out_h = (input_spk.size(2) - self.kernel_size[0] + 2*self.padding[0]) // self.stride[0]+1
                out_w = (input_spk.size(3) - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1
                self.mem_post = torch.zeros(input_spk.size(0), self.weight.size(0),out_h,out_w,device=self.device)
                self.spk_cnt = torch.zeros(input_spk.size(0), self.weight.size(0), out_h, out_w, device=self.device)
            self.mem_post, output_spk = mem_update(self.ann_forward_nobn,self.mem_post,input_spk,self.cfg)
            self.spk_cnt += output_spk
        return output_spk

    def forward(self,x):
        if self.mode == "ANN":
            o = F.relu(self.ann_forward(x))
        else:
            o = self.snn_forward(x)
        return o


class AvgPool2d(nn.Module):
    def __init__(self,kernel_size, stride=2, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(AvgPool2d, self).__init__()
        self.mode = 'ANN' # or "SNN"
        self.kernel_size = kernel_size
        #self.weight = torch.ones(1,1,kernel_size,kernel_size) / (kernel_size*kernel_size)
        self.stride = stride
        self.padding = padding
        self.forward_scale = 1.0

        self.device = None
        self.cfg = {'threshold':1.0}
        self.mem_post = None
        self.spk_cnt = None

    def ann_forward(self,x):
        return F.avg_pool2d(x,self.kernel_size, self.stride, self.padding)

    def ann_forward_scale(self,x):
        return self.forward_scale * self.ann_forward(x)

    def snn_reset(self):
        self.mem_post = None
        self.spk_cnt = None

    def snn_forward(self,input_spk):
        # history version
        # assert(len(input_spk.size())==4)
        # if self.mem_post is None or self.spk_cnt is None:
        #     out_h = (input_spk.size(2) - self.kernel_size + 2*self.padding) // self.stride+1
        #     out_w = (input_spk.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
        #     self.mem_post = self.spk_cnt = torch.zeros(input_spk.size(0), input_spk.size(1),out_h,out_w,device=self.device)
        # self.mem_post, output_spk = mem_update(self.ann_forward_scale,self.mem_post,input_spk,self.cfg)
        # #print(self.spk_cnt.size(),output_spk.size())
        # self.spk_cnt += output_spk
        # #print(output_spk)
        # #return output_spk

        assert(len(input_spk.size()) == 4)
        if self.spk_cnt is None:
            out_h = (input_spk.size(2) - self.kernel_size + 2 * self.padding) // self.stride + 1
            out_w = (input_spk.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
            self.spk_cnt = torch.zeros(input_spk.size(0), input_spk.size(1), out_h, out_w, device=self.device)
        output_spk = self.ann_forward_scale(input_spk)
        self.spk_cnt += output_spk
        return output_spk

    def forward(self,x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o


class MaxPool2d(nn.Module):
    def __init__(self,kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.mode = 'ANN'  # or "SNN"
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        #self.weight = torch.ones(1, 1, kernel_size, kernel_size)
        self.forward_scale = 1.0
        #self.type = 'Spatial-Avg' # or 'Max-Gate'
        self.type = 'Max-Gate' # or 'Spatial-Avg'

        self.device = None
        self.cfg = {'threshold': 1.0}
        self.mem_pre = None # running mean
        self.spk_cnt = None
        self.history_momentum = None

    def ann_forward(self,x):
        return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,False)

    def snn_reset(self):
        self.mem_pre = None
        self.spk_cnt = None
        self.mem_post = None

    def spatial_avg(self,x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

    def snn_forward(self,input_spk):
        if self.type == 'Max-Gate':
            assert (len(input_spk.size()) == 4)
            if self.mem_pre is None or self.spk_cnt is None:
                if self.stride is None:
                    self.stride = self.kernel_size
                out_h = (input_spk.size(2) - self.kernel_size + 2 * self.padding) // self.stride + 1
                out_w = (input_spk.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
                self.mem_pre = torch.zeros(input_spk.size(),device=self.device)
                self.spk_cnt = torch.zeros(input_spk.size(0), input_spk.size(1),out_h,out_w,device=self.device)
            if self.history_momentum is not None:
                self.mem_pre = self.mem_pre * self.history_momentum + (1-self.history_momentum)*input_spk
            else:
                self.mem_pre += input_spk
                #print(self.mem_pre)
            (output,ind) = F.max_pool2d(self.mem_pre, self.kernel_size, self.stride,
                                self.padding, self.dilation, self.ceil_mode,True)
            #print(ind)
            #print(self.mem_pre)
            unpooloutput = F.max_unpool2d(output, ind, self.kernel_size, self.stride,self.padding,self.mem_pre.size())
            indice = (unpooloutput != 0.0).float()
            #print(input_spk)
            gated_spk = input_spk * indice

            #print(gated_spk)
            # channel = input_spk.size(1)
            # weight = self.weight.expand(channel, channel, self.kernel_size, self.kernel_size).to(self.device)
            # output_spk = F.conv2d(gated_spk, weight, None, self.stride,
            #                 self.padding)
            output_spk = F.max_pool2d(gated_spk, self.kernel_size, self.stride,
                                                  self.padding)
            self.spk_cnt += output_spk

        elif self.type == 'Spatial-Avg':
            if self.mem_post is None or self.spk_cnt is None:
                out_h = (input_spk.size(2) - self.kernel_size + 2 * self.padding) // self.stride + 1
                out_w = (input_spk.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
                self.mem_post = self.spk_cnt = torch.zeros(input_spk.size(0), input_spk.size(1), out_h, out_w,
                                                           device=self.device)
            self.mem_post, output_spk = mem_update(self.spatial_avg, self.mem_post, input_spk, self.cfg)
            # print(self.spk_cnt.size(),output_spk.size())
            output_spk = output_spk * self.forward_scale # TODO: verify true or not
            self.spk_cnt += output_spk
        return output_spk

    def forward(self, x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o

'''

class MaxPool2d(nn.Module):
    def __init__(self,kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.mode = 'ANN'  # or "SNN"
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        #self.weight = torch.ones(1, 1, kernel_size, kernel_size)
        self.forward_scale = 1.0
        #self.type = 'Spatial-Avg' # or 'Max-Gate'
        self.type = 'Max-Gate' # or 'Spatial-Avg'

        self.device = None
        self.cfg = {'threshold': 1.0}
        self.spk_cnt = None
        self.mem_pre = None

    def ann_forward(self,x):
        return F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,False)

    def ann_forward_scale(self,x):
        return self.forward_scale * F.max_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,False)

    def snn_reset(self):
        self.spk_cnt = None
        self.mem_pre = None

    def snn_forward(self,input_spk):
        assert (len(input_spk.size()) == 4)
        #if self.type == 'Max-Gate':

        if self.mem_pre is None or self.spk_cnt is None:
            if self.stride is None:
                self.stride = self.kernel_size
            out_h = (input_spk.size(2) - self.kernel_size + 2 * self.padding) // self.stride + 1
            out_w = (input_spk.size(3) - self.kernel_size + 2 * self.padding) // self.stride + 1
            self.mem_pre = torch.zeros(input_spk.size(),device=self.device)

            self.spk_cnt = torch.zeros(input_spk.size(0), input_spk.size(1),out_h,out_w,device=self.device)
        self.mem_pre += input_spk
        output_spk = self.max_gate(input_spk, self.mem_pre) * self.forward_scale
        self.spk_cnt += output_spk
        return output_spk

    def forward(self, x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o

    def max_gate(self, input_spk, sum_spk):
        (n,c,h,w) = input_spk.size()
        h_o = (h - self.kernel_size + 2 * self.padding) // self.stride + 1
        w_o = (w - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_spk = torch.zeros(n, c,h_o,w_o,device=self.device)

        spikes_t = torch.zeros(n,c,h + self.padding * 2, w + self.padding * 2)
        sum_spikes_t = torch.zeros(n,c,h + self.padding * 2, w + self.padding * 2)
        spikes_t[:,:,self.padding: self.padding + h, self.padding: self.padding + w]=input_spk
        sum_spikes_t[:,:,self.padding: self.padding + h, self.padding: self.padding + w]=sum_spk
        for j in range(h_o):
            for i in range(w_o):
                line1 = sum_spikes_t[:,:,j * self.stride:j * self.stride + self.kernel_size,
                i * self.stride: i * self.stride + self.kernel_size].reshape(n*c,-1).t()
                line2 = spikes_t[:, :, j * self.stride:j * self.stride + self.kernel_size,
                        i * self.stride: i * self.stride + self.kernel_size].reshape(n*c,-1).t()
                # print(line1)
                # print(line2)
                _,I = torch.max(line1,dim=0)
                # print(I)
                temp = torch.zeros(n*c)
                for k in range(c*n):
                    temp[k] = line2[I[k],k]
                temp = temp.reshape(n,c)
                output_spk[:,:,j,i] = temp
        # print(input_spk)
        # print(sum_spk)
        # print(output_spk)
        # exit(-2)
        return output_spk
'''

class Resize1d(nn.Module):
    def __init__(self, outsize):
        super(Resize1d, self).__init__()
        self.mode = 'ANN' # or "SNN"
        self.outsize = outsize
        self.device = None

    def forward(self,x):
        x = x.view(-1,self.outsize)
        return x


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()
        self.mode = 'ANN'  # or "SNN"
        self.device = None
        self.cfg = {'threshold': 1.0}
        self.spk_cnt = None
        self.forward_scale = 1.0

    def ann_forward(self,x):
        return x

    def ann_forward_scale(self,x):
        return self.forward_scale * self.ann_forward(x)

    def snn_reset(self):
        self.spk_cnt = None

    def snn_forward(self, input_spk):
        if self.spk_cnt is None:
            self.spk_cnt = torch.zeros(input_spk.size(),device=self.device)
        self.spk_cnt += self.ann_forward_scale(input_spk)
        return self.ann_forward_scale(input_spk)

    def forward(self, x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o


class Softmax(nn.Module): # TODO: with bug, acc loss
    def __init__(self, dim=None):
        super(Softmax, self).__init__()
        self.dim = dim
        self.mode = 'ANN' # or "SNN"

        self.device = None
        self.cfg = {'threshold':1.0}
        self.spk_cnt = None
        self.t = None

        self.forward_scale = 1.0

    def ann_forward(self,x):
        return F.softmax(x,dim=self.dim)

    def ann_forward_scale(self,x):
        return F.softmax(x*self.forward_scale,dim=self.dim)

    def snn_reset(self):
        self.mem_pre = None
        self.spk_cnt = None
        self.t = None

    def snn_forward(self, input_spk):
        if self.t is None or self.spk_cnt is None or self.mem_pre is None:
            self.t = 0
            self.mem_pre = torch.zeros(input_spk.size(),device=self.device)
            self.spk_cnt = torch.zeros(input_spk.size(),device=self.device)
        self.t += 1
        self.mem_pre += input_spk
        u = self.mem_pre * (1.0 / self.t)
        prob = self.ann_forward_scale(u)
        # print(self.t)
        # print(self.mem_pre)
        # print(u)
        # print(prob)
        spk = (torch.rand(input_spk.size(), device=self.device) <= prob).float()
        self.spk_cnt += spk
        # print(self.spk_cnt)
        return spk

    def forward(self, x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o


class Dropout(nn.Dropout):
    def __init__(self,p=0.5, inplace=False):
        super(Dropout, self).__init__(p, inplace)
        self.mode = 'ANN'  # or "SNN"
        self.device = None
        self.cfg = {'threshold': 1.0}
        self.mem_post = None
        self.spk_cnt = None
        self.forward_scale = 1.0

    def ann_forward(self,x):
        if len(x.size())==4:
            x = F.dropout2d(x, self.p, self.training, self.inplace)
        elif len(x.size())==2:
            x = F.dropout(x, self.p, self.training, self.inplace)
        return x

    def ann_forward_scale(self,x):
        return self.forward_scale * self.ann_forward(x)

    def snn_reset(self):
        self.spk_cnt = None

    def forward(self,x):
        if self.mode == "ANN":
            o = self.ann_forward(x)
        else:
            o = self.snn_forward(x)
        return o

    def snn_forward(self, input_spk):
        state = self.training
        self.training = False
        if self.spk_cnt is None:
            self.spk_cnt = torch.zeros(input_spk.size(), device=self.device)
        output_spk = self.ann_forward_scale(input_spk)
        self.spk_cnt += output_spk
        self.training = state
        return output_spk


if __name__ == "__main__":
    pass


    torch.random.manual_seed(0)
    k = torch.Tensor([1.0])
    k.requires_grad = True
    x = torch.rand(1,3,32)
    y = torch.rand(1,3,32)
    lossfunc = nn.MSELoss()
    loss = lossfunc(k*x,y)
    loss.backward()
    print(loss,k.grad)

    import functools
    loss = torch.pow(torch.norm(k*x-y,p=2) ,2)/ (3*32)
    grad = 1.0/ functools.reduce(lambda m, n: m * n, x.size()) *2*k*torch.sum(x*(k*x-y))
    print(loss,grad)
    print(torch.std(k*x-y))
    exit(-2)
    # l = Linear(10,20)
    # print(l.snn_forward(torch.sign(l.weight)))
    #
    # c = Conv2d(3,9,7,stride=1,padding=1)
    # print(c.snn_forward(torch.sign(torch.rand(11,3,28,28))))

    i = torch.rand(3,1,10,8)
    # print(i)
    # print(F.avg_pool2d(i,kernel_size=3,stride=2,padding=1))
    # c = nn.Conv2d(1,1,3,2,1,bias=False)
    # c.weight.data = torch.ones(c.weight.size())*1.0/(3*3)
    # print(c.weight.size())
    #
    # print(c(i))
    # a = AvgPool2d(3,2,1)
    # print(a(i))

    m = MaxPool2d(2,2)
    x = m.snn_forward((i-0.9>0).float())

    print(x)

    # input = torch.rand(4,3,28,28)
    # cn = Conv2dwithBN(3,7,5,2,2)
    # cn.transBatchNorm()
    # print(cn(input))

