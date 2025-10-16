
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np
import time
# TFEDNC3copy

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]  # 从输入的三维张量 x 中提取每个第一维元素的偶数索引位置的第二维元素，最终返回一个新的三维张量

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, in_planes, splitting=True,
                 kernel = 5, dropout=0.5, groups = 1, hidden_size = 1, INN = True,
                 fft_ratio=0.5, use_fft=True, fft_mode='combined',
                 middle_layer_count = 1, activation = 'GELU', dropout1d_v = 0.1, 
                 visualize=False, data_name = 'ETTh2'):  # fft_mode='combined'
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
       
        self.data_name = data_name
        self.fft_ratio = fft_ratio
        self.use_fft = use_fft
        self.fft_mode = fft_mode

        self.middle_layer_count = middle_layer_count

        # 定义激活函数映射
        activation_mapping = {
            'ELU': nn.ELU,
            'Tanh': nn.Tanh,
            'GELU': nn.GELU,
            'SELU': nn.SELU,
            'Softplus': nn.Softplus
        }

        # 根据激活函数名称字符串获取激活函数类并实例化
        try:
            activation_class = activation_mapping[activation]
            self.activation = activation_class()
        except KeyError:
            raise ValueError(f"不支持的激活函数 '{activation}'。请选择 GELU, SELU, ELU, Tanh 或 Softplus。")
        
        self.dropout1d_v = dropout1d_v

        self.visualize = visualize
        self.gate_activations = []  # 改为直接存储张量
        self.fft_vis_data = {'raw': None, 'enhanced': None}

        if self.use_fft:  
            
            self.freq_processor = self.build_freq_processor(in_planes=in_planes)
                
                # 频域门控机制
            self.freq_gate = nn.Sequential(
                    nn.Conv1d(in_planes, in_planes//4, 1),
                    self.activation,
                    nn.Conv1d(in_planes//4, 2 if self.fft_mode == 'combined' else 1, 1),
                    nn.Sigmoid()
            )
              

        if self.kernel_size % 2 == 0:
            pad_l = self.dilation * (self.kernel_size - 2) // 2 + 1 #by default: stride==1 
            pad_r = self.dilation * (self.kernel_size) // 2 + 1 #by default: stride==1 

        else:
            pad_l = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
            pad_r = self.dilation * (self.kernel_size - 1) // 2 + 1
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = self.hidden_size
        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= self.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= self.groups),
            nn.Tanh()   
            
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    
    def build_freq_processor(self, in_planes):
        layers = []
        # 固定的第一层
        layers.extend([
            nn.Conv1d(in_planes, in_planes * 2, 3, padding=1),
            self.activation,
            nn.Dropout1d(self.dropout1d_v)
        ])

        # 动态的中间层
        for _ in range(self.middle_layer_count):
            layers.extend([
                nn.Conv1d(in_planes * 2, in_planes * 2, 3, padding=1),
                self.activation,
                nn.Dropout1d(self.dropout1d_v)
            ])

        # 固定的最后一层
        layers.append(nn.Conv1d(in_planes * 2, in_planes, 1))

        return nn.Sequential(*layers)
    

    def fft_enhance(self, x):

        """频域增强模块"""
        B, D, L = x.shape  # [Batch, Dimension, Length]
        
        # FFT变换
        x_freq = torch.fft.rfft(x, dim=-1)
        
        # 动态频率截断
        F = x_freq.shape[-1]

        keep_freq = max(1, int(F * self.fft_ratio))
        x_freq = x_freq[..., :keep_freq]
        
        # 频域处理
        mag = torch.abs(x_freq) + 1e-8
        phase = torch.angle(x_freq)
        
        # 特征拼接
        freq_feat = torch.stack([mag, phase], dim=-1)  # [B, D, K, 2]
        freq_feat = freq_feat.view(B, D, -1)  # [B, D, 2K]
        
        # 频域变换
        processed = self.freq_processor(freq_feat)  # [B, D, K']
        
        # 门控机制
        if self.fft_mode == 'combined':
            gate = self.freq_gate(processed)  # [B, 2, K']
            mag_gate, phase_gate = gate.chunk(2, dim=1)
        elif self.fft_mode == 'split':
            gate = self.freq_gate(processed)
            mag_gate = gate
            phase_gate = 1 - gate  # 互补门控

        # ===== 新增：存储门控数据用于可视化 =====
        if self.visualize and gate is not None:
            with torch.no_grad():
                # 取第一个batch和channel的门控数据
                gate_sample = gate[0].detach().cpu().numpy().astype(np.float32)  # [2, K'] 或 [1, K']
                self.gate_activations.append(gate_sample)

        # 重建复数信号
        real = processed * mag_gate
        imag = processed * phase_gate
        new_freq = torch.complex(real, imag)
        
        # 逆FFT
        enhanced = torch.fft.irfft(new_freq, n=L, dim=-1)


        if self.visualize:
            # ===== 安全初始化 =====
            if not isinstance(self.fft_vis_data, dict):
                self.fft_vis_data = {'raw': None, 'enhanced': None}

            # 确保使用深拷贝避免引用问题
            with torch.no_grad():
                if self.fft_vis_data['raw'] is None:
                    raw_signal = x[0,0].clone().detach().cpu()
                    raw_fft = torch.fft.rfft(raw_signal)
                    self.fft_vis_data['raw'] = (
                        torch.abs(raw_fft).clone().numpy(), 
                        torch.angle(raw_fft).clone().numpy()
                    )

                enhanced_signal = enhanced[0,0].clone().detach().cpu()
                enhanced_fft = torch.fft.rfft(enhanced_signal)
                self.fft_vis_data['enhanced'] = (
                    torch.abs(enhanced_fft).clone().numpy(),
                    torch.angle(enhanced_fft).clone().numpy()
                )

        return enhanced

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
           
            # ------------- 新增频域增强 ----------------
            if self.use_fft:
                if self.fft_mode == 'combined':
                    # 合并分支处理
                    x_combined = torch.cat([x_even, x_odd], dim=1)  
                    x_combined = x_combined.permute(0, 2, 1)  
                    # 频域增强
                    enhanced = self.fft_enhance(x_combined)

                    enhanced = enhanced.permute(0, 2, 1)  
                    # 拆分为原始分支
                    x_even = enhanced[:, ::2, :]
                    x_odd = enhanced[:, 1::2, :]
                elif self.fft_mode == 'split':
                   # 新增分路处理方式
                    x_even = x_even.permute(0, 2, 1)
                    x_odd = x_odd.permute(0, 2, 1)

                    # 分别处理奇偶分支
                    x_even = self.fft_enhance(x_even).permute(0, 2, 1)
                    x_odd = self.fft_enhance(x_odd).permute(0, 2, 1)
            
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, in_planes, kernel, dropout, groups , hidden_size, INN, 
                 fft_ratio, use_fft, fft_mode, 
                 middle_layer_count, activation, dropout1d_v, visualize, data_name):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(in_planes = in_planes, splitting=True,
                 kernel = kernel, dropout=dropout, groups = groups, hidden_size = hidden_size, INN = INN, 
                 fft_ratio = fft_ratio, use_fft = use_fft, fft_mode = fft_mode,
                 middle_layer_count=middle_layer_count, activation=activation, dropout1d_v=dropout1d_v, visualize=visualize, data_name = data_name)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)
        return (x_even_update, x_odd_update)

class LevelSCINet(nn.Module):
    def __init__(self,in_planes, kernel_size, dropout, groups, hidden_size, INN, fft_ratio, use_fft,  fft_mode, middle_layer_count, activation, dropout1d_v, visualize, data_name):
        super(LevelSCINet, self).__init__()
        self.interact = InteractorLevel(in_planes= in_planes, kernel = kernel_size, dropout = dropout, groups =groups , hidden_size = hidden_size, INN = INN, 
                                        fft_ratio=fft_ratio, use_fft=use_fft, fft_mode=fft_mode, 
                                        middle_layer_count=middle_layer_count, activation=activation, dropout1d_v=dropout1d_v, visualize=visualize,
                                        data_name=data_name)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, T, D odd: B, T, D

class SCINet_Tree(nn.Module):
    def __init__(self, in_planes, current_level, kernel_size, dropout, groups, hidden_size, INN, 
                 fft_ratio, use_fft, fft_mode, middle_layer_count, activation, dropout1d_v, visualize, data_name):
        super().__init__()
        self.current_level = current_level


        self.workingblock = LevelSCINet(
            in_planes = in_planes,
            kernel_size = kernel_size,
            dropout = dropout,
            groups= groups,
            hidden_size = hidden_size,
            INN = INN,
            fft_ratio=fft_ratio,
            use_fft=use_fft,
            fft_mode=fft_mode,
            middle_layer_count=middle_layer_count,
            activation=activation,
            dropout1d_v=dropout1d_v,
            visualize=visualize,
            data_name=data_name)


        if current_level!=0:
            self.SCINet_Tree_odd=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN, fft_ratio, use_fft, fft_mode, middle_layer_count, activation, dropout1d_v, visualize, data_name)
            self.SCINet_Tree_even=SCINet_Tree(in_planes, current_level-1, kernel_size, dropout, groups, hidden_size, INN, fft_ratio, use_fft, fft_mode, middle_layer_count, activation, dropout1d_v, visualize, data_name)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_level ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))

class EncoderTree(nn.Module):
    def __init__(self, in_planes,  num_levels, kernel_size, dropout, groups, hidden_size, INN, fft_ratio, use_fft, fft_mode, middle_layer_count, activation, dropout1d_v, visualize, data_name, keep_dim = True):
        super().__init__()
        #self.keep_dim = keep_dim
        self.levels=num_levels
        self.SCINet_Tree = SCINet_Tree(
            in_planes = in_planes,
            current_level = num_levels-1,
            kernel_size = kernel_size,
            dropout =dropout ,
            groups = groups,
            hidden_size = hidden_size,
            INN = INN,
            fft_ratio = fft_ratio, 
            use_fft = use_fft,
            fft_mode=fft_mode,
            middle_layer_count=middle_layer_count,
            activation=activation,
            dropout1d_v=dropout1d_v,
            visualize=visualize,
            data_name=data_name)
        
    def forward(self, x):
        y = x
        x= self.SCINet_Tree(x)
        if y.shape != x.shape:
            print("shape deffrence!")
        # if self.keep_dim:
        #     return x.reshape(x.shape[0], x.shape[1], -1)
        return x

class SCINet(nn.Module):
    def __init__(self, output_len, input_len, input_dim = 9, hid_size = 1, num_stacks = 1,
                num_levels = 3, num_decoder_layer = 1, concat_len = 0, groups = 1, kernel = 5, dropout = 0.5,
                single_step_output_One = 0, input_len_seg = 0, positionalE = False, modified = True, RIN=False,
                fft_ratio=0.6, use_fft=True, fft_mode='combined', middle_layer_count=1, activation='GELU', dropout1d_v=0.1, visualize=True, data_name='PEMS03'):
        super(SCINet, self).__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_size = hid_size
        self.num_levels = num_levels
        self.groups = groups
        self.modified = modified
        self.kernel_size = kernel
        self.dropout = dropout
        self.single_step_output_One = single_step_output_One
        self.concat_len = concat_len
        self.pe = positionalE
        self.RIN=RIN
        self.num_decoder_layer = num_decoder_layer


        self.fft_ratio = fft_ratio
        self.use_fft = use_fft
        self.fft_mode = fft_mode
        
        self.middle_layer_count = middle_layer_count
        self.activation = activation
        self.dropout1d_v = dropout1d_v
        
        self.visualize = visualize

        self.data_name = data_name


        self.blocks1 = EncoderTree(
            in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified,
            fft_ratio = self.fft_ratio,
            use_fft = self.use_fft,
            fft_mode=self.fft_mode,

            middle_layer_count=self.middle_layer_count,
            activation=self.activation,
            dropout1d_v=self.dropout1d_v,
            visualize=self.visualize,
            data_name=self.data_name)

        if num_stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(
                in_planes=self.input_dim,
            num_levels = self.num_levels,
            kernel_size = self.kernel_size,
            dropout = self.dropout,
            groups = self.groups,
            hidden_size = self.hidden_size,
            INN =  modified,
            fft_ratio = self.fft_ratio,
            use_fft = self.use_fft,
            fft_mode=self.fft_mode,

            middle_layer_count=self.middle_layer_count,
            activation=self.activation,
            dropout1d_v=self.dropout1d_v,
            visualize=self.visualize,
            data_name=self.data_name)

        self.stacks = num_stacks

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        self.projection1 = nn.Conv1d(self.input_len, self.output_len, kernel_size=1, stride=1, bias=False)
        self.div_projection = nn.ModuleList()
        self.overlap_len = self.input_len//4
        self.div_len = self.input_len//6

        if self.num_decoder_layer > 1:
            self.projection1 = nn.Linear(self.input_len, self.output_len)  # original
        
            for layer_idx in range(self.num_decoder_layer-1):
                div_projection = nn.ModuleList()
                for i in range(6):
                    lens = min(i*self.div_len+self.overlap_len,self.input_len) - i*self.div_len
                    div_projection.append(nn.Linear(lens, self.div_len))

                self.div_projection.append(div_projection)

        if self.single_step_output_One: # only output the N_th timestep.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, 1,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            if self.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(self.concat_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(self.input_len + self.output_len, self.output_len,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.pe_hidden_size = input_dim
        if self.pe_hidden_size % 2 == 1:
            self.pe_hidden_size += 1
    
        num_timescales = self.pe_hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

        ### RIN Parameters ###
        if self.RIN:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, input_dim))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_dim))
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.pe_hidden_size % 2))
        signal = signal.view(1, max_length, self.pe_hidden_size)
    
        return signal


    def forward(self, x):
       
        assert self.input_len % (np.power(2, self.num_levels)) == 0 # evenly divided the input length into two parts. (e.g., 32 -> 16 -> 8 -> 4 for 3 levels)
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        ### activated when RIN flag is set ###
        if self.RIN:
            print('/// RIN ACTIVATED ///\r',end='')
            means = x.mean(1, keepdim=True).detach()
            #mean
            x = x - means
            #var
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev
            # affine
            # print(x.shape,self.affine_weight.shape,self.affine_bias.shape)
            x = x * self.affine_weight + self.affine_bias

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        if self.num_decoder_layer == 1:
            x = self.projection1(x)
        else:
            x = x.permute(0,2,1)
            for div_projection in self.div_projection:
                output = torch.zeros(x.shape,dtype=x.dtype).cuda()
                for i, div_layer in enumerate(div_projection):
                    div_x = x[:,:,i*self.div_len:min(i*self.div_len+self.overlap_len,self.input_len)]
                    output[:,:,i*self.div_len:(i+1)*self.div_len] = div_layer(div_x)
                x = output
            x = self.projection1(x)
            x = x.permute(0,2,1)

        if self.stacks == 1:
            ### reverse RIN ###
            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            
            ### Reverse RIN ###
            if self.RIN:
                MidOutPut = MidOutPut - self.affine_bias
                MidOutPut = MidOutPut / (self.affine_weight + 1e-10)
                MidOutPut = MidOutPut * stdev
                MidOutPut = MidOutPut + means

            if self.RIN:
                x = x - self.affine_bias
                x = x / (self.affine_weight + 1e-10)
                x = x * stdev
                x = x + means

            return x, MidOutPut


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=96)
    parser.add_argument('--horizon', type=int, default=12)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=1)

    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)

    parser.add_argument('--single_step_output_One', type=int, default=0)

    args = parser.parse_args()

    model = SCINet(output_len = args.horizon, input_len= args.window_size, input_dim = 9, hid_size = args.hidden_size, num_stacks = 1,
                num_levels = 3, concat_len = 0, groups = args.groups, kernel = args.kernel, dropout = args.dropout,
                 single_step_output_One = args.single_step_output_One, positionalE =  args.positionalEcoding, modified = True).cuda()
    x = torch.randn(32, 96, 9).cuda()
    y = model(x)
    print(y.shape)
