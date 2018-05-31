import torch
import torch.nn as nn

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def unflatten(x, N=-1, C=128, H=7, W=7):
    return x.view(N, C, H, W)


class Generator1(nn.Module):
    """
        Assumptions:
           Expects x to be NCHW with C=3
           pose_target is also NCHW (C=18)
           Total input channels = 21
    """
    def __init__(self, z_num, repeat_num, hidden_num, min_fea_map_H=8):
        
        super(Generator1, self).__init__()
        
        self.repeat_num     = repeat_num
        self.hidden_num     = hidden_num
        self.z_num          = z_num
        self.min_fea_map_H  = min_fea_map_H
        		
		#ENCODER
        self.Econv1 = nn.Sequential(nn.Conv2d(21,hidden_num,3,1,1),nn.ReLU()) 
        self.Elayers = {}
        inp_channels = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num*(idx+1)
            self.Elayers[str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(inp_channels,channel_num,3,1,1),nn.ReLU())
            self.Elayers[str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),nn.ReLU())
            inp_channels     = channel_num
            if idx < repeat_num - 1:
                self.Elayers[str(idx+1)+"_3"] = nn.Sequential(nn.Conv2d(channel_num,hidden_num*(idx+2),3,2,1),nn.ReLU())
                inp_channels = hidden_num*(idx+2)

        self.Elinear1 = nn.Linear(min_fea_map_H * min_fea_map_H * channel_num, z_num)

        #DECODER
        self.Dlinear1 = nn.Linear(z_num, min_fea_map_H * min_fea_map_H * hidden_num * repeat_num) # NOTE: Different from their code (repeat_num)
        self.Dlayers = {}
        for idx in range(repeat_num):
            channel_num = hidden_num * (repeat_num - idx) * 2
            self.Dlayers[str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),nn.ReLU())
            self.Dlayers[str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),nn.ReLU())
            if idx < repeat_num - 1:
                self.Dlayers[str(idx+1)+"_3"] = nn.Sequential(nn.Conv2d(channel_num,hidden_num*(repeat_num-idx-1),3,1,1),nn.ReLU())

        self.Dconv    = nn.Conv2d(channel_num, 3, 3, 1, 1) # generates a 3 channel image
        self.upscale  = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, pose_target):
        x   = torch.cat((x,pose_target), dim=1)
        x   = self.Econv1(x)
        
        encoder_outputs = []
        # Encode x
        for idx in range(self.repeat_num):
            res = x
            x   = self.Elayers[str(idx+1)+"_1"](x)
            x   = self.Elayers[str(idx+1)+"_2"](x)
            x  += res
            encoder_outputs.append(x)
            if idx < self.repeat_num - 1:
                x = self.Elayers[str(idx+1)+"_3"](x)

        # x.shape = batch_size x hidden_num (128) x repeat_num (6) x min_fea_map_H (8) x (8)
        # FC layer
        x = self.Elinear1(flatten(x)) 

        # Decode x
        x = self.Dlinear1(x)
        x = unflatten(x, x.shape[0], self.hidden_num * self.repeat_num, self.min_fea_map_H, self.min_fea_map_H)

        for idx in range(self.repeat_num):
            x   = torch.cat((x, encoder_outputs[self.repeat_num-1-idx]), dim=1)
            res = x
            x  = self.Dlayers[str(idx+1)+"_1"](x)
            x  = self.Dlayers[str(idx+1)+"_2"](x)
            x += res
            if idx < self.repeat_num - 1:
                x = self.upscale(x)
                x = self.Dlayers[str(idx+1)+"_3"](x)
        
        # Generate image
        x = self.Dconv(x) 
        return x
            


class Generator2(nn.Module):
    """
        Assumes concatenated input of the form : x + G1
        Total input channels = 6
    """
    def __init__(self, repeat_num, hidden_num, noise_dim=64):
        super(Generator2, self).__init__()

        self.repeat_num = repeat_num
        self.hidden_num = hidden_num
        self.noise_dim  = noise_dim

        #ENCODER
        self.Econv1 = nn.Sequential(nn.Conv2d(6,hidden_num,3,1,1),nn.ReLU()) 
        self.Elayers = {}
        inp_channels = hidden_num
        for idx in range(repeat_num):
            channel_num = hidden_num*(idx+1)
            self.Elayers[str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(inp_channels,channel_num,3,1,1),nn.ReLU())
            self.Elayers[str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),nn.ReLU())
            inp_channels     = channel_num
            if idx < repeat_num - 1:
                self.Elayers[str(idx+1)+"_3"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,2,1),nn.ReLU())
        
        
        self.Dlayers = {}
        for idx in range(repeat_num):
            if idx==0:
                channel_num = hidden_num * repeat_num * 2
            else:
                channel_num = hidden_num * (repeat_num - idx + 1)
            self.Dlayers[str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(channel_num,hidden_num,3,1,1),nn.ReLU())
            self.Dlayers[str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(hidden_num,hidden_num,3,1,1),nn.ReLU())
            

        self.Dconv    = nn.Conv2d(hidden_num, 3, 3, 1, 1) # generates a 3 channel image
        self.upscale  = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.Econv1(x)

        encoder_outputs = []
        # Encode x 
        for idx in range(self.repeat_num):
            x   = self.Elayers[str(idx+1)+"_1"](x)
            x   = self.Elayers[str(idx+1)+"_2"](x)
            encoder_outputs.append(x)
            if idx < self.repeat_num - 1:
                x = self.Elayers[str(idx+1)+"_3"](x)

        # Add noise
        if self.noise_dim>0:
            noise = 2 * torch.rand_like(x) - 1
            x     = torch.cat((x,noise), dim=1)
        
        # Decode x
        for idx in range(self.repeat_num):
            x   = torch.cat((x, encoder_outputs[self.repeat_num-1-idx]), dim=1)
            x   = self.Dlayers[str(idx+1)+"_1"](x)
            x   = self.Dlayers[str(idx+1)+"_2"](x)
            if idx < self.repeat_num - 1:
                x = self.upscale(x)
        
        # Generate image
        x = self.Dconv(x) 
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, dim=64):
        super(Discriminator, self).__init__()

        self.conv1  = nn.Sequential(nn.Conv2d(input_dim, dim, 5, 2, 2),nn.LeakyReLU())
        self.conv2  = nn.Sequential(nn.Conv2d(dim, dim*2, 5, 2, 2), nn.BatchNorm2d(dim*2), nn.LeakyReLU())
        self.conv3  = nn.Sequential(nn.Conv2d(dim*2, dim*4, 5, 2, 2), nn.BatchNorm2d(dim*4), nn.LeakyReLU())
        self.conv4  = nn.Sequential(nn.Conv2d(dim*4, dim*8, 5, 2, 2), nn.BatchNorm2d(dim*8), nn.LeakyReLU())
        self.conv5  = nn.Sequential(nn.Conv2d(dim*8, dim*8, 5, 2, 2), nn.BatchNorm2d(dim*8), nn.LeakyReLU())

        self.linear = nn.Linear(8*8*8*dim, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = flatten(x)
        x = self.linear(x)

        return x.view(-1)
    

