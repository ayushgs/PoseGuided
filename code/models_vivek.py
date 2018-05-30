import numpy as np
import torch
import torch.nn as nn
import torch.nn.Functional as F
import utils_wgan ##make this file

class ResBottleNeckBlock(nn.Module):
	def __init__(self,n1,n2,n3,activation_fn):
		super(ResBottleNeckBlock,self).__init__()
		self.unconnected = (n1 != n3)
		self.h0 = nn.Conv2d(n1,n3,1,1,0)  ## this is the shortcut layer in case n1 == n3
		self.h1 = nn.Sequential(nn.Conv2d(n1,n2,1,1,0),activation_fn)
		self.h2 = nn.Sequential(nn.Conv2d(n2,n2,3,1,1),activation_fn)
		self.h3 = nn.Conv2d(n2,n3,1,1,0)
		self.final = activation_fn

	def forward(x):
		if self.unconnected:
			shortcut = self.h0(x)
		else:
			shortcut = x
			x = self.h1(x)
			x = self.h2(x)
			x = self.h3(x)
		return self.final(shortcut+x)



class ResBlock(nn.Module):
	def __init__(self,n1,n2,n3,data_format,activation_fn):
		super(ResBlock,self).__init__()
		self.unconnected = (n1 != n3)
		self.h0 = nn.Conv2d(n1,n3,1,1,0)  ## this is the shortcut layer in case n1 == n3
		self.h1 = nn.Sequential(nn.Conv2d(n1,n2,3,1,1),activation_fn)
		self.h2 = nn.Conv2d(n2,n3,3,1,1)
		self.final = activation_fn

	def forward(x):
		if self.unconnected:
			shortcut = self.h0(x)
		else:
			shortcut = x
			x = self.h1(x)
			x = self.h2(x)
		return self.final(shortcut+x)

class LuNet(nn.Module):
	def __init__(self,input_H,input_W,activation_fn=nn.LeakyRelu):
		self.conv1 = nn.Sequential(nn.Conv2d(3,128,7,1,3),activation_fn)
		self.RBNB1 = ResBottleNeckBlock(128,32,128,activation_fn)
		self.maxpool1 = nn.MaxPool2d(3,2,1)

		self.RBNB2 = ResBottleNeckBlock(128,32,128,activation_fn)
		self.RBNB3 = ResBottleNeckBlock(128,32,128,activation_fn)
		self.RBNB4 = ResBottleNeckBlock(128,64,256,activation_fn)
		self.maxpool2 = nn.MaxPool2d(3,2,1)

		self.RBNB5 = ResBottleNeckBlock(256,64,256,activation_fn)
		self.RBNB6 = ResBottleNeckBlock(256,64,256,activation_fn)
		self.maxpool3 = nn.MaxPool2d(3,2,1)

		self.RBNB7 = ResBottleNeckBlock(256,64,256,activation_fn)
		self.RBNB8 = ResBottleNeckBlock(256,64,256,activation_fn)
		self.RBNB9 = ResBottleNeckBlock(256,128,512,activation_fn)
		self.maxpool4 = nn.MaxPool2d(3,2,1)

		self.RBNB10 = ResBottleNeckBlock(512,128,512,activation_fn)
		self.RBNB11 = ResBottleNeckBlock(512,128,512,activation_fn)
		self.maxpool5 = nn.MaxPool2d(3,2,1)

		self.RB1 = ResBlock(512,512,128,activation_fn)
		self.linear1 = nn.Linear(input_H*input_W/8,512)
		self.BN1 = nn.BatchNorm1d()
		self.LR1 = nn.LeakyRelu(0.2)
		
		self.linear2 = nn.Linear(512,128)

	def forward(x):
		x = self.conv1(x)
		x = self.RBNB1(x)
		x = self.maxpool1(x)

		x = self.RBNB2(x)
		x = self.RBNB3(x)
		x = self.RBNB4(x)
		x = self.maxpool2(x)

		x = self.RBNB5(x)
		x = self.RBNB6(x)
		x = self.maxpool3(x)

		x = self.RBNB7(x)
		x = self.RBNB8(x)
		x = self.RBNB9(x)
		x = self.maxpool4(x)

		x = self.RBNB10(x)
		x = self.RBNB11(x)
		x = self.maxpool5(x)

		x = self.RB1(x)
		x = x.view(x.size(0),-1)
		x = self.linear1(x)
		x = self.BN1(x)
		x = self.LR1(x)

		x = self.linear2(x)

		return x


class GeneratorCNN_Pose_UAEAfterResidual(nn.Module):
	def __init__(self,repeat_num,hidden_num,min_fea_map_H,z_num,noise_dim,activation_fn):
		
		#ENCODER
		self.Econv1 = nn.Sequential(nn.Conv2d(4,hidden_num,3,1,1),activation_fn) #the pose_target is concatenated to the channel in the input image. Hence input channels is 4.
		self.Elayers = {}
		inp_channels = hidden_num
		for idx in range(repeat_num):

			channel_num = hidden_num*(idx+1)
			self.Elayers[rep+str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(inp_channels,channel_num,3,1,1),activation_fn)
			self.Elayers[rep+str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),activation_fn)
			if idx < repeat_num - 1:
				self.Elayers[rep+str(idx+1)+"_3"] = nn.Sequential(nn.Conv2d(channel_num,hidden_num*(idx+2),3,2,1),activation_fn)
			inp_channels = hidden_num*(idx+2)

		self.Elinear1 = nn.Linear(min_fea_map_H*min_fea_map_H*channel_num/2,z_num)

		#DECODER
		self.Dlinear1 = nn.Linear(z_num+noise_dim,min_fea_map_H*min_fea_map_H*hidden_num/2)
		#reshape now
		self.Dlayers = {}
		for idx in range(repeat_num):
			channel_num = hidden_num*(repeat_num+1)
			self.Dlayers[rep+str(idx+1)+"_1"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),activation_fn)
			self.Dlayers[rep+str(idx+1)+"_2"] = nn.Sequential(nn.Conv2d(channel_num,channel_num,3,1,1),activation_fn)
			if idx < repeat_num - 1:
				self.Dlayers[rep+str(idx+1)+"_3"] = nn.Sequential(nn.Conv2d(channel_num,hidden_num*(repeat_num-idx-1),3,1,1),activation_fn)








			

















