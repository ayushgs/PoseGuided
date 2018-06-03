import torch
import torch.nn as nn
from torch.utils import data
from models import *
import numpy as np
from utils import *
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
from logger import Logger
import os
from data_loader import Dataset
from torchvision.utils import save_image



class Trainer:
    def __init__(self, config):
        self.config         = config
        self.z_num          = config['z_num']
        self.hidden_num     = config['conv_hidden_num']
        self.height         = 256
        self.width          = 256
        self.channels       = 3
        self.repeat_num     = int(np.log2(self.height)) - 2
        self.noise_dim      = 0
        self.model_dir      = config['results_dir']
        self.num_epochs     = config['num_epochs']
        self.batch_size     = config['batch_size']
        self.logger         = Logger(config['log_dir'])
        self.pretrained_path= config['pretrained_path']
        self.log_every      = config['log_every']
        self.save_every     = config['save_every']
        self.print_every    = config['print_every']

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.data_loader_params  = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': 6}

        if config['use_cuda']:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        if config['train']:
            self.dataset   = Dataset(**get_split('train'))
            self.generator = data.DataLoader(self.dataset, **self.data_loader_params)
            self.n_samples = get_split('train')['total_data']
        else:
            self.dataset   = Dataset(**get_split('test'))
            self.generator = data.DataLoader(self.dataset, **self.data_loader_params)
            self.n_samples = get_split('train')['total_data']



    def _gan_loss(self, real, fake):
        gen_cost  =  generator_loss(fake)
        disc_cost =  0.5 * discriminator_loss(real, fake)

        return gen_cost, disc_cost

    def loss(self):
        #Generate coarse image
        G1      = self.Gen1(self.x, self.pose_target)

        #Generate Diffmap by concatenating G1 and image
        Diffmap = self.Gen2(torch.cat((G1, self.x), dim=1))
        G2      = G1 + Diffmap

        # Denormalize images to be in range 0-255
        self.G1 = denorm_img(G1).detach()
        self.G2 = denorm_img(G2)
        self.G  = self.G2.detach()
        self.DiffMap = denorm_img(Diffmap)

        # Feed to Discriminator
        triplet = torch.cat((self.x_target, G2, self.x), dim=0)
        self.Dz = self.D(triplet)

        # Split scores for loss functions
        D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = torch.split(self.Dz, 3, dim=0)[0]

        # Positive
        D_z_pos = D_z_pos_x_target

        # Negative
        D_z_neg = torch.Tensor((D_z_neg_g2, D_z_neg_x)).to(device=self.device)

        # Losses
        self.g1_loss = torch.mean(torch.abs(G1-self.x_target))
        self.g2_loss, self.d_loss = self._gan_loss(D_z_pos, D_z_neg)
        self.PoseMaskLoss = torch.mean(torch.abs(G2 - self.x_target) * (self.mask_target))
        self.L1Loss2 = torch.mean(torch.abs(G2 - self.x_target)) + self.PoseMaskLoss
        self.g2_loss += self.L1Loss2 * 50

    def _set_optimizers(self):
        self.g1_solver  = torch.optim.Adam(self.Gen1.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        self.g2_solver  = torch.optim.Adam(self.Gen2.parameters(), lr = 2e-5, betas=(0.5, 0.999))
        self.d_solver   = torch.optim.Adam(self.D.parameters(), lr = 2e-5, betas=(0.5, 0.999))

    def _init_net(self):
        """Initializes the models for training/testing
        """
        if self.pretrained_path is None:
            self.Gen1 = Generator1(self.z_num, self.repeat_num, self.hidden_num, device=self.device).to(device=self.device)
            self.Gen2 = Generator2(self.repeat_num, self.hidden_num, self.noise_dim, device=self.device).to(device=self.device)
            self.D    = Discriminator(device=self.device).to(device=self.device)

            def weights_init(m):
                classname = m.__class__.__name__
                if classname.find('Conv2d') != -1:
                    m.weight.data.uniform_(-0.02/np.sqrt(3.0).astype(np.float32), 0.02/np.sqrt(3.0).astype(np.float32))
                    m.bias.data.fill_(0)
                if classname.find('Linear') != -1:
                    m.weight.data.uniform_(-0.02/np.sqrt(3.0).astype(np.float32), 0.02/np.sqrt(3.0).astype(np.float32))
                    m.bias.data.fill_(0)

            #self.Gen2.apply(weights_init)
            self.D.apply(weights_init)


        else:
            self.Gen1 = torch.load(self.pretrained_path + '/G1.pt').to(device=self.device)
            self.Gen2 = torch.load(self.pretrained_path + '/G2.pt').to(device=self.device)
            self.D    = torch.load(self.pretrained_path + '/D.pt').to(device=self.device)


    def train(self):
        self._init_net()
        self._set_optimizers()
        step = 0
        for epoch in range(self.num_epochs):
            for batch, batch_data in enumerate(self.generator):

                    self.x, self.x_target, self.pose, self.pose_target, \
                    self.mask, self.mask_target = batch_data

                    self.x           = self.x.to(device=self.device)
                    self.x_target    = self.x_target.to(device=self.device)
                    self.pose        = self.pose.to(device=self.device)
                    self.pose_target = self.pose_target.to(device=self.device)
                    self.mask        = self.mask.to(device=self.device)
                    self.mask_target = self.mask_target.to(device=self.device)

                    if step < 20000: # Train only G1
                        self.loss()
                        self.g1_solver.zero_grad()
                        self.g1_loss.backward()
                        self.g1_solver.step()
                    else: # Train G2 and D
                        self.loss()
                        self.d_solver.zero_grad()
                        self.d_loss.backward()
                        self.d_solver.step()

                        self.loss() #TODO: is it required for alternating optimization?
                        self.g2_solver.zero_grad()
                        self.g2_loss.backward()
                        self.g2_solver.step()

                    step += 1
                    
                    if step % self.print_every == 0:
                        print ('Step [{}/{}], G1 Loss: {:.4f}, G2 Loss: {:.4f}, D Loss: {:.4f}'
                                .format(step,
                                self.num_epochs * self.n_samples/self.batch_size,
                                self.g1_loss.item(),
                                self.g2_loss.item(),
                                self.d_loss.item()))

                    if step % self.log_every == 0:
                                               
                        x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed = self.get_image_from_loader()
                        
                        # Save all important images
                        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
                        save_image(x_target_fixed, '{}/x_target_fixed.png'.format(self.model_dir))
                        save_image((torch.max(pose_fixed, dim=1, keepdim=True)[0]+1.0)*127.5, '{}/pose_fixed.png'.format(self.model_dir))
                        save_image((torch.max(pose_target_fixed, dim=1, keepdim=True)[0]+1.0)*127.5, '{}/pose_target_fixed.png'.format(self.model_dir))
                        save_image(mask_fixed, '{}/mask_fixed.png'.format(self.model_dir))
                        save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(self.model_dir))

                        # Log scalar values
                        ssim = self.generate(x_fixed, x_target_fixed, pose_target_fixed, idx=step)

                        info = { 'G1 loss': self.g1_loss.item(),
                                'G2 loss': self.g2_loss.item(),
                                'D loss': self.d_loss.item(),
                                'SSIM': ssim}

                        for tag, value in info.items():
                            self.logger.scalar_summary(tag, value, step)

                        # Log generated images
                        info = { 'Target': 255 * x_target_fixed.view(-1, 3, 256, 256)[:10].cpu().numpy(),
                                'G1': self.G1.view(-1, 3, 256, 256)[:10].cpu().numpy(),
                                'G2': self.G.view(-1, 3, 256, 256)[:10].cpu().numpy()}

                        for tag, images in info.items():
                            self.logger.image_summary(tag, images, step)

                    if step % self.save_every == 0:
                        # Save checkpoints 
                        if not os.path.exists('./checkpoints'):
                            os.makedirs('./checkpoints')
                        if not os.path.exists('./checkpoints/'+str(step)):
                            os.makedirs('./checkpoints/'+str(step))
                        torch.save(self.Gen1, './checkpoints/'+str(step)+'/G1.pt')
                        torch.save(self.Gen2, './checkpoints/'+str(step)+'/G2.pt')
                        torch.save(self.D, './checkpoints/'+str(step)+'/D.pt')

                    

    def test(self):
        # TODO: fix the iteration and torch vs numpy
        self._init_net()
        test_result_dir = os.path.join(self.model_dir, 'test_result')
        test_result_dir_x = os.path.join(test_result_dir, 'x')
        test_result_dir_x_target = os.path.join(test_result_dir, 'x_target')
        test_result_dir_G = os.path.join(test_result_dir, 'G')
        test_result_dir_pose = os.path.join(test_result_dir, 'pose')
        test_result_dir_pose_target = os.path.join(test_result_dir, 'pose_target')
        test_result_dir_mask = os.path.join(test_result_dir, 'mask')
        test_result_dir_mask_target = os.path.join(test_result_dir, 'mask_target')
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
        if not os.path.exists(test_result_dir_x):
            os.makedirs(test_result_dir_x)
        if not os.path.exists(test_result_dir_x_target):
            os.makedirs(test_result_dir_x_target)
        if not os.path.exists(test_result_dir_G):
            os.makedirs(test_result_dir_G)
        if not os.path.exists(test_result_dir_pose):
            os.makedirs(test_result_dir_pose)
        if not os.path.exists(test_result_dir_pose_target):
            os.makedirs(test_result_dir_pose_target)
        if not os.path.exists(test_result_dir_mask):
            os.makedirs(test_result_dir_mask)
        if not os.path.exists(test_result_dir_mask_target):
            os.makedirs(test_result_dir_mask_target)

        for batch in self.generator:
            self.x, self.x_target, self.pose, self.pose_target, \
            self.mask, self.mask_target = batch
            x_fixed, x_target_fixed, pose_fixed, pose_target_fixed, mask_fixed, mask_target_fixed = self.get_image_from_loader()
            x = process_image(x_fixed, 127.5, 127.5)
            x_target = process_image(x_target_fixed, 127.5, 127.5)
            if i == 0:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=batch, save=True)
            else:
                x_fake = self.generate(x, x_target, pose_target_fixed, test_result_dir, idx=batch, save=False)
            p = (torch.max(pose_fixed, dim=1, keepdim=False)[0]+1.0)*127.5
            pt = (torch.max(pose_target_fixed, dim=1, keepdim=False)[0]+1.0)*127.5
            for j in range(self.batch_size):
                idx = i*self.batch_size+j
                im = Image.fromarray(x_fixed[j,:].numpy().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x, idx))
                im = Image.fromarray(x_target_fixed[j,:].numpy().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_x_target, idx))
                im = Image.fromarray(x_fake[j,:].numpy().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_G, idx))
                im = Image.fromarray(p[j,:].numpy().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose, idx))
                im = Image.fromarray(pt[j,:].numpy().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_pose_target, idx))
                im = Image.fromarray(mask_fixed[j,:].numpy().squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask, idx))
                im = Image.fromarray(mask_target_fixed[j,:].numpy().squeeze().astype(np.uint8))
                im.save('%s/%05d.png'%(test_result_dir_mask_target, idx))
            if 0==i:
                save_image(x_fixed, '{}/x_fixed.png'.format(test_result_dir))
                save_image(x_target_fixed, '{}/x_target_fixed.png'.format(test_result_dir))
                save_image(mask_fixed, '{}/mask_fixed.png'.format(test_result_dir))
                save_image(mask_target_fixed, '{}/mask_target_fixed.png'.format(test_result_dir))
                save_image((torch.max(pose_fixed, dim=1, keepdim=True)[0]+1.0)*127.5, '{}/pose_fixed.png'.format(test_result_dir))
                save_image((torch.max(pose_target_fixed, dim=1, keepdim=True)[0]+1.0)*127.5, '{}/pose_target_fixed.png'.format(test_result_dir))

    def get_image_from_loader(self):
        x = (self.x + 1)/2 # range 0-1 for torchvision.utils.save_image
        x_target = (self.x_target + 1)/2
        mask = self.mask.detach()*255
        mask_target = self.mask_target.detach()*255
        pose = self.pose.detach()
        pose_target = self.pose_target.detach()
        return x, x_target, pose, pose_target, mask, mask_target


    def generate(self, x_fixed, x_target_fixed, pose_target_fixed, root_path='./generated_images', path=None, idx=None, save=True):
        """Assumes x_target_fixed is a torch tensor"""

        ssim_G_x_list = []
        G = self.G # already unprocessed (0-255)
        for i in range(G.shape[0]):
            G_clamped = torch.clamp(G[i,:],min=0,max=255)
            G_hwc     = G_clamped.permute(1,2,0)
            x_t_hwc   = x_target_fixed[i,:].permute(1,2,0)
            G_gray    = rgb2gray(G_hwc.cpu().numpy().astype(np.uint8))
            x_target_gray = rgb2gray((torch.clamp((x_t_hwc+1)*127.5, min=0,max=255)).cpu().numpy().astype(np.uint8))
            ssim_G_x_list.append(ssim(G_gray, x_target_gray, data_range=x_target_gray.max() - x_target_gray.min(), multichannel=False))
        ssim_G_x_mean = np.mean(ssim_G_x_list)
        if path is None and save:
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            path = os.path.join(root_path, '{}_G_ssim{}.png'.format(idx,ssim_G_x_mean))
            save_image(G/255.0, path) #torchvision.utils.save_image requires it to be 0-1
            print("[*] Samples saved: {}".format(path))
        return ssim_G_x_mean
