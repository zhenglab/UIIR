import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import unsupervised_networks as networks
from . import base_networks as networks_init
from .patchnce import PatchNCELoss
import util.ssim as ssim
import math


class NonIdealGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='ris', dataset_mode='unaligned', netD='patchgan')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_REC', type=float, default=100.0, help='weight for Rec L2 loss')
            parser.add_argument('--lambda_i_smooth', type=float, default=10.0, help='weight for illumination smooth loss')
            parser.add_argument('--lambda_s_smooth', type=float, default=10.0, help='weight for scattering smooth loss')
            parser.add_argument('--lambda_i_L2', type=float, default=5.0, help='weight for illumination is closeto degraded loss')
            parser.add_argument('--lambda_GAN', type=float, default=10.0, help='weight for GAN loss')
            parser.add_argument('--lambda_vgg', type=float, default=2.0, help='weight for VGG loss')
            parser.add_argument('--lambda_r_grident', type=float, default=0.0, help='weight for reflectance grident loss')
            parser.add_argument('--lambda_NCE', type=float, default=5.0, help='weight for pNCE loss')

            
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['G','G_Rec','G_I_L2','G_I_smooth','G_S_smooth',"G_GAN","D","D_real","D_fake"]
        self.visual_names = ['reflectance','real','fake', 'reconstruction', 'illumination','scattering']
        
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        print(self.netG)  

        if self.isTrain:
            self.netD = networks.define_D(opt.netD, opt.init_type, opt.init_gain, self.opt)
            self.model_names.append("D")
            util.saveprint(self.opt, 'netG', str(self.netG))  
            util.saveprint(self.opt, 'netD', str(self.netD))  
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionVGG = networks_init.VGGLoss(self.device)
            self.criterionSSIM = ssim.SSIM()
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.criterionGAN = networks_init.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCos = torch.nn.CosineSimilarity(dim=1).to(self.device)


            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            if self.opt.lambda_NCE > 0:
                self.netF_R = networks.define_F(opt.input_nc, 'mlp_sample', not opt.no_dropout, opt.init_type, opt.init_gain, opt)
                self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
                self.model_names.append("F_R")
                self.loss_names.append("NCE_R")
                self.criterionNCE = []

                for nce_layer in self.nce_layers:
                    self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

                self.optimizer_F = torch.optim.Adam(self.netF_R.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)
            if self.opt.lambda_vgg > 0:
                self.loss_names.append("G_VGG")
            if self.opt.lambda_r_grident > 0:
                self.loss_names.append("G_R_grident")

    def set_position(self, pos, patch_pos=None):
        pass
    def set_input(self, input):
        self.fake = input['fake'].to(self.device)
        self.real = input['real'].to(self.device)
        self.image_paths = input['img_path']

    def data_dependent_initialize(self, data):
        pass


    def forward(self):
        self.reconstruction, self.reflectance, self.illumination, self.scattering = self.netG(inputs=self.fake)

    def compute_G_loss(self):
        self.loss_G_Rec = self.criterionL2(self.reconstruction, self.fake)*self.opt.lambda_REC
        
        self.loss_G_I_L2 = self.criterionL2(self.illumination, self.fake)*self.opt.lambda_i_L2
        self.loss_G_I_smooth = (util.compute_smooth_loss(self.illumination))*self.opt.lambda_i_smooth
        self.loss_G_S_smooth = (util.compute_smooth_loss(self.scattering))*self.opt.lambda_s_smooth

        D_output = self.netD(self.reflectance)
        self.loss_G_GAN = self.criterionGAN(D_output, True)*self.opt.lambda_GAN

        self.loss_G = self.loss_G_Rec + self.loss_G_GAN + self.loss_G_I_L2 + self.loss_G_I_smooth + self.loss_G_S_smooth
        if self.opt.lambda_vgg > 0:
            self.loss_G_VGG = self.criterionVGG(self.reflectance, self.fake) * self.opt.lambda_vgg
            self.loss_G = self.loss_G + self.loss_G_VGG
        if self.opt.lambda_NCE > 0:
            self.loss_NCE_R = self.calculate_NCE_loss(self.fake, self.reflectance)
            self.loss_G = self.loss_G+self.loss_NCE_R
        if self.opt.lambda_r_grident > 0:
            self.loss_G_R_grident = self.gradient_loss(self.reflectance, self.fake)*self.opt.lambda_r_grident
            self.loss_G = self.loss_G+self.loss_G_R_grident
        return self.loss_G

    def compute_D_loss(self):
        pred_real = self.netD(self.real)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD(self.reflectance.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        return self.loss_D

    def optimize_parameters(self):
        # forward
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad() 
        loss_D = self.compute_D_loss()
        loss_D.backward()
        self.optimizer_D.step()
        

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad() 
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0:
            self.optimizer_F.step()
        
    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y


    def calculate_NCE_loss(self, src, tgt): 
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF_R(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF_R(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

