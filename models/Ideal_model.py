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


class IdealModel(BaseModel):
  
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='ris', dataset_mode='uieb')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_REC', type=float, default=50.0, help='weight for Rec L2 loss')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_ssim', type=float, default=50.0, help='weight for SSIM loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        self.loss_names = ['G','G_Rec',"G_SSIM","G_L1"]
        self.visual_names = ['reflectance','real','fake', 'reconstruction', 'illumination','scattering']
        
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        print(self.netG)  
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionSSIM = ssim.DSSIM().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

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
        self.loss_G_L1 = self.criterionL1(self.reflectance, self.real)*self.opt.lambda_L1
        self.loss_G_SSIM = self.criterionSSIM(self.reflectance, self.real)*self.opt.lambda_ssim
        self.loss_G = self.loss_G_Rec + self.loss_G_L1 + self.loss_G_SSIM
        
        return self.loss_G

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad() 
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()