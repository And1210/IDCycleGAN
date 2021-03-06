import torch
import itertools
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks as networks
from losses.GreyscaleLoss import GreyscaleLoss
from losses.EdgeLoss import EdgeLoss
from models.pretrainnet import VGGPreTrained

class IDCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.
    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'D_B', 'D_C', 'G_A2C', 'G_C2A', 'G_B2C', 'G_C2B', 'idt_A', 'idt_B', 'cycle_A', 'cycle_B', 'cycle_C_A', 'cycle_C_B']
        #self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'gray_A', 'edge_A', 'gray_B', 'edge_B', 'feat_A2B', 'feat_B2A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'C_A', 'fake_B', 'fake_C_B', 'recon_A']
        visual_names_B = ['real_B', 'C_B', 'fake_A', 'fake_C_A', 'recon_B']
        if self.is_train and self.opt['lambda_identity'] > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.is_train:
            self.model_names = ['G_A2C', 'G_C2A', 'G_B2C', 'G_C2B', 'D_A', 'D_B', 'D_C']
            self.network_names = ['netG_A2C', 'netG_C2A', 'netG_B2C', 'netG_C2B', 'netD_A', 'netD_B', 'netD_C']
        else:  # during test time, only load Gs
            self.model_names = ['G_A2C', 'G_C2A', 'G_B2C', 'G_C2B']
            self.network_names = ['netG_A2C', 'netG_C2A', 'netG_B2C', 'netG_C2B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A2C = networks.define_G(opt['input_nc'], opt['output_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])
        self.netG_C2A = networks.define_G(opt['input_nc'], opt['output_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])
        self.netG_B2C = networks.define_G(opt['output_nc'], opt['input_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])
        self.netG_C2B = networks.define_G(opt['output_nc'], opt['input_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                        not opt['no_dropout'], opt['init_type'], opt['init_gain'], [0])

        if self.is_train:  # define discriminators
            self.netD_A = networks.define_D(opt['output_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])
            self.netD_B = networks.define_D(opt['input_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])
            self.netD_C = networks.define_D(opt['input_nc'], opt['ndf'], opt['netD'],
                                            opt['n_layers_D'], opt['norm'], opt['init_type'], opt['init_gain'], [0])

        if self.is_train:
            # self.vgg_pretrained = VGGPreTrained().to(self.device)
            # self.vgg_pretrained.setup(self.device)
            if opt['lambda_identity'] > 0.0:  # only works when input and output images have the same number of channels
                assert(opt['input_nc'] == opt['output_nc'])
            self.fake_A_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt['pool_size'])  # create image buffer to store previously generated images
            self.C_A_pool = ImagePool(opt['pool_size'])
            self.C_B_pool = ImagePool(opt['pool_size'])
            self.recon_A_pool = ImagePool(opt['pool_size'])
            self.recon_B_pool = ImagePool(opt['pool_size'])
            self.recon_C_A_pool = ImagePool(opt['pool_size'])
            self.recon_C_B_pool = ImagePool(opt['pool_size'])
            self.fake_C_A_pool = ImagePool(opt['pool_size'])
            self.fake_C_B_pool = ImagePool(opt['pool_size'])
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt['gan_mode']).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2C.parameters(), self.netG_C2A.parameters(), self.netG_B2C.parameters(), self.netG_C2B.parameters()), lr=opt['lr'], betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_C.parameters()), lr=opt['lr'], betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt['direction'] == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.C_A = self.netG_A2C(self.real_A) #G_A2C(A) C
        self.C_B = self.netG_B2C(self.real_B) #G_B2C(B) C
        self.recon_C_A = self.netG_C2A(self.C_A) #G_C2A(G_A2C(A)) A
        self.recon_C_B = self.netG_C2B(self.C_B) #G_C2B(G_B2C(B)) B
        self.fake_B = self.netG_C2B(self.C_A) #G_C2B(G_A2C(A)) B
        self.fake_A = self.netG_C2A(self.C_B) #G_C2A(G_B2C(B)) A
        self.fake_C_B = self.netG_B2C(self.fake_B) #G_B2C(G_C2B(G_A2C(A))) C
        self.fake_C_A = self.netG_A2C(self.fake_A) #G_A2C(G_C2A(G_B2C(B))) C
        self.recon_A = self.netG_C2A(self.fake_C_B) #G_C2A(G_B2C(G_C2B(G_A2C(A)))) A
        self.recon_B = self.netG_C2A(self.fake_C_A) #G_C2B(G_A2C(G_C2A(G_B2C(B)))) B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        recon_C_A = self.recon_C_A_pool.query(self.recon_C_A)
        fake_C_A = self.fake_C_A_pool.query(self.fake_C_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A) + self.backward_D_basic(self.netD_A, self.real_A, recon_C_A) + self.backward_D_basic(self.netD_A, self.real_A, fake_C_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        recon_C_B = self.recon_B_pool.query(self.recon_C_B)
        fake_C_B = self.fake_C_B_pool.query(self.fake_C_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B) + self.backward_D_basic(self.netD_B, self.real_B, recon_C_B) + self.backward_D_basic(self.netD_B, self.real_B, fake_C_B)

    def backward_D_C(self):
        C_A = self.C_A_pool.query(self.C_A)
        C_B = self.C_B_pool.query(self.C_B)
        fake_C_A = self.fake_C_A_pool.query(self.fake_C_A)
        fake_C_B = self.fake_C_B_pool.query(self.fake_C_B)
        #real_A = self.real_A_pool.query(self.real_A)
        #real_B = self.real_B_pool.query(self.real_B)
        self.loss_D_C = self.backward_D_basic(self.netD_C, C_A, self.real_A) + self.backward_D_basic(self.netD_C, C_B, self.real_B) + self.backward_D_basic(self.netD_C, fake_C_A, self.real_A) + self.backward_D_basic(self.netD_C, fake_C_B, self.real_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt['lambda_identity']
        lambda_A = self.opt['lambda_A']
        lambda_B = self.opt['lambda_B']
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_C2A(self.real_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * lambda_A * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_C2B(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        #['D_A', 'D_B', 'D_C', 'G_A2C', 'G_C2A', 'G_B2C', 'G_C2B', 'idt_A', 'idt_B', 'cycle_A', 'cycle_B']

        # GAN loss D_C(G_A2C(A))
        self.loss_G_A2C = self.criterionGAN(self.netD_C(self.C_A), False) + self.criterionGAN(self.netD_C(self.fake_C_A), False) #THESE TWO COULD BE A PROBLEM
        # GAN loss D_C(G_B2C(B))
        self.loss_G_B2C = self.criterionGAN(self.netD_C(self.C_B), False) + self.criterionGAN(self.netD_C(self.fake_C_B), False)
        # GAN loss D_C(G_C2A(G_A2C(A)))
        self.loss_G_C2A = self.criterionGAN(self.netD_A(self.recon_C_A), True) + self.criterionGAN(self.netD_A(self.recon_A), True)
        # GAN loss D_C(G_C2B(G_B2C(B)))
        self.loss_G_C2B = self.criterionGAN(self.netD_B(self.recon_C_B), True) + self.criterionGAN(self.netD_B(self.recon_B), True)
        # Forward cycle loss || G_C2A(G_B2C(G_C2B(G_A2C(A)))) - A||
        self.loss_cycle_A = self.criterionCycle(self.recon_A, self.real_A) * lambda_A
        # Backward cycle loss || G_C2B(G_A2C(G_C2A(G_B2C(A)))) - B||
        self.loss_cycle_B = self.criterionCycle(self.recon_B, self.real_B) * lambda_B

        self.loss_cycle_C_A = self.criterionCycle(self.recon_C_A, self.real_A) * lambda_A
        self.loss_cycle_C_B = self.criterionCycle(self.recon_C_B, self.real_B) * lambda_B

        # combined loss and calculate gradients
        #self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_gray_A + self.loss_edge_A + self.loss_gray_B + self.loss_edge_B
        self.loss_G = self.loss_G_A2C + self.loss_G_B2C + self.loss_G_C2A + self.loss_G_C2B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_cycle_C_A + self.loss_cycle_C_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B, self.netD_C], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.backward_D_C()
        self.optimizer_D.step()  # update D_A and D_B's weights
