# coding:utf-8
from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

"""
在Deep Learning中，往往loss function是非凸的，没有解析解，我们需要通过优化方法来求解。
solver的主要作用就是交替调用前向（forward)算法和后向（backward)算法来更新参数，从而最小化loss，实际上就是一种迭代的优化算法。
"""
class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD']:
            self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)

        # 优化器对象:保持当前的状态【state】和基于梯度计算更新参数.
        # 这个state是权重？
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        if torch.cuda.is_available():
            self.G.cuda()
            self.D.cuda()

    def print_network(self, model, name):
        """Print out the network information."""
        # model是一个类对象，输出model中的内容
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    # restore恢复的意思，加载模型
    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))   # 加载张量到内存
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    # ？了解此处减小学习率的方法和时机
    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    # 计算损失时，每次都要重新reset梯度为0
    # 如果不置零，Variable的梯度在每次 backward 的时候都会累加
    def reset_grad(self):
        """Reset the gradient buffers."""
        # 把优化器对象的梯度设成0
        # 梯度，反向传播时用到
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    """
    tensor是PyTorch中的完美组件，但是构建神经网络还远远不够，我们需要能够构建计算图的tensor，这就是Variable
    Variable是对tensor的封装，操作和tensor是一样的，但是每个Variable都有三个属性
    Variable中的.data：对应tensor本身
                .grad：对应tensor的梯度
                .grad_fn：实现autograd相关

    理解：tensor是描述数据流动的图 variable是实现图计算的对象
    """
    # 把张量转换为变量
    def tensor2var(self, x, volatile=False):
        """Convert torch tensor to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        # volatile: Boolean indicating that the Variable should be used in
        #     inference mode, i.e. don't save the history. See
        #     :ref:`excluding-subgraphs` for more details.
        #     Can be changed only on leaf Variables.
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        # clamp_(min,max)：把小于min的数置于min，大于max的数置max，区间内的数维持不变
        return out.clamp_(0, 1)     # 此处有些像保险操作，确保return数据范围在[0,1]

    """ ？似乎是来自于Wasserstein GAN（WGAN）的想法，待研究 """
    # gradient penalty目的：稳定训练过程（让梯度在后向传播的过程中保持平稳）和提高生成图像质量
    def gradient_penalty(self, y, x, dtype):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).type(dtype)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        """
            >>> np.arange( 5)
            array([0, 1, 2, 3, 4])
        """
        out[np.arange(batch_size), labels.long()] = 1   # ？
        return out

    # 生成用于调试和测试的目标域标签？
    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []     # 内容与c_trg一样，类型从FloatTensor变为Variable存储在list中
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()

                # 若i代表发色
                if i in hair_color_indices:     # Set one hair color to 1 and the rest to 0.   # 每张图片只有一种发色
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                # 若i为除发色外的其他属性
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.    # 属性取反？

            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(self.tensor2var(c_trg, volatile=True))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader. 获取操作变量
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        # data_loader是一个DataLoader对象，具有应该是全部数据属性
        # self是Solver对象，具有Solver的所有属性

        # Fetch fixed inputs for debugging.
        # c_org是输入图像具有的属性，内容为attribute的数字表示形式，是个二维矩阵，[ [1 0 0 1 1], ... , [0 0 0 1 1] ]，shape[ 16 x 5 ]
        # x_fixed存储的是图像信息（由[-1,1]之间的浮点值组成的二维矩阵），shape[ 16 x 3 x 128 x 128 ]
        # 16是batch_size大小
        data_iter = iter(data_loader)           # iter()函数用来生成迭代器，此处生成迭代器对象data_iter
        x_fixed, c_org = next(data_iter)        # next()获取迭代器内容，迭代器每次取batch_size张图片
        x_fixed = self.tensor2var(x_fixed, volatile=True)

        # 生成用于调试和测试的target label
        c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        # 从第resume_iters步恢复训练
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                # 注意实际训练是从第二次取数据开始
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.   随机生成目标域标签
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]             # 这两步实在有趣

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)

            x_real = self.tensor2var(x_real)           # Input images.
            c_org = self.tensor2var(c_org)             # Original domain labels.
            c_trg = self.tensor2var(c_trg)             # Target domain labels.
            label_org = self.tensor2var(label_org)     # Labels for computing classification loss.
            label_trg = self.tensor2var(label_trg)     # Labels for computing classification loss.

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            # ouc_scr: shape = [ 2, 1, 2, 2 ]
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)     # 对out_src里的所有值求和然后求平均，再取负
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset)     # 计算分类损失，交叉熵loss

            # Compute loss with fake images.
            x_fake = self.G(x_real, c_trg)
            # 在训练判别器时，需要对生成器生成的图片用 detach 操作进行计算图截断，避免反向传播将梯度传到生成器中
            # 因为在训练判别器时不需要训练生成器，也就不需要生成器的梯度
            out_src, out_cls = self.D(x_fake.detach())      # detach是分离的意思
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            # 优化GAN的收敛
            alpha = torch.rand(x_real.size(0), 1, 1, 1).type(self.dtype)
            # alpha [ 16x1x1x1 ], x_real[ 16x3x128x128 ] 如何做矩阵乘法运算？
            # x_hat: shape = [ 16 x 3 x 128 x 128 ]
            x_hat = Variable(alpha * x_real.data + (1 - alpha) * x_fake.data, requires_grad=True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat, self.dtype)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp

            self.reset_grad()           # reset梯度
            d_loss.backward()           # 计算梯度
            self.d_optimizer.step()     # 调用优化器更新参数

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.data[0]
            loss['D/loss_fake'] = d_loss_fake.data[0]
            loss['D/loss_cls'] = d_loss_cls.data[0]
            loss['D/loss_gp'] = d_loss_gp.data[0]

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            # 每训练n_critic次判别器，才训练一次生成器
            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset)

                # Target-to-original domain.
                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.data[0]
                loss['G/loss_rec'] = g_loss_rec.data[0]
                loss['G/loss_cls'] = g_loss_cls.data[0]

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time   # endtime
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(self.G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.     Decay-衰变
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = self.tensor2var(x_fixed, volatile=True)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = self.tensor2var(torch.zeros(x_fixed.size(0), self.c_dim))            # Zero vector for CelebA.
        zero_rafd = self.tensor2var(torch.zeros(x_fixed.size(0), self.c2_dim))             # Zero vector for RaFD.
        mask_celeba = self.tensor2var(self.label2onehot(torch.zeros(x_fixed.size(0)), 2))  # Mask vector: [1, 0].
        mask_rafd = self.tensor2var(self.label2onehot(torch.ones(x_fixed.size(0)), 2))     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = self.tensor2var(x_real)             # Input images.
                c_org = self.tensor2var(c_org)               # Original domain labels.
                c_trg = self.tensor2var(c_trg)               # Target domain labels.
                label_org = self.tensor2var(label_org)       # Labels for computing classification loss.
                label_trg = self.tensor2var(label_trg)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.G(x_real, c_trg)
                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).type(self.dtype)
                x_hat = Variable(alpha * x_real.data + (1 - alpha) * x_fake.data, requires_grad=True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat, self.dtype)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.data[0]
                loss['D/loss_fake'] = d_loss_fake.data[0]
                loss['D/loss_cls'] = d_loss_cls.data[0]
                loss['D/loss_gp'] = d_loss_gp.data[0]

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.G(x_real, c_trg)
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.data[0]
                    loss['G/loss_rec'] = g_loss_rec.data[0]
                    loss['G/loss_cls'] = g_loss_cls.data[0]

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                x_fake_list = [x_fixed]
                for c_fixed in c_celeba_list:
                    c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_fixed, c_trg))
                for c_fixed in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_fixed, c_trg))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader

        for i, (x_real, c_org) in enumerate(data_loader):
            
            # Prepare input images and target domain labels.
            x_real = self.tensor2var(x_real, volatile=True)
            c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
            
            # Translate images.
            x_fake_list = [x_real]
            for c_trg in c_trg_list:
                x_fake_list.append(self.G(x_real, c_trg))
            
            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)

        for i, (x_real, c_org) in enumerate(self.celeba_loader):

            # Prepare input images and target domain labels.
            x_real = self.tensor2var(x_real, volatile=True)
            c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
            c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
            zero_celeba = self.tensor2var(torch.zeros(x_real.size(0), self.c_dim))            # Zero vector for CelebA.
            zero_rafd = self.tensor2var(torch.zeros(x_real.size(0), self.c2_dim))             # Zero vector for RaFD.
            mask_celeba = self.tensor2var(self.label2onehot(torch.zeros(x_real.size(0)), 2))  # Mask vector: [1, 0].
            mask_rafd = self.tensor2var(self.label2onehot(torch.ones(x_real.size(0)), 2))     # Mask vector: [0, 1].

            # Translate images.
            x_fake_list = [x_real]
            for c_celeba in c_celeba_list:
                c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                x_fake_list.append(self.G(x_real, c_trg))
            for c_rafd in c_rafd_list:
                c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                x_fake_list.append(self.G(x_real, c_trg))

            # Save the translated images.
            x_concat = torch.cat(x_fake_list, dim=3)
            result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
            save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))