import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from src.losses import CARL, kl_divergence, loss_hinge_dis, loss_hinge_gen, tv_loss, long_live_gan_adv_loss, long_live_gan_disc_loss
from src.utils.grad_norm import grad_norm
from src.models.cgan.counterfactual_cgan import posterior2bin, CounterfactualCGAN
from src.models.cgan.loss_balancer import PastGradientLossBalancer

FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


class CounterfactualInpaintingCGAN(CounterfactualCGAN):
    
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(img_size, opt, *args, **kwargs)
        self.lambda_tv = opt.get('lambda_tv', 0.0)
        self.loss_balancer = opt.get('loss_balancer',False)
        self.adv_gamma = opt.get('adv_gamma', 1.0)
        self.apply_tanh_to_non_gen_imgs = opt.get('apply_tanh_to_non_gen_imgs', False)
        self.dilation_kernel_size = opt.get('dilation_kernel_size', 2)
        self.cf_gt_seg_mask_idx = opt.get('cf_gt_seg_mask_idx', -1)
        self.cf_threshold = opt.get('cf_threshold', 0.25)
        self.lambda_iou = opt.get('lambda_iou', 0.0)
        self.always_generate_healthy = opt.get('always_generate_healthy', False)
        
        print("Always_generate_Helathy is set to: ", self.always_generate_healthy)
        print(f"Kernel based recon loss set to {opt.get('reconstruction_dilation', False)} used for this is: {self.dilation_kernel_size}")
        if self.loss_balancer:
            self.loss_balancer_obj = PastGradientLossBalancer(['g_adv_loss','g_kl','g_rec_loss','g_tv'],
                                                              smoothing = 0.8,
                                                              intensities = {'g_adv_loss':self.lambda_adv,
                                                               'g_kl':self.lambda_kl,
                                                               'g_rec_loss':self.lambda_rec,
                                                               'g_tv':self.lambda_tv},
                                                              initial_weights = {'g_adv_loss':self.lambda_adv,
                                                               'g_kl':self.lambda_kl,
                                                               'g_rec_loss':self.lambda_rec,
                                                               'g_tv':self.lambda_tv})
            print("USING LOSS BALANCER!")
    
    def posterior_prob(self, x):
        f_x, f_x_discrete, _, _, penultimate = super().posterior_prob(x)
        f_x_desired = f_x.clone().detach()
        f_x_desired_discrete = f_x_discrete.clone().detach()
        
        # mask of what samples classifier predicted as `abnormal`
        if self.always_generate_healthy:
            inpaint_group = torch.ones_like(f_x_discrete).bool()
        else:
            inpaint_group = f_x_discrete.bool()
        
        
        # `abnormalities` need to be inpainted and classifier should predict `normal` on them
        f_x_desired[inpaint_group] = 1e-6
        f_x_desired_discrete[inpaint_group] = 0
        return f_x, f_x_discrete, f_x_desired, f_x_desired_discrete, penultimate
    
    def torch_dilation(self, images, kernel_size):
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        
        if images.ndim == 3:  # Assuming input shape is (B, H, W)
            images = images.unsqueeze(1)  # Add a channel dim: (B, 1, H, W)
           
        dilated_images = F.max_pool2d(images, (kernel_size,kernel_size), stride=1)
        return dilated_images.squeeze(1)


    def perfect_classifier_IoU(self, real_imgs, gen_imgs, gt_masks):
            # FUnction to emulate perfect calssifer to understand if the GAN pipeline is able to complete the task
            gt_masks = gt_masks[:, self.cf_gt_seg_mask_idx]
            if gt_masks.ndim == 3:
                gt_masks = gt_masks.unsqueeze(1)

            real_imgs = (real_imgs + 1) / 2
            gen_imgs = (gen_imgs + 1) / 2
            
            diff = (real_imgs - gen_imgs).abs() # [0; 1] values
            diff_seg = (diff > self.cf_threshold).byte()
            intersection = (diff_seg * gt_masks).sum(dim=[2, 3])

            union = diff_seg.sum(dim=[2, 3]) + gt_masks.sum(dim=[2, 3]) - intersection
            batch_ious = (intersection / union.clamp(min=1e-8)).flatten()
            batch_ious = torch.ones_like(batch_ious) - batch_ious
            
            
            return kl_divergence(batch_ious, torch.zeros_like(batch_ious))
        
        

    def reconstruction_loss(self, real_imgs, gen_imgs, masks, f_x_discrete, f_x_desired_discrete, z=None):
        if self.opt.get('reconstruction_dilation', False):
            diff = (real_imgs - gen_imgs).abs()
            dilated = self.torch_dilation(diff, self.dilation_kernel_size)
            return dilated.sum()/diff.numel()
            
        
        if self.opt.get('only_cyclic_rec',False):
            ifxc_fx = self.explanation_function(
            x=gen_imgs,  # I_f(x, c)
            f_x_discrete=f_x_desired_discrete, # f_x_desired_discrete is always zeros
            )
            return  self.l1(gen_imgs, ifxc_fx)
        
        
        forward_term = self.l1(real_imgs, gen_imgs)
        
        if self.opt.get('only_forward_term', False):
            return forward_term

        ifxc_fx = self.explanation_function(
            x=gen_imgs,  # I_f(x, c)
            f_x_discrete=f_x_desired_discrete, # f_x_desired_discrete is always zeros
        )
        # cyclic rec 1
        # L_rec(x, I_f(I_f(x, c), f(x)))
        cyclic_term = self.l1(real_imgs, ifxc_fx)

        # cyclic rec 2
        # cyclic_term = self.l1(gen_imgs, ifxc_fx)
        return forward_term + cyclic_term

    def forward(self, batch, training=False, validation=False, compute_norms=False, global_step=None):
        assert training and not validation or validation and not training

        # `real_imgs` and `gen_imgs` are in [-1, 1] range
        imgs, labels, masks, healthy_example = batch['image'], batch['label'], batch['masks'], batch['healthy_example']
        batch_size = imgs.shape[0]

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        masks = Variable(masks.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))
        healthy_example = Variable(healthy_example.type(FloatTensor))
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Classifier predictions and desired outputs for the explanation function
        with torch.no_grad():
            # technically condition `c` (real_f_x_desired) is now classifier driven choice to:
            # 1) `inpaint`  (real_f_x_desired == 1)
            # 2) `identity` (real_f_x_desired == 0)
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete, penultimate = self.posterior_prob(real_imgs)
            

        # -----------------
        #  Train Generator
        # -----------------
        if training:
            self.optimizer_G.zero_grad()

        # E(x)
        z = self.enc(real_imgs)
        # G(z, c) = I_f(x, c)
        gen_imgs = self.gen(z, real_f_x_desired_discrete, x=real_imgs if self.ptb_based else None, class_prob = real_f_x, classifier_output=penultimate)

        update_generator = global_step is not None and global_step % self.gen_update_freq == 0
        if self.apply_tanh_to_non_gen_imgs:
            real_imgs = torch.tanh(real_imgs)
            healthy_example = torch.tanh(healthy_example)
        
        ### Update generator without balancer.
        if update_generator or validation:
            
            if self.loss_balancer:
                
                ########## data consistency loss for generator
                dis_fake = self.disc(gen_imgs, real_f_x_desired_discrete)
                dis_real = self.disc(healthy_example, real_f_x_desired_discrete) ### SIIA TANH
                if self.adv_loss == 'hinge':
                    g_adv_loss = loss_hinge_gen(dis_fake)
                if self.adv_loss == 'long_live_gan':
                    g_adv_loss = long_live_gan_adv_loss(dis_real, dis_fake)
                else:
                    g_adv_loss = self.adversarial_loss(dis_fake, valid)
                ##################
                
                
                ######### Classifier loss
                gen_f_x, _, _, _,_ = self.posterior_prob(gen_imgs)
                # both y_pred and y_target are single-value probs for class k
                g_kl = (
                    kl_divergence(gen_f_x, real_f_x_desired)
                    if self.lambda_kl != 0 else torch.tensor(0.0, requires_grad=True)
                )
                ###############
                
                ########### Reconstruction Loss
                # reconstruction loss for generator
                g_rec_loss = (
                    self.reconstruction_loss(real_imgs, gen_imgs, masks, real_f_x_discrete, real_f_x_desired_discrete, z=z)
                    if self.lambda_rec != 0 else torch.tensor(0.0, requires_grad=True)
                )
                ##################
                
                ############ 
                if self.lambda_minc != 0:
                    g_minc_loss =  self.l1(real_imgs, gen_imgs)
                else:
                    g_minc_loss = torch.tensor(0.0, requires_grad=True)
                ######################
                
                ###########
                if self.lambda_tv != 0:
                    g_tv = tv_loss(torch.abs(real_imgs.add(1).div(2) - gen_imgs.add(1).div(2)).mul(255))
                else:
                    g_tv = torch.tensor(0.0, requires_grad=True)
                ##################
                
                loss_dict = {'g_adv_loss':g_adv_loss,
                            'g_kl':g_kl,
                            'g_rec_loss':g_rec_loss,
                            'g_tv':g_tv}

                loss_dict_copy = {'g_adv_loss':g_adv_loss.clone().detach(),
                            'g_kl':g_kl.clone().detach(),
                            'g_rec_loss':g_rec_loss.clone().detach(),
                            'g_tv':g_tv.clone().detach()}
                
                balanced_weights = self.loss_balancer_obj.get_loss_weights(loss_dict_copy)
                
                
                # total generator loss
                g_loss = sum(balanced_weights[loss_name] * loss_value for loss_name, loss_value in loss_dict.items())
                
                self.gen_loss_logs.update({f'{loss_name}_weight': weight for loss_name, weight in balanced_weights.items()})
                
            else:
                
                # data consistency loss for generator
                dis_fake = self.disc(gen_imgs, real_f_x_desired_discrete)
                dis_real = self.disc(healthy_example, real_f_x_desired_discrete)
                if self.adv_loss == 'hinge':
                    g_adv_loss = self.lambda_adv * loss_hinge_gen(dis_fake)
                    
                if self.adv_loss == 'long_live_gan':
                    g_adv_loss = self.lambda_adv * long_live_gan_adv_loss(dis_real, dis_fake)
                    
                else:
                    g_adv_loss = self.lambda_adv * self.adversarial_loss(dis_fake, valid)

                # classifier consistency loss for generator
                # f(I_f(x, c)) ≈ c
                gen_f_x, _, _, _,_ = self.posterior_prob(gen_imgs)
                # both y_pred and y_target are single-value probs for class k
                g_kl = (
                    self.lambda_kl * kl_divergence(gen_f_x, real_f_x_desired)
                    if self.lambda_kl != 0 else torch.tensor(0.0, requires_grad=True)
                )
                # reconstruction loss for generator
                g_rec_loss = (
                    self.lambda_rec * self.reconstruction_loss(real_imgs, gen_imgs, masks, real_f_x_discrete, real_f_x_desired_discrete, z=z)
                    if self.lambda_rec != 0 else torch.tensor(0.0, requires_grad=True)
                )
                if self.lambda_minc != 0:
                    g_minc_loss = self.lambda_minc * self.l1(real_imgs, gen_imgs)
                else:
                    g_minc_loss = torch.tensor(0.0, requires_grad=True)
                
                if self.lambda_tv != 0:
                    g_tv = self.lambda_tv * tv_loss(torch.abs(real_imgs.add(1).div(2) - gen_imgs.add(1).div(2)).mul(255))
                else:
                    g_tv = torch.tensor(0.0, requires_grad=True)
                # total generator loss
                if self.lambda_iou != 0:
                    g_iou = self.lambda_iou * self.perfect_classifier_IoU(real_imgs, gen_imgs, masks)
                else:
                    g_iou = torch.tensor(0.0, requires_grad=True)
                    
                g_loss = g_adv_loss + g_kl + g_rec_loss + g_minc_loss + g_tv + g_iou
                

            # update generator
            if update_generator:
                self.fabric.backward(g_loss)
                if compute_norms:
                    self.norms['E'] = grad_norm(self.enc)
                    self.norms['G'] = grad_norm(self.gen)
                self.optimizer_G.step()

            self.gen_loss_logs['g_adv'] = g_adv_loss.item()
            self.gen_loss_logs['g_kl'] = g_kl.item()
            self.gen_loss_logs['g_rec_loss'] = g_rec_loss.item()
            self.gen_loss_logs['g_minc_loss'] = g_minc_loss.item()
            self.gen_loss_logs['g_tv'] = g_tv.item()
            self.gen_loss_logs['g_loss'] = g_loss.item()




        # ---------------------
        #  Train Discriminator
        # ---------------------
        if training:
            self.optimizer_D.zero_grad()
            
        healthy_example.requires_grad_() 
        
        gen_imgs_detached = gen_imgs.detach()
        gen_imgs_detached.requires_grad_() 
            
        
        # Compute discriminator outputs
        dis_real = self.disc(healthy_example)
        dis_fake = self.disc(gen_imgs_detached)

        # data consistency loss for discriminator (real and fake images)
        if self.adv_loss == 'hinge':
            d_real_loss, d_fake_loss = loss_hinge_dis(dis_fake, dis_real)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
        elif self.adv_loss == 'long_live_gan' and training:
            # Use detached version with gradients enabled
            d_loss, R1Penalty, R2Penalty = long_live_gan_disc_loss(dis_real, dis_fake, healthy_example, gen_imgs_detached, self.adv_gamma)
        
        elif self.adv_loss == 'long_live_gan' and not training:
            RelativisticLogits = dis_real - dis_fake
            AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
                
            d_loss = AdversarialLoss.mean() 
        
        else:
            d_real_loss = self.adversarial_loss(dis_real, valid)
            d_fake_loss = self.adversarial_loss(dis_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
        


        if training:
            # Perform backward pass
            self.fabric.backward(d_loss)
            
            if compute_norms:
                self.norms['D'] = grad_norm(self.disc)
            self.optimizer_D.step()
            
        # Store losses for logging
        if self.adv_loss == 'long_live_gan':
            self.disc_loss_logs['d_real_loss'] = 0.0  # Not separately computed for long_live_gan
            self.disc_loss_logs['d_fake_loss'] = 0.0  # Not separately computed for long_live_gan
            if 'R1Penalty' in locals():
                self.disc_loss_logs['R1Penalty'] = R1Penalty
                self.disc_loss_logs['R2Penalty'] = R2Penalty
        else:
            self.disc_loss_logs['d_real_loss'] = d_real_loss.item()
            self.disc_loss_logs['d_fake_loss'] = d_fake_loss.item()
            
        self.disc_loss_logs['d_loss'] = d_loss.item()

        outs = {
            'loss': {**self.gen_loss_logs, **self.disc_loss_logs},
            'gen_imgs': gen_imgs,
            'healthy_examples': healthy_example,
        }
        return outs
