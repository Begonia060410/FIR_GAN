import logging
import random

import torch
from kornia.losses import SSIMLoss

from Loss.D_loss import DLoss
from Loss.measureG import CLoss, GLoss
from python.iqa import qem
from moudle.discriminator import Discriminator
from moudle.generator import Generator
from utils.read_h5 import read_data, read_data_vi

class Train:
    # defined a train process, run
    def __init__(self,environment_probe, config):
        logging.info(f'ATFusionGAN training process!')
        self.config=config
        self.environment_probe=environment_probe

        logging.info(f'generator module define!')
        self.generator=Generator()
        logging.info(f'discriminator module define!')
        self.discriminator=Discriminator()

        logging.info(f'learning rate define!')
        self.opt_generator=torch.optim.Adam(self.generator.parameters(),lr=config.learning_rate)
        self.opt_discriminator=torch.optim.Adam(self.discriminator.parameters(),lr=config.learning_rate)

        logging.info(f'assign device:{environment_probe.device} to modules!')
        self.generator.to(environment_probe.device)
        self.discriminator.to(environment_probe.device)

        logging.info(f'loss function define!')
        # generator M_content
        self.C_loss_ob=CLoss()
        self.G_loss_ob=GLoss(device=self.environment_probe.device)
        # generator M_structure
        self.SSIM_loss_ob=SSIMLoss(window_size=11, reduction='none')
        self.d_loss_ob=DLoss()

        #loss to cuda
        self.C_loss_ob.cuda()
        self.G_loss_ob.cuda()
        self.SSIM_loss_ob.cuda()
        self.d_loss_ob.cuda()
        self.quality_aware=0.6


    def train_step(self,ir_images,ir_labels,vi_images,vi_labels,index_,batch_size):
        logging.debug('train generator')
        self.generator.train()
        d_loss=0
        g_loss=0
        fused_img=self.generator(ir_images,vi_images)

        with torch.no_grad():
            fusion_out=fused_img
        logging.debug('train discriminator')

        for d_train_number in range(2):

            real_ir=ir_labels#tensor:2;[[1,128,128],[1,128,128]]
            real_vi=vi_labels#tensor:2;[[1,128,128],[1,128,128]]
            self.quality_aware=qem(ir_labels,vi_labels,d_train_number,index_,batch_size)

            self.opt_discriminator.zero_grad()
            #判别器
            real_ir_pro=self.discriminator(real_ir)#tensor:2;[[1,2],[1,2]]
            real_vi_pro=self.discriminator(real_vi)#tensor:2;[[1,2],[1,2]]
            fake_fused_pro=self.discriminator(fusion_out)#tensor:2;[[1,128,128],[1,128,128]]

            d1_loss=self.d_loss_ob(real_vi_pro,1,0)+self.d_loss_ob(real_ir_pro,0,1)


            d2_loss=self.d_loss_ob(fake_fused_pro,0,0)

            d_train_loss=d1_loss+d2_loss
            d_loss+=d_train_loss.cpu().item()
            d_train_loss.backward(retain_graph=True)
            self.opt_discriminator.step()
        self.opt_generator.zero_grad()


        #ssim
        ssim_i_r = float(self.SSIM_loss_ob(fusion_out, vi_images).mean())
        if ssim_i_r > 0.6:
            alph = 1 - (ssim_i_r - 1) ** 2
        else:
            alph = ssim_i_r
        loss_measure=self.quality_aware*(5*self.G_loss_ob(fusion_out,vi_images)+10*self.C_loss_ob(fusion_out,vi_images))+(1-self.quality_aware)*(5*self.G_loss_ob(fusion_out,ir_images)+10*self.C_loss_ob(fusion_out,ir_images))

        loss_structure=(self.quality_aware*self.SSIM_loss_ob(fusion_out,ir_images)*alph+(1-self.quality_aware)*self.SSIM_loss_ob(fusion_out,vi_images)*(1-alph)).mean()

        #adverse loss
        self.discriminator.eval()
        fused_img_pro=self.discriminator(fusion_out)
        g_loss_adv=self.d_loss_ob(fused_img_pro,random.uniform(0.7,1) ,random.uniform(0.7,1) )
        gc_train_loss = loss_measure+loss_structure
        g_loss=gc_train_loss+0.4*g_loss_adv
        g_loss.backward()
        self.opt_generator.step()

        return g_loss,d_loss/2


    def run(self):
        # train dataset
        logging.info(f'dataset define!')

        # # img--->h5 file 生成裁减图像
        # input_setup(self.config, r"E:\begoina\fir-gan\dataset\vehicle\vis", ir_flag=True)
        # input_setup(self.config, r"E:\begoina\fir-gan\dataset\vehicle\ir", ir_flag=False)

        data_dir_ir = r'E:\begoina\fir-gan\dataset\vehicle\ir'
        data_dir_vi = r'E:\begoina\fir-gan\dataset\vehicle\vis'


        batch_size=self.config.batch_size

        import time

        for epoch in range(1,self.config.epoch+1):
            # read h5 file
            train_data_ir, train_label_ir, index = read_data(data_dir_ir)
            train_data_vi, train_label_vi = read_data_vi(data_dir_vi, index)
            start_time = time.time()

            batch_idxs = len(train_data_ir)//self.config.batch_size

            for idx in range(1,1+batch_idxs):#batch:128


                start_idx = (idx - 1) * batch_size
                ir_images = train_data_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])#train_data_ir:43277
                ir_labels = train_label_ir[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                vi_images = train_data_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])
                vi_labels = train_label_vi[start_idx: start_idx + batch_size].transpose([0, 3, 1, 2])

                ir_images = torch.tensor(ir_images).float().to(self.environment_probe.device)
                ir_labels = torch.tensor(ir_labels).float().to(self.environment_probe.device)
                vi_images = torch.tensor(vi_images).float().to(self.environment_probe.device)
                vi_labels = torch.tensor(vi_labels).float().to(self.environment_probe.device)

                index_ = index[start_idx: start_idx + batch_size]
                g_loss,d_loss=self.train_step(ir_images,ir_labels,vi_images,vi_labels,index_,batch_size)

                if idx%1 == 0:
                    # 记录结束时间
                    end_time = time.time()

                    # 计算并打印执行时间
                    elapsed_time = end_time - start_time

                    print('Epoch {}/{}, Step {}/{}, gen_loss = {:.4f}, dis_loss = {:.4f}'.format(epoch, self.config.epoch, idx,
                                                                                              batch_idxs, g_loss,d_loss) + 'time:'+ str(elapsed_time))
            import os
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            model_path = os.path.join(
                os.getcwd(),
                '../checkpoint',
                f'epoch_{epoch}_time_{timestamp}'
            )

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(
                self.generator.state_dict(),
                os.path.join(model_path, f'model_{epoch}_time_{timestamp}.pt')
            )

        print('Saving final model......')
        torch.save(self.generator.state_dict(), '%s/model-final.pt' % (self.config.checkpoint_dir))
        print("Training Finished, Total EPOCH = %d" % self.config.epoch)









