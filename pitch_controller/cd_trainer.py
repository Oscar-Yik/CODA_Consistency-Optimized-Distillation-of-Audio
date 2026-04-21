import torch
import torch.nn.functional as F
import copy

from models.consistency import ConsistencyPitcher
from models.unet import UNetPitcher
from torch.amp import autocast


class ConsistencyTrainer:
    def __init__(self, teacher_model, unet_cfg, device='cuda', lr=1e-5, weight_decay=1e-6, ema_mu=0.999, mixed_precision=False):
        self.device = device
        self.teacher = teacher_model.to(device).eval()

        # initialize student with same weights as teacher
        student_unet = UNetPitcher(**unet_cfg).to(device)
        student_unet.load_state_dict(self.teacher.state_dict())

        self.student = ConsistencyPitcher(student_unet).to(device)
        self.target_student = copy.deepcopy(self.student).to(device).eval()

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_mu = ema_mu # how much of the target student parameters we keep each update
        self.losses = []

        self.mixed_precision = mixed_precision

    def update_target_model(self):
        with torch.no_grad():
            # for each parameter, we "blend" the online and target student, making the target a smoothed out version of online student
            for s_param, t_param in zip(self.student.parameters(), self.target_student.parameters()):
                t_param.data.mul_(self.ema_mu).add_(s_param.data, alpha=1 - self.ema_mu)

    def train_step(self, source_x, mean, f0_ref, noise_scheduler, t_idx, sigma_data=0.5):
        self.optimizer.zero_grad()

        batch_size = source_x.shape[0]
        t = noise_scheduler.timesteps[t_idx]
        t_prev = noise_scheduler.timesteps[t_idx + 1] if t_idx + 1 < len(noise_scheduler.timesteps) else 0

        t_batch = torch.tensor([t] * batch_size, device=self.device)
        t_prev_batch = torch.tensor([t_prev] * batch_size, device=self.device)

        noise = torch.randn_like(source_x)
        x_t = noise_scheduler.add_noise(source_x, noise, t_batch)

        #  teacher x_t -> x_{t-1}
        with torch.no_grad():

            if self.mixed_precision:
                with autocast('cuda'):
                    model_output = self.teacher(x=x_t, mean=mean, f0=f0_ref, t=t_batch)
            else: 
                model_output = self.teacher(x=x_t, mean=mean, f0=f0_ref, t=t_batch)

            step_results = noise_scheduler.step(model_output.float(), t, x_t.float())
            x_t_prev = step_results.prev_sample.to(self.device)
            teacher_x0 = step_results.pred_original_sample.to(self.device)

        # online student predicts x_0 from x_t (gradients flow here)
        if self.mixed_precision:
            with autocast('cuda'):
                pred_online = self.student(x_t, t_batch, mean, f0_ref, noise_scheduler)
        else:
            pred_online = self.student(x_t, t_batch, mean, f0_ref, noise_scheduler)

        # target student predicts x_0 from x_{t-1} (EMA model, no gradients)
        with torch.no_grad():
            if self.mixed_precision:
                with autocast('cuda'):
                    pred_target = self.target_student(x_t_prev, t_prev_batch, mean, f0_ref, noise_scheduler)
            else: 
                pred_target = self.target_student(x_t_prev, t_prev_batch, mean, f0_ref, noise_scheduler)

        # Consistency distillation 
        cd_loss = F.huber_loss(pred_online.float(), pred_target.float(), delta=0.5)    
        anchor_loss = F.huber_loss(pred_online.float(), teacher_x0.float().detach(), delta=0.5)
        loss = cd_loss + anchor_loss
        # loss = cd_loss + 2.0 * anchor_loss

        # L1 
        # cd_loss = F.l1_loss(pred_online, pred_target)
        # anchor_loss = F.l1_loss(pred_online, teacher_x0.detach())
        # loss = cd_loss + (2.0 * anchor_loss)

        # Knowledge distillation
        # loss = F.mse_loss(pred_online, teacher_x0.detach())

        loss.backward()

        # Clip them to a maximum norm of 1.0
        # torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.update_target_model()
        
        return loss.item()