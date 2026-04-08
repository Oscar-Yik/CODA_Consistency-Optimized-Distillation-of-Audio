import torch
import torch.nn.functional as F
import copy

from models.consistency import ConsistencyPitcher
from models.unet import UNetPitcher


class ConsistencyTrainer:
    def __init__(self, teacher_model, unet_cfg, device='cuda', lr=1e-5, weight_decay=1e-6):
        self.device = device
        self.teacher = teacher_model.to('cpu').eval()

        # initialize student with same weights as teacher
        student_unet = UNetPitcher(**unet_cfg).to(device)
        student_unet.load_state_dict(self.teacher.state_dict())

        self.student = ConsistencyPitcher(student_unet).to(device)
        self.target_student = copy.deepcopy(self.student).to(device).eval()

        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_mu = 0.999 # how much of the target student parameters we keep each update
        self.losses = []

    def update_target_model(self):
        with torch.no_grad():
            # for each parameter, we "blend" the online and target student, making the target a smoothed out version of online student
            for s_param, t_param in zip(self.student.parameters(), self.target_student.parameters()):
                t_param.data.mul_(self.ema_mu).add_(s_param.data, alpha=1 - self.ema_mu)

    def train_step(self, source_x, mean, f0_ref, noise_scheduler, t_idx):
        self.optimizer.zero_grad()
        t = noise_scheduler.timesteps[t_idx]
        t_prev = noise_scheduler.timesteps[t_idx + 1] if t_idx + 1 < len(noise_scheduler.timesteps) else 0 # if we r at last step, t_prev is 0

        noise = torch.randn_like(source_x)
        t_batch = torch.tensor([t] * source_x.shape[0], device=self.device)
        x_t = noise_scheduler.add_noise(source_x, noise, t_batch)
        
        #  teacher x_t -> x_{t-1}
        with torch.no_grad():
            x_t_cpu = x_t.to('cpu')
            mean_cpu = mean.to('cpu')
            f0_cpu = f0_ref.to('cpu')
            t_cpu = t.to('cpu') if torch.is_tensor(t) else t

            model_output = self.teacher(x=x_t_cpu, mean=mean_cpu, f0=f0_cpu, t=t_cpu)
            noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to('cpu')
            x_t_prev_cpu = noise_scheduler.step(model_output, t, x_t_cpu).prev_sample

            x_t_prev = x_t_prev_cpu.to(self.device)
            noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(self.device)

        # online student predicts x_0 from x_t (gradients flow here)
        t_gpu = torch.as_tensor(t).to(self.device)
        t_prev_gpu = torch.as_tensor(t_prev).to(self.device)
        pred_online = self.student(x_t, t_gpu, mean, f0_ref, noise_scheduler)

        # target student predicts x_0 from x_{t-1} (EMA model, no gradients)
        with torch.no_grad():
            pred_target = self.target_student(x_t_prev, t_prev_gpu, mean, f0_ref, noise_scheduler)

    
        loss = F.mse_loss(pred_online, pred_target)    
        loss.backward()
        self.optimizer.step()
        self.update_target_model()
        
        return loss.item()