import torch
from ProMP_Model import Model
from utils.model import *
from torch.optim.lr_scheduler import StepLR
class ProMPOuter:
    def __init__(self, state_dim, action_dim, env, penalty, device, beta=0.001, lr_step_size: int = 5,
                 lr_gamma: float = 0.9):
        self.device = device
        self.env = env
        # 메타 파라미터(theta)를 위한 모델 초기화
        self.theta = Model(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.theta.parameters(), lr = beta)
        # StepLR 스케줄러: 매 lr_step_size번 update 시마다 lr *= lr_gamma
        self.scheduler = StepLR(self.optimizer,
                                step_size=lr_step_size,
                                gamma=lr_gamma)
        self.kl_penalty = penalty

    def update(self, step, all_inner_thetas, mean_all_kl_loss, tasks, post_updated_trajectories):
        meta_loss_list = []
        
        for idx in range(TASK_BATCH_SIZE):
            self.env.scenario = tasks[idx]
            self.env.reset()
            # 1. Meta-objective 계산: task별 promp 손실 평균, kl-loss 계산
            clip_obj = likelihood(all_inner_thetas[idx] ,self.theta, post_updated_trajectories[idx], self.env, self.device, "outer")
            kl_loss = mean_all_kl_loss[idx]
            meta_loss_list.append(clip_obj - self.kl_penalty * kl_loss) 
        
        meta_loss = -sum(meta_loss_list)

        # 2. Backward 및 θ 업데이트
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        print(f"[Step {step}] Meta Loss: {meta_loss.item():.4f}")
    
