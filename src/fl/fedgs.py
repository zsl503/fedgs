# t2是要33行时间系数不要50行（），t3是不要33行时间系数要50行（非常平稳，但是精度不及其它，比fedbuff好），t4是33行和105行时间系数不要50行（后期乏力）
import logging
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .fedbuff import FedBuffClient

from .base import BaseServer

def layer_can_select(key):
    if 'bn' in key or 'downsample' in key or 'bias' in key:
        return False
    return True

class FedGSServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size
        self.h_t = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.m_t = {key: torch.zeros_like(value, device=self.device).float() for key, value in model.state_dict().items()}
        self.mode = params.mode if hasattr(params, 'mode') else 'full'  # full, local, global

    def recv_msg(self, msg):
        delta, sample, client_id, model_version = msg
        self.buffer.append((delta, sample, client_id, model_version))
        
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[2] for x in self.buffer]
        )
        self.total_staleness += self.model_version - model_version
        self.client_update_count += 1

        # 检查是否达到缓冲区大小
        if not self.aggregation_trigger.triggered and len(self.buffer) >= self.buffer_size:
            self.aggregation_trigger.succeed()

    def aggregate(self):    
        avg_delta = {}
        current_weights = self.global_model.state_dict()
        new_weights = {}  
        for key, value in self.global_model.state_dict().items():
            # 使用累积方式而不是存储所有更新
            layer_sum = torch.zeros_like(value, device=self.device)
            for delta, sample, _, _ in self.buffer:
                layer_sum += delta[key].to(self.device)
            avg_delta[key] = layer_sum / len(self.buffer)

            if not layer_can_select(key):
                new_weights[key] = current_weights[key] - self.params.server_lr * (avg_delta[key])
            else:
                self.m_t[key] = self.params.gamma * self.m_t[key] + (1 - self.params.gamma) * (avg_delta[key])
                denom = max(1 - self.params.gamma ** self.model_version, 1e-3)
                # denom = 1
                m_corrected = self.m_t[key] / denom
                if self.mode == 'global':
                    new_weights[key] = current_weights[key] - self.params.server_lr * self.h_t[key].to(self.device) # Only use global
                elif self.mode == 'local':
                    new_weights[key] = current_weights[key] - self.params.server_lr * (m_corrected) # Only use local
                elif self.mode == 'full':
                    new_weights[key] = current_weights[key] - self.params.server_lr * (m_corrected) - self.params.server_lr * self.h_t[key].to(self.device)
                # new_weights[key] = current_weights[key] - self.params.server_lr * (avg_delta[key]) - self.params.server_lr * self.h_t[key].to(self.device)

        for key, value in self.global_model.state_dict().items():
            if not layer_can_select(key):
                continue
            p = avg_delta[key] * len(self.buffer) / self.params.num_clients
            self.h_t[key] += p

        self.global_model.load_state_dict(new_weights)

        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1
        
        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

    def send_msg(self, client_id):
        ''' 默认返回全局模型，此处没有深拷贝，请注意使用 '''
        if self.model_version == 0:
            return (self.model_version, self.global_model.state_dict())
        else:
            return (self.model_version, self.global_model.state_dict())

class FedGSClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.use_ddiff = params.use_ddiff
        self.old_delta = None
        # self.local_train = self.new_minibatch_local_train

    def new_minibatch_local_train(self):
        self.model.train()
        # print(f"Client {self.client_id} training on {len(self.train_data_loader.dataset)} samples")
        for _ in range (self.params.local_rounds):
            loop = tqdm(self.train_data_loader,
                        desc=f"Client {self.client_id} epoch {_+1}/{self.params.local_rounds}",
                        leave=False,
                        total=len(self.train_data_loader))
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
            loop.close()

    def recv_from_server(self, server):
        self.model_version, state_dict = server.send_msg(self.client_id)
        self.model.load_state_dict(state_dict)
        
    def send_to_server(self, server):
        if self.use_ddiff: # 使用梯度差分
            delta = {}
            ddelta = {}
            for k in self.initial_state:
                delta[k] = (self.initial_state[k] - self.model.state_dict()[k]).cpu()
                ddelta[k] = delta[k].cpu() - self.old_delta[k] \
                    if self.old_delta is not None and layer_can_select(k) else delta[k].detach()

            self.old_delta = delta

            # 上传前注册
            self.registration(server)
            # 上传参数
            with server.res.request() as req:
                yield req
                server.recv_msg((ddelta, len(self.train_data_loader.dataset), self.client_id, self.model_version))

        else:
            delta = {}
            for k in self.initial_state:
                delta[k] = (self.initial_state[k] - self.model.state_dict()[k]).cpu()

            # 上传前注册
            self.registration(server)
            # 上传参数
            with server.res.request() as req:
                yield req
                server.recv_msg((delta, len(self.train_data_loader.dataset), self.client_id, self.model_version))

Client = FedGSClient
Server = FedGSServer
