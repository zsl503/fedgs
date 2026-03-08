import copy
import logging
import math
import random
import time
import torch
from .base import BaseClient, BaseServer
from .fedbuff import FedBuffClient

class CA2FLServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size
        self.h_t = {}
        self.h_t_i = {}
        self.delta = {}
        self.use_stale = params.use_stale
        if self.use_stale:
            print("Using stale correction")
        else:
            print("Not using stale correction")

    def is_special_layer(self, key):
        # return False
        return 'bn' in key.lower() or 'downsample' in key.lower()

    def recv_msg(self, msg):
        def staleness(model_version):
            # 计算每个客户端的过时程度
            if self.use_stale:
                return 1 / math.sqrt(1 + self.model_version - model_version)
            else:
                return 1 
        
        delta, _, client_id, model_version = msg
        if client_id not in self.h_t_i:
            self.h_t_i[client_id] = {}

        for k in delta:
            delta[k] = delta[k].cpu().float()
            # 特殊层直接存入buffer，不进行缓存处理
            if self.is_special_layer(k):
                continue
                
            if k not in self.delta:
                self.delta[k] = torch.zeros_like(delta[k], device='cpu').float()
            d = delta[k] - self.h_t_i[client_id].get(k, torch.zeros_like(delta[k]))
            self.delta[k] += staleness(model_version)*d.cpu()
            self.h_t_i[client_id][k] = delta[k].clone()

        self.buffer.append(msg)
        
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
        # 计算全局校准变量
        v_t = {}
        state_dict = self.global_model.state_dict()
        # total_samples = sum([num_samples for (_, num_samples, _, _) in self.buffer])
        logging.info(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")
        print(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")

        # 分别处理普通层和特殊层
        for k in state_dict.keys():
            if self.is_special_layer(k):
                # 特殊层直接平均
                layer_sum = torch.stack([delta[k].cpu().float() for delta, _, _, _ in self.buffer]).sum(0)
                v_t[k] = layer_sum / len(self.buffer)
            else:
                # 普通层使用原有的缓存机制
                v_t[k] = (self.delta[k] / self.buffer_size) + self.h_t.get(k, torch.zeros_like(state_dict[k], device='cpu'))
        
        new_weights = {
            k: state_dict[k] - self.params.server_lr * v_t[k].to(state_dict[k].device)
            for k in v_t.keys()
        }
        self.global_model.load_state_dict(new_weights, strict=False)

        for k in state_dict.keys():
            if self.is_special_layer(k):
                continue
            self.h_t[k] = torch.zeros_like(state_dict[k], device='cpu').float()
            for client_id in self.h_t_i:
                self.h_t[k] += self.h_t_i[client_id][k]
            self.h_t[k] = self.h_t[k] / self.params.num_clients
      
        # 重置当前轮次的状态
        self.delta = {}
        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1
        
        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

Client = FedBuffClient
Server = CA2FLServer