import copy
import logging
import math
import random
import time

from .base import BaseServer, BaseClient
import torch

class FedBuffServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.buffer_size = params.buffer_size


    def recv_msg(self, msg):
        (_, _, client_id, model_version) = msg
        self.buffer.append(msg)
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[2] for x in self.buffer]
        )
        self.client_update_count += 1
        self.total_staleness += self.model_version - model_version
        
        # 检查缓冲区并聚合
        if not self.aggregation_trigger.triggered:
            self.aggregation_trigger.succeed()

    def aggregate(self):
        def staleness(model_version):
            # 计算每个客户端的过时程度
            return 1 / math.sqrt(1 + self.model_version - model_version)
            # return 1
        
        avg_delta = {}
        # 输出全部过时程度
        logging.info(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")
        print(f"Staleness: {[self.model_version - model_version for _, _, _, model_version in self.buffer]}")
        # 加权平均参数差异
        for key in self.global_model.state_dict().keys():
            layer_sum = torch.stack(
                [delta[key].float() * staleness(model_version) for delta, _, _, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)
        # 应用全局更新（含服务器学习率）
        current_weights = self.global_model.state_dict()
        new_weights = {
            k: current_weights[k] - self.params.server_lr * avg_delta[k].to(self.device)
            for k in current_weights
        }
        self.global_model.load_state_dict(new_weights)

        self.buffer.clear()
        self.aggregation_count += 1
        self.model_version += 1

        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

class FedBuffClient(BaseClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)
        self.initial_state = None
        self.server = None

    def recv_from_server(self, server):
        self.model_version, state_dict = server.send_msg(self.client_id)
        self.model.load_state_dict(state_dict)

    def send_to_server(self, server):
        delta = {}
        for k in self.initial_state:
            delta[k] = self.initial_state[k] - self.model.state_dict()[k]

        # 上传前注册
        self.registration(server)
        # 上传参数
        with server.res.request() as req:
            yield req
            server.recv_msg((delta, len(self.train_data_loader.dataset), self.client_id, self.model_version))

    def _client_process(self, server: BaseServer):
        # ========== 等待阶段 ==========
        idle_begin_time = server.env.now
        self.recorder.record_client_status(
            time = server.env.now, 
            client_id = self.client_id, 
            type = 'client',
            status = 'idle', 
            model_version = self.model_version,
            speed_factor=self.speed_factor
        )
        
        yield self.wakeup_event
        # 网络延迟
        if self.params.use_random_delay:
            yield server.env.timeout(1)

        # ========== 下载阶段 ==========
        self.recv_from_server(server)
        self.recorder.record_client_status(
            server.env.now,
            self.client_id,
            type = 'client',
            status = 'downloading',
            model_version = server.model_version,
            speed_factor=self.speed_factor
        )
        real_idle_time = server.env.now - idle_begin_time  # 实际耗时（秒）
        self.recorder.record_waiting_time(self.client_id, real_idle_time)
        
        # ========== 训练阶段 ==========
        # 开始训练
        self.recorder.record_client_status(
            server.env.now, 
            self.client_id,
            type = 'client',
            status = 'training',
            model_version = server.model_version,
            speed_factor=self.speed_factor
        )
        
        # 执行本地训练并获取真实时间
        self.model.cpu()
        self.initial_state = copy.deepcopy(self.model.state_dict())
        self.model.to(self.device)
        train_time = self.local_train_with_time()
        self.model.cpu()

        logging.info(f"Client {self.client_id} train time: {train_time}, factor: {self.speed_factor}, mode: {self.params.speed_mode}")        
        self.recorder.record_training_time(self.client_id, train_time)
        yield server.env.timeout(train_time)
        yield from self.send_to_server(server)



Client = FedBuffClient
Server = FedBuffServer