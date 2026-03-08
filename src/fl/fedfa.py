import copy
import logging
import math
from .fedbuff import FedBuffClient
from .base import BaseServer
import torch

class FedFAServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        super().__init__(model, test_loader, recorder, params)
        self.initial_wait = True  # 标记是否在等待初始buffer填满

    def recv_msg(self, msg):
        (_, _, client_id, _) = msg
        self.buffer.append(msg)
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[2] for x in self.buffer]
        )
        
        self.total_staleness += self.model_version - msg[3]
        self.client_update_count += 1

        # 如果buffer已满，移除最旧的更新
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            logging.info(f"Removed oldest update, buffer size: {len(self.buffer)}")
        
        # 检查是否需要等待初始buffer填满
        if self.initial_wait and len(self.buffer) < self.buffer_size:
            logging.info(f"Waiting for initial buffer to fill ({len(self.buffer)}/{self.buffer_size})")
            return
            
        # 一旦buffer填满，后续每次收到更新就触发聚合
        self.initial_wait = False
        if not self.aggregation_trigger.triggered:
            self.aggregation_trigger.succeed()

    def aggregate(self):
        def staleness(model_version):
            # 计算每个客户端的过时程度
            # return 1 / math.sqrt(1 + self.model_version - model_version)
            return 1 # 不设置过时衰减
        
        avg_delta = {}
        # 加权平均参数差异
        for key in self.global_model.state_dict().keys():
            layer_sum = torch.stack(
                [delta[key].float() * staleness(model_version) for delta, _, _, model_version in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / len(self.buffer)
        # 应用全局更新（含服务器学习率）
        new_weights = {
            k: avg_delta[k]
            for k in self.global_model.state_dict()
        }
        self.global_model.load_state_dict(new_weights)

        self.aggregation_count += 1
        self.model_version += 1
        
        # 聚合完成，执行记录

        self.check_and_validate()
        self.recorder.record_aggregation(self.env.now, self.model_version)
        self.recorder.aggregation_times.append(self.env.now)
        self.check_stop_condition()

class FedFAClient(FedBuffClient):
    def __init__(self, client_id, base_model, data_loaders, recorder, params, speed_factor):
        super().__init__(client_id, base_model, data_loaders, recorder, params, speed_factor)

    def send_to_server(self, server):
        # 上传前注册
        self.registration(server)

        sample_dict = {}
        for data, target in self.train_data_loader:
            for t in target:
                sample_dict[t.item()] = sample_dict.get(t.item(), 0) + 1
        # 上传参数
        with server.res.request() as req:
            yield req
            server.recv_msg((copy.deepcopy(self.model.state_dict()), sample_dict, self.client_id, self.model_version))


Client = FedFAClient
Server = FedFAServer 