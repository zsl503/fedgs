import logging
import torch

from .base import BaseServer
from .fedbuff import FedBuffClient

class FedAvgServer(BaseServer):
    def __init__(self, model, test_loader, recorder, params):
        """
        :param params: 包含以下参数的配置对象
            - num_clients: 总客户端数量
            - participation_ratio: 客户端参与比例 (默认1.0表示全参与)
            - server_lr: 服务器学习率
        """
        super().__init__(model, test_loader, recorder, params)
        
        # 动态缓冲区管理
        self.selected_clients = set()
        self.num_clients = self.params.num_clients

    def recv_msg(self, msg):
        (delta, _, client_id, _) = msg
        if client_id not in self.selected_clients:
            logging.info(f"Warning: Unexpected update from client {client_id}. Selected {self.selected_clients}")
            return
        
        self.selected_clients.remove(client_id)
        self.buffer.append(msg)
        logging.info(f"[Client {client_id}] Update uploaded ({len(self.buffer)}/{self.buffer_size}) at {self.env.now:.2f}s")
        self.recorder.record_buffer_update(
            self.env.now,
            [x[2] for x in self.buffer]
        )
        self.total_staleness += self.model_version - msg[3]
        self.client_update_count += 1
        

        # 检查缓冲区并聚合
        if not self.aggregation_trigger.triggered:
            self.aggregation_trigger.succeed()
    
    def send_msg(self, client_id):
        ''' 默认返回全局模型，此处没有深拷贝，请注意使用 '''
        state_dict = self.global_model.state_dict()
        # 计算发送数据大小
        return (self.model_version, state_dict)

    def aggregate(self):
        """执行加权平均聚合"""
        total_samples = sum(samples for _, samples, _, _ in self.buffer)
        # 加权平均参数差异
        avg_delta = {}

        for key in self.global_model.state_dict().keys():
            layer_sum = torch.stack(
                [delta[key].float() * samples for delta, samples, _, _ in self.buffer]
            ).sum(0)
            avg_delta[key] = layer_sum / total_samples
        # 应用全局更新（含服务器学习率）
        current_weights = self.global_model.state_dict()
        new_weights = {
            k: current_weights[k] - self.params.server_lr * avg_delta[k]
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

FedAvgClient = FedBuffClient

Client = FedBuffClient
Server = FedAvgServer