import copy
import logging
import random
import time
from simpy import Environment, Event
import simpy
import torch
import torch.nn.functional as F
 
from ..utils.constants import OPTIMIZERS

from ..config.params import BaseExperimentParams
from ..utils.record import SimulationRecorder

    
class BaseServer:
    def __init__(self, model, test_loader, recorder:SimulationRecorder, params:BaseExperimentParams):
        self.global_model = model
        self.device = next(model.parameters()).device
        self.params = params
        self.recorder = recorder
        self.buffer = []
        self.aggregation_count = 0
        self.model_version = 0
        self.env:Environment = None
        self.stop_event:Event = None

        self.stop_type = self.params.stop_type if hasattr(self.params, 'stop_type') else 'rounds'
        
        self.test_loader = test_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.buffer_size = params.buffer_size
        
        self.client_pool = {}  # {client_id: wakeup_event}
        self.aggregation_trigger = None
        self.validation_interval = params.validation_interval if hasattr(params, 'validation_interval') else 1
        
        self.total_staleness = 0
        self.client_update_count = 0

        if select_method := getattr(self.params, 'select_method', None):
            if select_method == "random_async":
                self.select_clients = self._select_random_async_clients
            elif select_method == "random_sync":
                self.select_clients = self._select_round_sync_clients
            else:
                raise ValueError(f"Unknown client selection method: {select_method}")
        else:
            # 默认使用随机选择方法
            logging.info("No client selection method specified, using default random sync selection.")
            self.select_clients = self._select_round_sync_clients

    def _wake_up_client(self, client_id):
        if event := self.client_pool.get(client_id):
            if not event.triggered:
                event.succeed()
                self.client_pool.pop(client_id)

    def check_and_validate(self):
        # 聚合完成，执行记录
        if self.aggregation_count % self.validation_interval == 0:
            accuracy, loss = self.validate()
            log_str = f"[Round {self.aggregation_count}] Aggregation completed at {self.env.now:.2f}s. Acc {accuracy:.4f} Loss {loss:.4f}. Total staleness: {self.total_staleness}, avg staleness: {self.total_staleness / self.client_update_count:.2f}"
            logging.info(log_str)
            print(log_str)
            self.recorder.record_validation(accuracy, loss, self.aggregation_count, self.client_update_count, self.env.now)
            return accuracy, loss
        return None, None

    def wake_up_clients(self, selected_ids):
        for cid in selected_ids:
            self._wake_up_client(cid)

    def check_aggregation(self):
        return len(self.buffer) >= self.buffer_size
            
    def server_process(self):
        """独立的后台聚合进程"""
        logging.info("Server run process...")
        selected_ids = self.select_clients()
        self.wake_up_clients(selected_ids)
        while True:
            # 等待触发条件
            yield self.aggregation_trigger
            
            # 执行聚合操作
            if self.check_aggregation():
                self.aggregate()
                
                # 选择并唤醒客户端
                selected_ids = self.select_clients()
                self.wake_up_clients(selected_ids)
                
            # 重置触发事件
            self.aggregation_trigger = self.env.event()
            
    def _select_round_sync_clients(self):
        """可选池子中未选过的客户端，随机选择Mc个活跃客户端"""
        all_ids = [client_id for client_id in self.client_pool.keys() if client_id not in self.selected_clients]
        client_per_round = self.params.clients_per_round if self.params.clients_per_round is not None else self.params.num_clients
        if client_per_round <= len(all_ids):
            selected_ids = random.sample(all_ids, client_per_round)
        else:
            selected_ids = all_ids

        # 选择的客户端数量不足时，从可选池中随机选择一些客户端
        if len(selected_ids) < client_per_round:
            rest_set = set(list(self.client_pool.keys())) - set(selected_ids)
            print(f"selected_ids: {len(selected_ids)}, client_per_round: {client_per_round}, rest_set: {len(rest_set)}")
            if len(rest_set) == 0:
                self.selected_clients = set(selected_ids)
                rest_set = set(list(self.client_pool.keys())) - set(self.selected_clients)
            extra_ids = random.sample(rest_set, client_per_round - len(selected_ids))
            selected_ids += extra_ids
            # 更新已选择的客户端集合
            self.selected_clients = set(selected_ids)
        else:
            # 更新已选择的客户端集合
            self.selected_clients |= set(selected_ids)

        logging.info(f"[Round {self.model_version}] Selected clients: {sorted(selected_ids)}")
        return selected_ids
    
    def _select_random_async_clients(self):
        """客户端选择策略：随机选择Mc个活跃客户端"""
        all_ids = list(self.client_pool.keys())

        if not hasattr(self, "selected_client_history"):
            self.selected_client_history = set()
        
        if self.params.clients_per_round is not None:
            # 计算当前剩余未被选中过的客户端
            need_num = self.params.clients_per_round - (self.params.num_clients - len(all_ids))
            unselected_ids = list(set(all_ids) - self.selected_client_history)

            if len(unselected_ids) >= need_num:
                selected_ids = random.sample(unselected_ids, need_num)
            else:
                # 不足 clients_per_round 个未选中过的客户端，则重置历史
                self.selected_client_history = set()
                selected_ids = random.sample(all_ids, need_num)

            # 更新历史
            self.selected_client_history.update(selected_ids)
        else:
            selected_ids = all_ids
        logging.info(f"[Round {self.model_version}] Selected clients: {sorted(selected_ids)}")
        return selected_ids
        
    def registration(self, client_id, wakeup_event):
        """客户端注册接口"""
        self.client_pool[client_id] = wakeup_event

    def init_env(self, env:Environment):
        self.env = env
        self.stop_event = env.event()  # 新增事件初始化
        self.aggregation_trigger = env.event()  # 新增聚合触发事件初始化
        self.res = simpy.Resource(env, capacity=1)

    def send_msg(self, client_id):
        ''' 默认返回全局模型，此处没有深拷贝，请注意使用 '''
        return (self.model_version, self.global_model.state_dict())

    def recv_msg(self, msg):
        raise NotImplementedError

    def aggregate(self):
        raise NotImplementedError

    def check_stop_condition(self):
        if self.stop_type == 'rounds':
            if self.aggregation_count >= self.params.num_rounds:
                self.stop_event.succeed()
        elif self.stop_type == 'time':
            if self.env.now >= self.params.max_time:
                self.stop_event.succeed()
        elif self.stop_type == 'update':
            if self.client_update_count >= self.params.max_updates:
                self.stop_event.succeed()
        else:
            raise ValueError(f"Unknown stop condition: {self.stop_type}")

    def validate(self):
        self.global_model.eval()
        total_correct = 0
        total_loss = 0
        class_num = len(self.test_loader.dataset.dataset.classes)
        class_correct = [0] * class_num
        class_total = [0] * class_num
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.global_model(data)
                total_loss += self.criterion(outputs, targets).item()
                pred = outputs.argmax(dim=1)
                total_correct += pred.eq(targets).sum().item()

                c = (pred == targets).squeeze()
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
    
        # 打印各类别准确率
        logging.info("Class-wise Accuracy:")
        if class_num <= 20:
            for i in range(class_num):
                if class_total[i] > 0:
                    logging.info(f"Class {i}: {100*class_correct[i]/class_total[i]:.2f}%")
        else:
            # 如果类别数量超过20，则一行打印10个
            for i in range(0, class_num, 10):
                class_str = ", ".join([f"Class {j}: {100*class_correct[j]/class_total[j]:.2f}%" for j in range(i, min(i+10, class_num)) if class_total[j] > 0])
                logging.info(class_str)

        accuracy = total_correct / len(self.test_loader.dataset)
        avg_loss = total_loss / len(self.test_loader)
        return accuracy, avg_loss


class BaseClient:
    def __init__(self, client_id, 
                 base_model, 
                 data_loaders, 
                 recorder:SimulationRecorder, 
                 params:BaseExperimentParams,
                 speed_factor:float):
        self.client_id = client_id
        self.model = copy.deepcopy(base_model)
        self.device = next(self.model.parameters()).device
        self.params = params
        (self.train_data_loader,
         self.val_data_loader, 
         self.test_data_loader) = data_loaders
        self.model_version = 0
        self.recorder = recorder
        self._init_optimizer()
        self.speed_factor = speed_factor
        if train_method := getattr(self.params, 'train_method', None):
            if train_method == "minibatch":
                self.local_train = self.minibatch_local_train
            elif train_method == "fullbatch":
                self.local_train = self.fullbatch_local_train
            else:
                raise ValueError(f"Unknown training method: {train_method}")


    def _init_optimizer(self):
        opt = OPTIMIZERS[self.params.optimizer]
        self.optimizer = opt(
            self.model.parameters(),
            lr=self.params.learning_rate,
            momentum=self.params.momentum
        )
    
    def local_train_with_time(self):
        start_time = time.time()
        self.local_train()
        real_train_time = time.time() - start_time  # 实际耗时（秒）
        if self.params.speed_mode == "assign":
            return (self.speed_factor) * self.params.local_rounds
        elif self.params.speed_mode == "multi":
            return real_train_time * self.speed_factor
        elif self.params.speed_mode == "add":
            return real_train_time + self.speed_factor

    def minibatch_local_train(self):
        self.model.train()
        # 增加每个 mini-batch 的日志，用于调试本地训练
        # log_every = getattr(self.params, 'log_batch_every', 10)
        for epoch in range(self.params.local_rounds):
            batch_losses = []
            batch_accs = []
            for batch_idx, (images, labels) in enumerate(self.train_data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 记录 batch loss 和 batch accuracy
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    correct = preds.eq(labels).sum().item()
                    acc = correct / labels.size(0) if labels.size(0) > 0 else 0.0

                batch_losses.append(loss.item())
                batch_accs.append(acc)

                # 每隔若干 batch 打印一次，既有 logging 也有 stdout 输出，便于观察
                # if (batch_idx + 1) % log_every == 0:
                #     avg_loss = sum(batch_losses[-log_every:]) / min(len(batch_losses), log_every)
                #     avg_acc = sum(batch_accs[-log_every:]) / min(len(batch_accs), log_every)
                #     msg = (f"Client {self.client_id} LocalEpoch {epoch+1}/{self.params.local_rounds} "
                #            f"Batch {batch_idx+1}/{len(self.train_data_loader)} Loss {avg_loss:.4f} Acc {avg_acc:.4f}")
                #     logging.info(msg)
                #     print(msg)

            # 每个本地 epoch 结束时打印该 epoch 的平均 loss/acc
            # if len(batch_losses) > 0:
            #     epoch_loss = sum(batch_losses) / len(batch_losses)
            #     epoch_acc = sum(batch_accs) / len(batch_accs)
            #     epoch_msg = (f"Client {self.client_id} Finished LocalEpoch {epoch+1}/{self.params.local_rounds} "
            #                  f"AvgLoss {epoch_loss:.4f} AvgAcc {epoch_acc:.4f}")
            #     logging.info(epoch_msg)
            #     print(epoch_msg)

    def fullbatch_local_train(self):
        self.model.train()
        all_images, all_labels = [], []
        for images, labels in self.train_data_loader:
            all_images.append(images)
            all_labels.append(labels)
        images = torch.cat(all_images).to(self.device)
        labels = torch.cat(all_labels).to(self.device)

        for _ in range(self.params.local_rounds):
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
    
    def _client_process(self, server:BaseServer):
        raise NotImplementedError
    
    def registration(self, server:BaseServer):
        self.wakeup_event = server.env.event()
        server.registration(self.client_id, self.wakeup_event)

    def client_process(self, server:BaseServer):
        # 初始化注册
        try:
            while True:
                yield from self._client_process(server)
        except simpy.Interrupt:
            logging.info(f"Client {self.client_id} terminated")
    