from collections import deque
import json
import os
from typing import Iterable
from matplotlib import animation, pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Rectangle

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

class SimulationRecorder:
    def __init__(self, num_clients, use_tensorboard=True, tensorboard_dir='logs'):
        self.events = []
        # TensorBoard 初始化
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.client_metrics = {
            i: {'BTm': 0.0, 'wait_time': 0.0} 
            for i in range(num_clients)
        }
        self.aggregation_times = []
        self.round_record = {
            "accuracy": [],
            "loss": [],
            "send_bytes": [],
            "recv_bytes": [],
        }
               
    def visualize_client_times(self, file_path='client_times.png'):
        client_ids = sorted(self.client_metrics.keys())
        total_times = [
            self.client_metrics[i]['BTm'] + self.client_metrics[i]['wait_time'] 
            for i in client_ids
        ]
        BTms = [self.client_metrics[i]['BTm'] for i in client_ids]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(client_ids))
        
        # 绘制总时间虚线柱
        ax.bar(x, total_times, width=0.6, 
               edgecolor='black', linestyle='--', 
               hatch='//', alpha=0.5, label='Total Time')
        
        # 覆盖训练时间实心柱
        ax.bar(x, BTms, width=0.4,
               color='tab:orange', label='Training Time')
        
        ax.set_xticks(x)
        ax.set_xticklabels(client_ids)
        ax.set_xlabel('Client ID')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Client Training Time Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    def record_training_time(self, client_id, duration):
        self.client_metrics[client_id]['BTm'] += duration
        
    def record_waiting_time(self, client_id, duration):
        self.client_metrics[client_id]['wait_time'] += duration
        
    def record_validation(self, accuracy, loss, current_round, update_cnt, time):
        self.writer.add_scalar('Validation-Round/Accuracy', accuracy, current_round)
        self.writer.add_scalar('Validation-Round/Loss', loss, current_round)

        self.writer.add_scalar('Validation-Time(ms)/Accuracy', accuracy, time*1000)
        self.writer.add_scalar('Validation-Time(ms)/Loss', loss, time*1000)

        self.writer.add_scalar('Validation-Update/Accuracy', accuracy, update_cnt)
        self.writer.add_scalar('Validation-Update/Loss', loss, update_cnt)

    def record_overhead(self, time, current_round, recv_byte, send_byte, accuracy, loss):
        send_GB = send_byte / 1073741824 # 1024*1024*1024
        recv_GB = recv_byte / 1073741824
        self.writer.add_scalar('Round/Recv_GB', recv_GB, current_round)
        self.writer.add_scalar('Round/Send_GB', send_GB, current_round)
        self.writer.add_scalar('Round/Total_GB', recv_GB + send_GB, current_round)

        self.writer.add_scalar('Time(ms)/Recv_GB', recv_GB, time*1000)
        self.writer.add_scalar('Time(ms)/Send_GB', send_GB, time*1000)
        self.writer.add_scalar('Time(ms)/Total_GB', recv_GB + send_GB, time*1000)

        if accuracy is not None:  
            self.writer.add_scalar('Recv_MB/Accuracy', accuracy, recv_byte/(1024*1024))
            self.writer.add_scalar('Send_MB/Accuracy', accuracy, send_byte/(1024*1024))
            self.writer.add_scalar('Total_MB/Accuracy', accuracy, (send_byte+recv_byte)/(1024*1024))
        
        self.round_record["accuracy"].append(accuracy)
        self.round_record["loss"].append(loss)
        self.round_record["send_bytes"].append(send_byte)
        self.round_record["recv_bytes"].append(recv_byte)

    def record_client_status(self, time, client_id, **kwargs):
        status, model_version, speed_factor = kwargs['status'], kwargs['model_version'], kwargs['speed_factor']
        if 'window_id' in kwargs or 'BTm' in kwargs or 'LDu' in kwargs:
            window_id, BTm, LDu = kwargs['window_id'], kwargs['BTm'], kwargs['LDu']
            self.events.append(('client', time, client_id, status, model_version, speed_factor, window_id, BTm, LDu))
        else:
            self.events.append(('client', time, client_id, status, model_version, speed_factor))

    def record_buffer_update(self, time, buffer_state, *args):
        self.events.append(('buffer', time, buffer_state, *args))
    
    def record_aggregation(self, time, new_version):
        self.events.append(('aggregate', time, new_version))
    
    def record_window_change(self, time, cur_window_num):
        self.events.append(('window_change', time, cur_window_num))

    def save(self, save_dir):
        # 保存事件到文件，格式为json
        os.makedirs(save_dir, exist_ok=True)
        json.dump(self.events, open(os.path.join(save_dir, 'event.json'), 'w'), indent=4, ensure_ascii=False)
        json.dump(self.round_record, open(os.path.join(save_dir, 'round_record.json'), 'w'), indent=4, ensure_ascii=False)

    def load(self, save_dir):
        # 从文件加载事件
        self.events = open(os.path.join(save_dir, 'event.json'), 'r')
        self.round_record = open(os.path.join(save_dir, 'round_record.json'), 'r')
  
# 动画生成类
class FLAnimator:
    def __init__(self, recorder, num_clients, buffer_size, time_scale=1.0):
        """
        :param time_scale: 时间加速系数 (1.0表示实时，2.0表示2倍速)
        """
        self.base_fps = 10  # 基础帧率（0.1秒/帧 → 10帧/秒）
        self.fps = self.base_fps * time_scale

        self.recorder = recorder
        self.num_clients = num_clients
        self.buffer_size = buffer_size
        self.time_scale = time_scale
        
        # 预处理事件队列
        self.event_queue = deque(sorted(recorder.events, key=lambda x: x[1]))
        self.max_time = max(e[1] for e in recorder.events) if recorder.events else 0
        
        # 计算动画参数
        self.total_frames = int(self.max_time * 10)  # 每0.1秒一个帧（可根据需要调整）        
        self._init_plot()

    def _init_plot(self):
        # 初始化绘图
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 8))
        self.time_text = self.fig.text(0.5, 0.95, '', ha='center')
        
        # 客户端状态图初始化
        self.client_rects = []
        self.client_texts = []
        for i in range(self.num_clients):
            rect = Rectangle((0, i*0.8), 1, 0.7, ec='black', fc='white')
            self.ax1.add_patch(rect)
            text = self.ax1.text(0.2, i*0.8+0.35, f'Client {i:3}     {"Idle":15}     Model v{"0":3}', 
                               ha='left', va='center')
            self.client_rects.append(rect)
            self.client_texts.append(text)
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, self.num_clients*0.8)
        self.ax1.axis('off')
        self.ax1.set_title('Clients Status')
        
        # 服务端缓冲区图初始化
        self.buffer_rects = []
        self.buffer_texts = []
        for i in range(self.buffer_size):
            rect = Rectangle((i*1.2, 0), 1, 1, ec='black', fc='white')
            self.ax2.add_patch(rect)
            text = self.ax2.text(i*1.2+0.5, 0.5, 'Empty', 
                               ha='center', va='center')
            self.buffer_rects.append(rect)
            self.buffer_texts.append(text)
        self.ax2.set_xlim(0, self.buffer_size*1.2)
        self.ax2.set_ylim(0, 1)
        self.ax2.axis('off')
        self.ax2.set_title('Server Buffer')
        
        # 全局模型版本显示
        self.model_version_text = self.fig.text(0.5, 0.05, 
                                              'Global Model Version: 0', 
                                              ha='center')
    
    def update(self, frame):
        current_time = frame * self.max_time / self.total_frames
    
        # 处理所有当前时间之前的事件
        while self.event_queue and self.event_queue[0][1] <= current_time:
            event = self.event_queue.popleft()
            
            if event[0] == 'client':
                _, time, client_id, status, model_version, speed_factor = event
                color = {'idle': 'white', 
                        'downloading': 'yellow',
                        'training': 'orange'}.get(status, 'gray')
                text = f'Client {client_id:3}     {status:15}     Model v{model_version:3}     speed factor {speed_factor:3}'
                self.client_rects[client_id].set_facecolor(color)
                self.client_texts[client_id].set_text(text)
                
            elif event[0] == 'buffer':
                _, time, buffer_state = event
                for i in range(self.buffer_size):
                    if i < len(buffer_state):
                        client_id = buffer_state[i]
                        self.buffer_rects[i].set_facecolor('lightblue')
                        self.buffer_texts[i].set_text(f'Client\n{client_id}')
                    else:
                        self.buffer_rects[i].set_facecolor('white')
                        self.buffer_texts[i].set_text('Empty')
                        
            elif event[0] == 'aggregate':
                _, time, new_version = event
                self.model_version_text.set_text(
                    f'Global Model Version: {new_version}')
        
        self.time_text.set_text(f'Simulation Time: {current_time:.1f}s (x{self.time_scale})')
        return self.client_rects + self.client_texts + self.buffer_rects + \
               self.buffer_texts + [self.time_text, self.model_version_text]

    def animate(self):
        # 计算实际帧间隔（考虑时间加速）
        
        ani = animation.FuncAnimation(
            self.fig, 
            self.update,
            frames=self.total_frames,
            interval=self.total_frames/(self.base_fps * self.time_scale),
            blit=True
        )
        plt.close()
        return ani
  
class AdvFLAnimator(FLAnimator):
    def __init__(self, recorder, num_clients = 10, max_buffer_size = 10, time_scale=1.0, max_window_size = 3):
        self.max_buffer_size = max_buffer_size
        self.max_window_size = max_window_size
        self.win_buffer_rects = {}
        self.win_buffer_texts = {}
        self.client_rects = {}
        self.client_texts = {}
        self.fig = None
        if max_buffer_size > 25:
            self.buffers_per_row = 25
        else:
            self.buffers_per_row = max_buffer_size
        super().__init__(recorder, num_clients, max_buffer_size, time_scale=time_scale)

    def _init_plot(self):
        # 初始化绘图
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 10), 
                                                      gridspec_kw={'width_ratios': [0.5, 0.5]})
        self.time_text = self.fig.text(0, 0.97, f'Simulation Time: {0:.1f}s (x{1})', ha='left', va='top')
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=0.97)
        self.ax1.invert_yaxis()
        self.ax2.invert_yaxis()

        # 客户端状态图初始化
        # 客户端状态图初始化（双列模式）
        self.client_rects = {}
        self.client_texts = {}
        margin = 0.05
        left_count = (self.num_clients + 1) // 2  # 左列数量（向上取整）

        for i in range(self.num_clients):
            client_id = i
            if i < left_count:
                col = 0  # 左列
                row = i
                x = margin
                id_x = 0.0  # 客户端ID显示位置
            else:
                col = 1  # 右列
                row = i - left_count
                x = 0.5 + margin
                id_x = 0.5  # 右列ID起始位置
            
            y = row * 0.8  # 垂直位置
            rect_width = 0.5 - 2 * margin  # 列宽计算
            rect = Rectangle((x, y), rect_width, 0.7, ec='black', fc='white')
            self.ax1.add_patch(rect)
            
            # 创建状态文本（向右偏移0.02）
            text_x = x + 0.01
            status_text = self.ax1.text(
                text_x, y + 0.35,
                f'w{0:<3}|S:x{1:<5.2f}|BTm:{0:>8.2f}|LDu:{0:>8.2f}|{"Idle":<8}',
                ha='left', va='center'
            )
            
            # 创建客户端ID文本
            self.ax1.text(id_x, y + 0.35, f'C {client_id}', ha='left', va='center')
            
            self.client_rects[client_id] = rect
            self.client_texts[client_id] = status_text

        # 设置坐标轴参数
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(max(left_count, self.num_clients - left_count) * 0.8, 0 )
        self.ax1.axis('off')
        self.ax1.set_title('Clients Status')

        # # 服务端缓冲区图初始化（分块模式）
        self.win_buffer_rects = {}
        self.win_buffer_texts = {}
        margin = 1.2
        spacing = 0.5         # 窗口间分隔线高度
        current_y = 0         # 当前绘制起始高度

        for w in range(self.max_window_size):
            window_id = w
            if window_id not in self.win_buffer_rects:
                self.win_buffer_rects[window_id] = []
                self.win_buffer_texts[window_id] = []
            # 计算当前窗口需要多少行
            rows_needed = (self.max_buffer_size + self.buffers_per_row - 1) // self.buffers_per_row
            
            # 绘制窗口标签（居中显示）
            window_height = rows_needed * 0.8
            self.ax2.text(-margin, current_y + window_height/2, 
                        f'W {window_id}', ha='left', va='center')
            
            # 绘制缓冲区块
            for row in range(rows_needed):
                y = current_y + row * 0.8
                for col in range(self.buffers_per_row):
                    idx = row * self.buffers_per_row + col
                    if idx >= self.max_buffer_size:
                        break
                    # 绘制矩形
                    rect = Rectangle((col*1.2, y), 1, 0.7, ec='black', fc='white')
                    self.ax2.add_patch(rect)
                    # 添加文本
                    text = self.ax2.text(col*1.2+0.5, y+0.4, 'E', 
                                    ha='center', va='center')
                    self.win_buffer_rects[window_id].append(rect)
                    self.win_buffer_texts[window_id].append(text)
            
            # 绘制窗口间分隔线（跳过最后一个窗口）
            if w < self.max_window_size - 1:
                self.ax2.hlines(current_y + window_height + spacing/2, 
                            -margin, self.buffers_per_row*1.2,
                            colors='gray', linestyle='--', linewidth=1)
            
            # 更新垂直位置
            current_y += window_height + spacing

        # 设置坐标轴范围
        self.ax2.set_xlim(-margin, self.buffers_per_row*1.2)
        self.ax2.set_ylim(current_y - spacing, 0)  # 减去最后一个间隔
        self.ax2.axis('off')
        self.ax2.set_title(f'Server Buffer (Current Model Version: v{0})')
    
    def update_plot(self, client_states=None, buffer_states=None, cur_window_num=None, cur_model_version=None, current_time=None) -> Iterable[Artist]:
        """
        更新绘图数据的接口方法
        参数格式：
        client_states = {
            'C0': {'version': 3, 'BTm': 5.2, 'LDu': 1.5, 'status': 'Training'},
            'C1': {'version': 2, 'BTm': 4.8, 'LDu': 1.2, 'status': 'Idle'},
            # ...其他客户端状态...
        }
        
        buffer_states = {
            'W0': {
                'positions': {
                    0: 'A',   # 缓冲区位置0的字符
                    1: 'B',   # 缓冲区位置1的字符
                    # ...其他位置...
                },
                'color_map': {
                    0: 'lightgreen',  # 位置0的颜色
                    1: 'lightblue'    # 位置1的颜色
                },
                'cur_max_buffer': 5,  # 当前最大缓冲区位置
            },
            # ...其他窗口状态...
        }
        """
        # 更新客户端状态
        if client_states:
            for client_id, state in client_states.items():
                # 将客户端ID转换为索引（例如 C3 -> 3）
                idx = int(client_id[1:])
                if 0 <= idx < len(self.client_texts):
                    # 构建状态字符串
                    BTm_str = f"{state['BTm']:>8.2f}" if state.get('BTm') is not None else f"{'':8}"
                    LDu_str = f"{state['LDu']:>8.2f}" if state.get('LDu') is not None else f"{'':8}"
                    status_str = (
                        f"w{state['window']:<3}|"
                        f"S:x{state['speed']:<5.2f}|"
                        f"BTm:{BTm_str}|"
                        f"LDu:{LDu_str}|"
                        f"{state.get('status', 'Idle'):<8}"
                    )
                    # 更新文本内容
                    self.client_texts[idx].set_text(status_str)
                    # 更新矩形颜色
                    self.client_rects[idx].set_facecolor(state.get('color', 'white'))

        # 更新缓冲区状态
        if buffer_states:
            for window_id, win_state in buffer_states.items():
                # 计算窗口起始索引
                win_num = int(window_id[1:])
                if 0 <= win_num < self.max_window_size:
                    win_cur_max_buffer = win_state.get('cur_max_buffer', self.max_buffer_size)
                    # 大于最大位置的缓冲区隐藏
                    for idx in range(0, self.max_buffer_size):
                        self.win_buffer_texts[win_num][idx].set_visible(idx < win_cur_max_buffer)
                        self.win_buffer_rects[win_num][idx].set_visible(idx < win_cur_max_buffer)
                        if idx in win_state.get('color_map', {}):
                            self.win_buffer_rects[win_num][idx].set_facecolor(win_state['color_map'][idx])
                        else:
                            self.win_buffer_rects[win_num][idx].set_facecolor('white')
                        if idx in win_state.get('positions', {}):
                            self.win_buffer_texts[win_num][idx].set_text(win_state['positions'][idx])
                        else:
                            self.win_buffer_texts[win_num][idx].set_text('E')
        
            cur_window_num = cur_window_num if cur_window_num is not None else self.max_window_size
            # 隐藏多余窗口
            for window_id in range(cur_window_num, self.max_window_size):
                for idx in range(0, self.max_buffer_size):
                    self.win_buffer_texts[window_id][idx].set_visible(False)
                    self.win_buffer_rects[window_id][idx].set_visible(False)
        if cur_model_version is not None:
            self.ax2.set_title(f'Server Buffer (Current Model Version: v{cur_model_version})')
        self.time_text.set_text(f'Simulation Time: {current_time:.1f}s (x{self.time_scale})')

        return list(self.client_texts.values()) + list(self.client_rects.values()) + \
                    [text for texts in self.win_buffer_texts.values() for text in texts] + \
                    [rect for rects in self.win_buffer_rects.values() for rect in rects] + \
                    [self.time_text, self.ax2.title]

    def update(self, frame):
        current_time = frame * self.max_time / self.total_frames
        # 处理所有当前时间之前的事件
        client_states = {}
        buffer_states = {}
        cur_window_num = None
        cur_model_version = None
        while self.event_queue and self.event_queue[0][1] <= current_time:
            event = self.event_queue.popleft()
            
            if event[0] == 'client':
                window_id, BTm, LDu = None, None, None
                if len(event) == 6:
                    _, time, client_id, status, model_version, speed_factor = event
                else:
                    _, time, client_id, status, model_version, speed_factor, window_id, BTm, LDu = event
                color = {'idle': 'white', 
                        'down': 'yellow',
                        'train': 'orange'}.get(status, 'gray')
                client_states[f'C{client_id}'] = {
                    'speed': speed_factor,
                    'status': status if len(status) < 8 else status[:8],
                    'window': window_id,
                    'BTm': BTm,
                    'LDu': LDu,
                    'color': color
                }
                
            elif event[0] == 'buffer':
                _, time, buffer_state, window_id, cur_max_buffer = event
                # buffer_states: [client_id, ...]
                # print(time, buffer_state)
                buffer_states[f'W{window_id}'] = {
                    'positions': {i: f'C{cid}' for i, cid in enumerate(buffer_state)},
                    'color_map': {i: 'lightblue' for i, cid in enumerate(buffer_state)},
                    'cur_max_buffer': cur_max_buffer
                }
            elif event[0] == 'aggregate':
                _, time, new_version = event
                cur_model_version = new_version
            elif event[0] == 'window_change':
                _, time, cur_window_num = event
                cur_window_num = cur_window_num
            elif event[0] == 'client_info':
                _, time, client_id = event
                if f'C{client_id}' not in client_states:
                    client_states[f'C{client_id}'] = {
                    'speed': 1,
                    }
                client_states[f'C{client_id}']['window'] = window_id
                client_states[f'C{client_id}']['BTm'] = BTm
                client_states[f'C{client_id}']['LDu'] = LDu
        
        return self.update_plot(client_states, buffer_states, cur_window_num, cur_model_version, current_time)
