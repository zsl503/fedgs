import datetime
import json
import os

from src.models.models import MODELS
os.environ['NUMEXPR_MAX_THREADS'] = "16"

import random
from matplotlib import pyplot as plt
import numpy as np
import simpy
import importlib
from torch.utils.data import DataLoader

import torch
from src.fl.base import BaseClient, BaseServer
from src.config.params import BaseExperimentParams
from src.utils.data_loader import DataHandler
from src.utils.record import AdvFLAnimator, SimulationRecorder, FLAnimator
import logging


# 配置随机种子保证可重复性
def fix_random_seed(seed: int) -> None:
    """Fix the random seed of FL training.

    Args:
        seed (int): Any number you like as the random seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def interrupt_handler(env, stop_event, process):
    yield stop_event
    process.interrupt()

def main(params:BaseExperimentParams, output_dir:str):
    logging.info("\nParams:\n"+str(params))
    recorder = SimulationRecorder(num_clients = params.num_clients, use_tensorboard=True, tensorboard_dir=output_dir)
    
    if params.device != "cpu" and torch.cuda.is_available():
        gpu_id = int(params.device.split(":")[-1])
        if gpu_id >= torch.cuda.device_count():
            logging.info(f"GPU ID {gpu_id} is out of range, using CPU instead.")
            params.device = "cpu"
    else:
        logging.info("GPU is not available, using CPU instead.")
        params.device = "cpu"
    # 加载数据
    (train_client_datasets, 
     val_client_datasets, 
     test_client_datasets) = \
        DataHandler.load_data(
        dataset_name = params.dataset_name, 
        file_dir = os.path.join(params.dataset_dir, params.dataset_name),
        args = params.dataset_args,
        center_test = False
    )

    # 加载测试数据集
    test_dataset = DataHandler.load_data(
        dataset_name = params.dataset_name, 
        file_dir = os.path.join(params.dataset_dir, params.dataset_name),
        args = params.dataset_args,
        center_test=True
    )
    print(f"Test dataset samples: {len(test_dataset)}")
    # 输出测试集中各标签数量
    # if hasattr(test_dataset, 'total_targets'):
    # print(len(test_dataset.dataset.test_targets))
    unique, counts = np.unique(test_dataset.dataset.total_targets[test_dataset.indices], return_counts=True)
    label_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Test dataset label distribution: {label_counts}")
    # return
    # 初始化模型
    global_model = MODELS[params.model_name](dataset = params.dataset_name).to(params.device)
    # 输出模型所有层名称、参数数据类型和参数的形状
    logging.info(f"Model {params.model_name} initialized")
    for name, param in global_model.named_parameters():
        logging.info(f"{name}: {param.data.dtype} {param.data.shape}")
    # 初始化算法
    algorithm_module = importlib.import_module(f"src.fl.{params.algorithm.lower()}")
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size)
    # logging.info(len(test_dataset))
    server:BaseServer = algorithm_module.Server(global_model, test_loader, recorder, params)
    
    # 创建仿真环境
    env = simpy.Environment()
    server.init_env(env)
    # 创建客户端进程
    clients = []
    for i in range(params.num_clients):
        # 初始化客户端
        # speed_factor = np.random.choice(params.speed_factors)
        # print(f"Client {i} train num_samples: {len(train_client_datasets[i])}, val num_samples: {len(val_client_datasets[i])}, test num_samples: {len(test_client_datasets[i])}")
        client:BaseClient = algorithm_module.Client(
            client_id=i,
            base_model=global_model,
            data_loaders = (
                DataLoader(train_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True),
                DataLoader(val_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True) if len(val_client_datasets[i]) else None,
                DataLoader(test_client_datasets[i], batch_size=params.batch_size, shuffle=True, drop_last=True) if len(test_client_datasets[i]) else None,
                ),
            recorder=recorder, 
            params=params,
            speed_factor=params.speed_factors[i%len(params.speed_factors)],
        )
        client.registration(server)
        # 包装为SimPy进程
        client_process = env.process(client.client_process(server))     
        # 设置中断处理
        env.process(
            interrupt_handler(env, server.stop_event, client_process))
        
        clients.append(client)

    server_process = env.process(server.server_process())
    env.process(
        interrupt_handler(env, server.stop_event, server_process))
    
    # 运行仿真
    logging.info("Simulation started...")

    try:
        env.run(until=server.stop_event)
    except simpy.Interrupt:
        logging.info("Simulation interrupted by user")
    # except Exception as e:
    #     logging.info(f"Simulation failed: {e}")

    # 保存最终模型
    torch.save(server.global_model.state_dict(), f"{output_dir}/final_global_model.pth")
    logging.info(f"Completed {server.aggregation_count} aggregation rounds")
    
    recorder.save(f"{output_dir}/recorder/")
    logging.info(f"Recorder events saved to {output_dir}/recorder/")


    # # 画图，每个client一张
    # os.makedirs(f"{output_dir}/loss_time", exist_ok=True)
    # # 画出clients的loss_list和time_list
    # for i in range(params.num_clients):
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(clients[i].time_list, clients[i].loss_list, label=f"Client {i}")
    #     plt.legend()
    #     plt.xlabel('Time')
    #     plt.ylabel('Loss')
    #     plt.title(f'Client {i} Loss-Time Curve')
    #     plt.savefig(f"{output_dir}/loss_time/client_{i}_loss_time.png")
    #     plt.close()

    # 最终可视化
    recorder.visualize_client_times(file_path=f"{output_dir}/client_times.png")
    logging.info("Client time visualization saved to client_times.png")
    
    # # 绘制聚合时间线
    plt.figure(figsize=(14, 6))
    plt.plot(recorder.aggregation_times, marker='o')
    plt.xlabel('Aggregation Round')
    plt.ylabel('Simulation Time (s)')
    plt.title('Aggregation Timeline')
    plt.grid(True)
    plt.savefig(f"{output_dir}/aggregation_timeline.png")

    # 生成动画
    # logging.info("Generating animation...")
    # if 'test' not in params.algorithm:
    #     animator = FLAnimator(recorder, params.num_clients, params.buffer_size, time_scale=1)
    # else:
    #     animator = AdvFLAnimator(recorder, params.num_clients, params.buffer_size, time_scale=1, max_window_size=4)
    # ani = animator.animate()
    # try:
    #     # 判断格式MP4是否可用
    #     ani.save(f"{output_dir}/fl_simulation.{params.video_format}", writer="ffmpeg", fps=animator.fps)
    #     logging.info(f"Animation saved to fl_simulation.{params.video_format}")
    # except Exception as e:
    #     logging.info(f"Animation failed: {e}")


def gen_speed_factor(data_dir, lda=0.01, output_dir=None):
    """
    按照fedbuff的指数分布，生成客户端速度因子，使用指数分布生成速度因子，平均值为lda * x，其中x为每个客户端的样本数
    """
    json_file = os.path.join(data_dir, 'all_stats.json')
    stats = json.load(open(json_file, 'r'))
    if 'sample per client' in stats:
        stats.pop('sample per client')

    speeds = []
    for k, v in stats.items():
        speeds.append(np.random.exponential(scale=lda * v['x']))

    if output_dir:
        x_values = [v['x'] for v in stats.values()]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, speeds, alpha=0.7)
        plt.xlabel('Data Number')
        plt.ylabel('Speeds')
        plt.title('Scatter Plot of Data Number vs Speeds')
        plt.savefig(os.path.join(output_dir, 'speed_factor.png'))
 
        # Save the speed factors to a file
        plt.figure(figsize=(10, 6))
        plt.hist(speeds, bins=30, edgecolor='black')
        plt.xlabel('Speed Range')
        plt.ylabel('Number of Clients')
        plt.title('Distribution of Speeds')
        plt.savefig(os.path.join(output_dir, 'speed_factor_distribution.png'))
    return speeds

def gen_speed_factor_mixture(
    data_dir,
    ratio_list,
    speed_list,
    c=0.1,
    use_data_ratio=True,
    output_dir=None,
    seed=None,
):
    """
    生成多峰客户端速度分布（混合正态分布）

    参数：
    - ratio_list: 各类客户端比例，例如 [0.7, 0.2, 0.1]
    - speed_list: 各类客户端基础速度均值，例如 [1.0, 5.0, 20.0]
    - c: 方差系数，sigma = c * mean
    - use_data_ratio: 是否按数据量缩放速度
    - output_dir: 是否保存可视化图像
    - seed: 随机种子（可选）

    返回：
    - speeds: 每个客户端的速度列表
    """
    if seed is not None:
        np.random.seed(seed)

    assert len(ratio_list) == len(speed_list), \
        "ratio_list 和 speed_list 长度必须一致"

    assert abs(sum(ratio_list) - 1.0) < 1e-6, \
        "ratio_list 必须归一化（和为1）"

    # 读取客户端统计信息
    json_file = os.path.join(data_dir, 'all_stats.json')
    stats = json.load(open(json_file, 'r'))

    if 'sample per client' in stats:
        stats.pop('sample per client')

    client_keys = list(stats.keys())
    num_clients = len(client_keys)

    # 按比例划分客户端类别
    category_counts = [int(r * num_clients) for r in ratio_list]

    # 修正整数截断误差
    while sum(category_counts) < num_clients:
        category_counts[np.argmax(ratio_list)] += 1

    # 打乱客户端顺序
    shuffled_indices = np.random.permutation(num_clients)

    speeds = np.zeros(num_clients)

    start = 0
    for cat_id, count in enumerate(category_counts):
        end = start + count
        indices = shuffled_indices[start:end]

        mean_speed = speed_list[cat_id]
        std_speed = c * mean_speed

        # 正态分布采样
        sampled_speeds = np.random.normal(
            loc=mean_speed,
            scale=std_speed,
            size=count
        )

        # 避免出现负速度
        sampled_speeds = np.clip(sampled_speeds, a_min=1e-6, a_max=None)

        speeds[indices] = sampled_speeds
        start = end


    # 是否按数据量缩放
    if use_data_ratio:
        # 数据总量归一化
        total_data = np.array([stats[k]['x'] for k in client_keys])
        # 原始速度总量
        total_speed = speeds.sum()
        for i, k in enumerate(client_keys):
            data_size = stats[k]['x']
            speeds[i] *= data_size / total_data.sum()

        # 最终速度归一化，使总速度与原始总速度一致
        speeds *= total_speed / speeds.sum()

    speeds = speeds.tolist()

    # 可视化
    if output_dir:
        x_values = [stats[k]['x'] for k in client_keys]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, speeds, alpha=0.7)
        plt.xlabel('Data Number')
        plt.ylabel('Speeds')
        plt.title('Data Number vs Speeds (Mixture)')
        plt.savefig(os.path.join(output_dir, 'speed_factor_mixture.png'))

        plt.figure(figsize=(10, 6))
        plt.hist(speeds, bins=30, edgecolor='black')
        plt.xlabel('Speed')
        plt.ylabel('Number of Clients')
        plt.title('Speed Distribution (Mixture)')
        # plt.show()
        plt.savefig(os.path.join(output_dir, 'speed_factor_mixture_distribution.png'))
    return speeds


def update_or_check_param(param_name, arg_value, params, force=False, explicitly_defined=False):
    if not hasattr(params, param_name):
        # 新增参数
        setattr(params, param_name, arg_value)
        print(f"[⚠️ 警告] 参数 `{param_name}` 不在参数类中定义，已动态添加为 `{arg_value}`，"
              f"建议在 `{params.__class__.__name__}` 中显式声明以提高代码安全性。")
    else:
        param_value = getattr(params, param_name)
        if force:
            if arg_value is not None:
                setattr(params, param_name, arg_value)
        else:
            if arg_value is not None and param_value != arg_value:
                raise ValueError(f"[❌ 冲突] `{param_name}` 不一致：参数文件为 `{param_value}`，命令行为 `{arg_value}`。")

    if not explicitly_defined:
        print(f"[⚠️ 警告] 命令行参数 `{param_name}` 未在 ArgumentParser 中显式声明，已自动处理为 `{arg_value}`。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iid', action='store_true', help='使用IID数据分布')
    parser.add_argument('--a', type=float, default=0.5, help='数据集Dirichlet 分布系数')
    
    parser.add_argument('--algo', type=str, default='fedBuffAdv', help='选择联邦学习算法')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--dataset_name', '--ds', type=str, default=None, help='数据集')
    parser.add_argument('--dataset_seed', '--dseed', type=int, default=None, help='数据集种子')

    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--learning_rate', '--lr', type=float, default=None, help='学习率')
    parser.add_argument('--num_clients','--client_num','--c',type=int, default=50, help='客户端数量')
    parser.add_argument('--speed_lda', type=float, default=0.01, help='速度因子lambda值')
    # parser.add_argument('--speed_shape', type=float, default=1, help='速度因子形状参数')

    parser.add_argument('--device', type=str, default=None, help='device')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--param_file', type=str, default=None, help='参数文件')
    parser.add_argument('--post_str', type=str, default='', help='后缀')
    parser.add_argument('--force', action='store_true', help='强制应用参数') 

    parser.add_argument('--speed_ratio_list', type=float, nargs='+', default=[0.5, 0.5], help='多峰速度分布中各类客户端的比例列表')
    parser.add_argument('--speed_speed_list', type=float, nargs='+', default=[5.0, 20.0], help='多峰速度分布中各类客户端的基础速度列表')
    parser.add_argument('--speed_c', type=float, default=0.1, help='多峰速度分布中速度的方差系数')
    parser.add_argument('--speed_use_data_ratio', action='store_true', help='多峰速度分布中是否按数据量缩放速度')

    # Step 2: 提取命令行原始参数
    known_args, unknown_args = parser.parse_known_args()

    # Step 3: 将 unknown_args 转换为字典
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--"):
            key_value = arg[2:].split("=", 1)
            if len(key_value) == 2:
                key, value = key_value
                # 尝试类型转换（简单处理为 int、float、str）
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # 保持字符串
                extra_args[key] = value
            else:
                print(f"[❌ 错误] 未识别的命令行参数格式: {arg}")

    args = known_args
    args_dict = vars(args)
    args_dict.update(extra_args)  # 整合显式参数与额外参数

    args_dict["post_str"] = f"_{args_dict['post_str']}" if args_dict["post_str"] else ""

    # 数据目录和分布字符串
    # Step 5: 数据路径构造
    if not args_dict["iid"]:
        dataset_dir = f'data/noniid-{args_dict["a"]}_{args_dict["num_clients"]}'
        dist_str = f"noniid-{args_dict['a']}{args_dict['post_str']}"
    else:
        dataset_dir = f'data/iid_{args_dict["num_clients"]}'
        dist_str = f"iid{args_dict['post_str']}"
    
    if args_dict["dataset_seed"] is not None:
        dataset_dir += f"_s{args_dict['dataset_seed']}"
        dist_str += f"_s{args_dict['dataset_seed']}"

    params_class_name = f"{args_dict['algo']}Params"
    param_module = importlib.import_module(args_dict['param_file']) \
        if args_dict['param_file'] else importlib.import_module("src.config.default")

    # 动态加载名为params_class_name的类
    params = getattr(param_module, params_class_name)(dataset_dir)
    fix_random_seed(params.seed)
    params.algorithm = args_dict['algo'].lower()

    explicitly_defined_args = {action.dest for action in parser._actions}

    for key, value in args_dict.items():
        if key in {"post_str", "algo", "output_dir", "param_file", "force", "iid", "a", "speed_lda"}:
            continue
        update_or_check_param(
            key,
            value,
            params,
            force=args_dict['force'],
            explicitly_defined=(key in explicitly_defined_args)
        )

    output_dir = f"output/{params.model_name}/{args_dict['algo']}_{dist_str}" \
        if args_dict['output_dir'] is None else \
            os.path.join(args_dict["output_dir"],f"{params.model_name}/{args_dict['algo']}_{dist_str}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if params.speed_factors is None:
        if params.use_mixture_speed:
            # 生成多峰速度因子
            params.speed_factors = gen_speed_factor_mixture(
                os.path.join(params.dataset_dir, params.dataset_name), 
                ratio_list=args_dict["speed_ratio_list"], 
                speed_list=args_dict["speed_speed_list"], 
                c=args_dict["speed_c"],
                use_data_ratio=args_dict["speed_use_data_ratio"],
                output_dir=output_dir,
                seed=args_dict["seed"]
            )
        else:
        # 生成速度因子
            params.speed_factors = gen_speed_factor(
                os.path.join(params.dataset_dir, params.dataset_name), 
                lda=args_dict["speed_lda"], 
                # shape=args_dict["speed_shape"],
                output_dir=output_dir)

    print(
f'''[✅  Startup Information]
Time:{datetime.datetime.now()}
output_dir:{output_dir}
dataset_dir:{dataset_dir}
Params:
{params}
''')

    logging.basicConfig(
        filename=os.path.join(output_dir, "output.log"), 
        filemode='w',
        level=logging.INFO,
        format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S')
    main(params, output_dir)
