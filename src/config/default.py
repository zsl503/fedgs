from .params import BaseExperimentParams, DatasetArgs

class ExperimentParams(BaseExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__()
        # 基础参数
        self.num_clients = 50
        self.local_rounds = 1
        self.sim_time = 600000
        self.server_lr = 0.1  # 通常设为1.0直接使用客户端更新
        self.buffer_size = 10
        self.clients_per_round = None
        self.device = "cuda:0"
        self.train_method = "minibatch"  # minibatch, fullbatch
        self.select_method = "random_async"  # random_async, random_sync 
        self.stop_type = 'update'
        self.max_updates = 60000

        # 设备参数
        self.optimizer = "sgd"
        self.batch_size = 32
        self.learning_rate = 0.01
        self.momentum = 0

        self.speed_mode = "assign" # assign, multi, add
        
        self.speed_factors = None

        self.use_random_delay = False

        # 数据集参数
        self.dataset_name = "mnist"
        self.dataset_dir = dataset_dir
        self.dataset_args = DatasetArgs()

        # 模型参数
        self.model_name = "lenet5"
        
        # 算法选择
        self.algorithm = None      

        self.seed = 42
        self.video_format = "gif"
        self.validation_interval = 1

        self.use_sample_weight = False

class FedBuffParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)

class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.buffer_size = 1
        self.alpha = 0.9
        self.validation_interval = 10

class CA2FLParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.use_stale = False

class FedFAParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        # self.server_lr = 1
        self.validation_interval = 10

class FedProxParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.mu = 0.01

class MimeParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.beta = 0.5
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10

class FedAvgParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10

class FedDynParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.select_method = "random_sync"  # random_async, random_sync 
        self.clients_per_round = 10
        self.mu = 0.1

class FADASParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.server_lr = 1
        self.local_rounds = 1
        self.learning_rate = 0.001

class FedAsyncParams(ExperimentParams):
    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.buffer_size = 1
        self.alpha = 0.9
        self.validation_interval = 10
