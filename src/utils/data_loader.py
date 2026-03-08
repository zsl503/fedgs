import pickle
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from src.utils.constants import *
from src.utils.data.datasets import CIFAR10, DATASETS

class BaseDataset(Dataset):
    def __init__(self) -> None:
        self.classes: list = None
        self.data: torch.Tensor = None
        self.targets: torch.Tensor = None
        self.train_data_transform = None
        self.train_target_transform = None
        self.test_data_transform = None
        self.test_target_transform = None
        self.data_transform = None
        self.target_transform = None

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return data, targets

    def train(self):
        self.data_transform = self.train_data_transform
        self.target_transform = self.train_target_transform

    def eval(self):
        self.data_transform = self.test_data_transform
        self.target_transform = self.test_target_transform

    def __len__(self):
        return len(self.targets)

class DataHandler:
    @staticmethod
    def load_data(dataset_name:str, file_dir, args = None, center_test = False) -> Subset | tuple[list[Subset], list[Subset], list[Subset]]:
        dataset_name = dataset_name.lower()
        # if dataset_name == "cifar10":
        #     train_client_datasets, val_client_datasets, test_client_datasets = \
        #         DataHandler.get_client_dataset(dataset_name, file_dir, args)
        #     if center_test:
        #         return DataHandler.get_global_test_dataset(dataset_name, file_dir, args)
        #         # indices = []
        #         # for c in test_client_datasets:
        #         #     # print("test",c.indices.keys())
        #         #     indices.extend(c.indices)
        #         # return Subset(dataset=test_client_datasets[0].dataset, indices=indices)
        #     else:
        #         return train_client_datasets, val_client_datasets, test_client_datasets
        if center_test:
            return DataHandler.get_global_test_dataset(dataset_name, file_dir, args)
        else:
            # train_client_datasets, val_client_datasets, test_client_datasets = \
            return DataHandler.get_client_dataset(dataset_name, file_dir, args)
        # 扩展其他数据集...

    @staticmethod
    def get_dataset_transforms(dataset_name):
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[dataset_name],
                    DATA_STD[dataset_name],
                )
            ]
            if dataset_name in DATA_MEAN
            and dataset_name in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[dataset_name],
                    DATA_STD[dataset_name],
                )
            ]
            if dataset_name in DATA_MEAN
            and dataset_name in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])
        return dict(
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )

    @staticmethod
    def get_client_dataset(dataset_name, file_dir, args = None) -> tuple[list[Subset], list[Subset], list[Subset]]:
        """Load FL dataset and partitioned data indices of clients.

        Raises:
            FileNotFoundError: When the target dataset has not beed processed.

        Returns:
            FL dataset.
        """
        try:
            with open(os.path.join(file_dir, "partition.pkl"), "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(
                f"Please partition {dataset_name} first."
            )

        # [0: {"train": [...], "val": [...], "test": [...]}, ...]
        data_indices: list[dict[str, list[int]]] = partition["data_indices"]

        dataset: BaseDataset = DATASETS[dataset_name](
            root=file_dir,
            args=args,
            **DataHandler.get_dataset_transforms(dataset_name),
        )

        # 创建客户端数据集
        train_client_datasets = []
        val_client_datasets = []
        test_client_datasets = []
        for indices in data_indices:
            train_client_datasets.append(Subset(dataset, indices["train"]))
            val_client_datasets.append(Subset(dataset, indices["val"]))
            test_client_datasets.append(Subset(dataset, indices["test"]))
        return train_client_datasets, val_client_datasets, test_client_datasets
    

    @staticmethod
    def get_global_test_dataset(dataset_name, file_dir, args = None) -> Subset:
        """Load FL dataset and partitioned data indices of clients.

        Raises:
            FileNotFoundError: When the target dataset has not beed processed.

        Returns:
            FL dataset.
        """
        


        # [0: {"train": [...], "val": [...], "test": [...]}, ...]

        if dataset_name in ["cifar10", "cifar100", "mnist"]:
            dataset: BaseDataset = DATASETS[dataset_name](
                root=file_dir,
                args=args,
                **DataHandler.get_dataset_transforms(dataset_name),
            )
            test_indices = list(range(0, len(dataset.test_targets)))
            return Subset(dataset, test_indices)
        
        if dataset_name in ["femnist"]:
            try:
                with open(os.path.join(file_dir, "partition.pkl"), "rb") as f:
                    partition = pickle.load(f)
            except:
                raise FileNotFoundError(
                    f"Please partition {dataset_name} first."
                )

            data_indices: list[dict[str, list[int]]] = partition["data_indices"]

            dataset: BaseDataset = DATASETS[dataset_name](
                root=file_dir,
                args=args,
                **DataHandler.get_dataset_transforms(dataset_name),
            )

            test_indices = []
            for indices in data_indices:
                test_indices.extend(indices["test"])
            return Subset(dataset, test_indices)