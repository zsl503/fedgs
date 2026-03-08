import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import requests
import zipfile
import io
import csv
import re
from .datasets import FEMNIST, CelebA, Synthetic, Sent140

DATA_ROOT = Path(__file__).parent.parent.absolute()


def prune_args(args: Namespace) -> dict:
    args_dict = {}
    # general settings
    args_dict["client_num"] = args.client_num
    args_dict["test_ratio"] = args.test_ratio
    args_dict["val_ratio"] = args.val_ratio
    args_dict["seed"] = args.seed
    args_dict["split"] = args.split
    args_dict["monitor_window_name_suffix"] = f"{args.dataset}-{args.client_num}clients"

    if args.dataset == "emnist":
        args_dict["emnist_split"] = args.emnist_split
    elif args.dataset == "cifar100":
        args_dict["super_class"] = bool(args.super_class)
        args_dict["monitor_window_name_suffix"] += "-use20superclasses"
    elif args.dataset == "synthetic":
        args_dict["beta"] = args.beta
        args_dict["gamma"] = args.gamma
        args_dict["dimension"] = args.dimension
        args_dict["class_num"] = 10 if args.classes <= 0 else args.classes
        args_dict["monitor_window_name_suffix"] += f"-beta{args.beta}-gamma{args.gamma}"
    elif args.dataset == "domain":
        with open(DATA_ROOT / "domain" / "metadata.json", "r") as f:
            metadata = json.load(f)
            args_dict["data_amount"] = metadata["data_amount"]
            args_dict["image_size"] = metadata["image_size"]
            args_dict["class_num"] = metadata["class_num"]
            args_dict["preprocess_seed"] = metadata["seed"]
            args_dict["monitor_window_name_suffix"] += f"-class{metadata['class_num']}"
    elif args.dataset in ["femnist", "celeba"]:
        with open(DATA_ROOT / args.dataset / "preprocess_args.json") as f:
            preprocess_args = json.load(f)
        args_dict.pop("seed")
        args_dict["split"] = preprocess_args["t"]
        args_dict["fraction"] = preprocess_args["tf"]
        args_dict["sample_seed"] = preprocess_args["smplseed"]
        args_dict["split_seed"] = preprocess_args["spltseed"]
        args_dict["least_samples"] = preprocess_args["k"]
        args_dict["monitor_window_name_suffix"] += f"-fraction{args_dict['fraction']}"
        if preprocess_args["s"] == "iid":
            args_dict["iid"] = True
            args_dict["monitor_window_name_suffix"] += f"-IID"
    if args.iid == 1:
        args_dict["iid"] = True
        args_dict["monitor_window_name_suffix"] += f"-IID"
    if args.ood_domains is not None:
        args_dict["ood_domains"] = args.ood_domains
        args_dict["monitor_window_name_suffix"] += f"-{args.ood_domains}OODdomains"
    else:
        # Dirchlet
        if args.alpha > 0:
            args_dict["alpha"] = args.alpha
            args_dict["least_samples"] = args.least_samples
            args_dict["monitor_window_name_suffix"] += f"-Dir({args.alpha})"
        # randomly assign classes
        elif args.classes > 0:
            args_dict["classes_per_client"] = args.classes
            args_dict["monitor_window_name_suffix"] += f"-{args.classes}classes"
        # allocate shards
        elif args.shards > 0:
            args_dict["shards_per_client"] = args.shards
            args_dict["monitor_window_name_suffix"] += f"-{args.shards}shards"
        elif args.semantic:
            args_dict["pca_components"] = args.pca_components
            args_dict["efficient_net_type"] = args.efficient_net_type
            args_dict["monitor_window_name_suffix"] += f"-semantic"
    args_dict["monitor_window_name_suffix"] += f"-seed{args.seed}"
    return args_dict

def _simple_tokenize(text):
    # very light-weight tokenizer: lowercase, remove URLs/mentions, split on whitespace and punctuation
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r"www\.[^\s]+", "", text)
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # remove mentions
    text = re.sub(r"[^0-9a-z\s']", " ", text)  # keep letters, numbers, apostrophe
    tokens = [t for t in text.split() if t]
    return tokens

def process_sent140(args, partition, stats):
    """
    Download + preprocess Sentiment140 and produce:
      - data.npy (N, max_len) int64
      - targets.npy (N,) int64
      - vocab.json (token->idx mapping)

    Uses args:
      args.sent140_max_len (default 50)
      args.vocab_size (default 20000)
    """
    max_len = getattr(args, "sent140_max_len", 50)
    vocab_size = getattr(args, "vocab_size", 20000)

    sent_root = DATA_ROOT / "sent140"
    sent_root.mkdir(parents=True, exist_ok=True)

    try:
        from src.utils.data.datasets import Sent140
        _ = Sent140(
            root=sent_root,
            args=args,
            test_data_transform=None,
            test_target_transform=None,
            train_data_transform=None,
            train_target_transform=None,
        )
        print(f"Sent140 dataset class found and data loaded from {sent_root}")
        return _
    
    except Exception:
        print("Sent140 dataset class not found, proceeding to download and preprocess data...")

    zip_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    training_csv_name = "training.1600000.processed.noemoticon.csv"
    csv_path = sent_root / training_csv_name

    # 下载并解压（如果 CSV 已存在则跳过）
    if not csv_path.exists():
        print("Downloading Sentiment140 dataset (trainingandtestdata.zip)...")
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # 提取 training CSV
        found = False
        for name in z.namelist():
            if name.endswith(training_csv_name):
                z.extract(name, path=sent_root)
                extracted = sent_root / name
                # 如果解压出带文件夹，移动到 sent_root 根目录
                if extracted.parent != sent_root:
                    extracted.rename(csv_path)
                    try:
                        extracted.parent.rmdir()
                    except Exception:
                        pass
                found = True
                break
        if not found:
            raise RuntimeError("Sent140 zip downloaded but training CSV not found inside.")

    # 解析 CSV: 格式 polarity,id,date,query,user,text
    texts, labels = [], []
    with open(csv_path, "r", encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                polarity = int(row[0])
            except Exception:
                continue
            text = row[5]
            label = 1 if polarity == 4 else 0  # 4 -> positive -> 1, 0 -> negative -> 0
            texts.append(text)
            labels.append(label)

    if len(texts) == 0:
        raise RuntimeError("No sentences found when processing Sent140.")

    # 建词汇表（global），保留 top (vocab_size-2)，0 PAD, 1 UNK
    token_counter = Counter()
    for t in texts:
        token_counter.update(_simple_tokenize(t))

    def text_to_seq(text, vocab):
        tokens = _simple_tokenize(text)
        seq = [vocab.get(t, 1) for t in tokens]  # 未知用 1
        if len(seq) >= max_len:
            return seq[:max_len]
        else:
            return seq + [0] * (max_len - len(seq))
        
    if not os.path.exists(sent_root / "vocab.json"):
        most_common = token_counter.most_common(max(0, vocab_size - 2))
        vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        vocab["<PAD>"] = 0
        vocab["<UNK>"] = 1

        with open(sent_root / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    else:
        with open(sent_root / "vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)

    all_seqs = np.stack([np.array(text_to_seq(t, vocab), dtype=np.int64) for t in texts], axis=0)
    all_labels = np.array(labels, dtype=np.int64)
    # 划分 train/val/test
    n = len(all_seqs)
    test_size = int(n * getattr(args, "test_ratio", 0.1))
    val_size = int(n * getattr(args, "val_ratio", 0.1))
    print(f"Total samples: {n}, test: {test_size}, val: {val_size}, train: {n - test_size - val_size}")
    train_size = n - test_size - val_size

    indices = np.arange(n)
    np.random.shuffle(indices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size: train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_data, train_targets = all_seqs[train_idx], all_labels[train_idx]
    val_data, val_targets = all_seqs[val_idx], all_labels[val_idx]
    test_data, test_targets = all_seqs[test_idx], all_labels[test_idx]
    # 保存文件
    np.save(sent_root / "train", train_data)
    np.save(sent_root / "train_targets", train_targets)
    np.save(sent_root / "val", val_data)
    np.save(sent_root / "val_targets", val_targets)
    np.save(sent_root / "test", test_data)
    np.save(sent_root / "test_targets", test_targets)

    print(f"Saved processed Sent140 to {sent_root}")
    from src.utils.data.datasets import Sent140
    return Sent140(
        root=sent_root,
        args=args,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    )


def process_femnist(args, partition: dict, stats: dict):
    train_dir = DATA_ROOT / "femnist" / "data" / "train"
    test_dir = DATA_ROOT / "femnist" / "data" / "test"
    client_cnt = 0
    data_cnt = 0
    all_data = []
    all_targets = []
    data_indices = {}
    clients_4_train, clients_4_test = None, None
    with open(DATA_ROOT / "femnist" / "preprocess_args.json", "r") as f:
        preprocess_args = json.load(f)

    # load data of train clients
    if preprocess_args["t"] == "sample":
        train_filename_list = sorted(os.listdir(train_dir))
        test_filename_list = sorted(os.listdir(test_dir))
        for train_js_file, test_js_file in zip(train_filename_list, test_filename_list):
            with open(train_dir / train_js_file, "r") as f:
                train = json.load(f)
            with open(test_dir / test_js_file, "r") as f:
                test = json.load(f)
            for writer in train["users"]:
                stats[client_cnt] = {}
                train_data = train["user_data"][writer]["x"]
                train_targets = train["user_data"][writer]["y"]
                test_data = test["user_data"][writer]["x"]
                test_targets = test["user_data"][writer]["y"]

                data = train_data + test_data
                targets = train_targets + test_targets
                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                data_indices[client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(train_data))),
                    "val": [],
                    "test": list(
                        range(data_cnt + len(train_data), data_cnt + len(data))
                    ),
                }
                stats[client_cnt]["x"] = len(data)
                stats[client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_test = list(range(client_cnt))
        clients_4_train = list(range(client_cnt))

        num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }
    else:
        stats["train"] = {}
        stats["test"] = {}
        for js_filename in os.listdir(train_dir):
            with open(train_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats["train"][client_cnt] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]

                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                data_indices[client_cnt] = {
                    "train": list(range(data_cnt, data_cnt + len(data))),
                    "val": [],
                    "test": [],
                }
                stats["train"][client_cnt]["x"] = len(data)
                stats["train"][client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_train = list(range(client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["train"].values()))
        )
        stats["train"]["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

        # load data of test clients
        for js_filename in os.listdir(test_dir):
            with open(test_dir / js_filename, "r") as f:
                json_data = json.load(f)
            for writer in json_data["users"]:
                stats["test"][client_cnt] = {"x": None, "y": None}
                data = json_data["user_data"][writer]["x"]
                targets = json_data["user_data"][writer]["y"]
                all_data.append(np.array(data))
                all_targets.append(np.array(targets))
                data_indices[client_cnt] = {
                    "train": [],
                    "val": [],
                    "test": list(range(data_cnt, data_cnt + len(data))),
                }
                stats["test"][client_cnt]["x"] = len(data)
                stats["test"][client_cnt]["y"] = Counter(targets)

                data_cnt += len(data)
                client_cnt += 1

        clients_4_test = list(range(len(clients_4_train), client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["test"].values()))
        )
        stats["test"]["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    np.save(DATA_ROOT / "femnist" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "femnist" / "targets", np.concatenate(all_targets))

    partition["separation"] = {
        "train": clients_4_train,
        "val": [],
        "test": clients_4_test,
        "total": client_cnt,
    }
    partition["data_indices"] = [indices for indices in data_indices.values()]
    args.client_num = client_cnt
    return FEMNIST(
        root=DATA_ROOT / "femnist",
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    )


def process_celeba(args, partition: dict, stats: dict):
    train_dir = DATA_ROOT / "celeba" / "data" / "train"
    test_dir = DATA_ROOT / "celeba" / "data" / "test"
    raw_data_dir = DATA_ROOT / "celeba" / "data" / "raw" / "img_align_celeba"
    train_filename = os.listdir(train_dir)[0]
    test_filename = os.listdir(test_dir)[0]
    with open(train_dir / train_filename, "r") as f:
        train = json.load(f)
    with open(test_dir / test_filename, "r") as f:
        test = json.load(f)

    data_cnt = 0
    all_data = []
    all_targets = []
    data_indices = {}
    client_cnt = 0
    clients_4_test, clients_4_train = None, None

    with open(DATA_ROOT / "celeba" / "preprocess_args.json") as f:
        preprocess_args = json.load(f)

    if preprocess_args["t"] == "sample":
        for client_cnt, ori_id in enumerate(train["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            train_data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in train["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            test_data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in test["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            train_targets = train["user_data"][ori_id]["y"]
            test_targets = test["user_data"][ori_id]["y"]

            data = np.concatenate([train_data, test_data])
            targets = train_targets + test_targets
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            data_indices[client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(train_data))),
                "val": [],
                "test": list(range(data_cnt + len(train_data), data_cnt + len(data))),
            }
            stats[client_cnt]["x"] = (
                train["num_samples"][client_cnt] + test["num_samples"][client_cnt]
            )
            stats[client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_train = list(range(client_cnt))
        clients_4_test = list(range(client_cnt))
        num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

    else:  # t == "user"
        # process data of train clients
        stats["train"] = {}
        for client_cnt, ori_id in enumerate(train["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in train["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            targets = train["user_data"][ori_id]["y"]
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            data_indices[client_cnt] = {
                "train": list(range(data_cnt, data_cnt + len(data))),
                "val": [],
                "test": [],
            }
            stats[client_cnt]["x"] = train["num_samples"][client_cnt]
            stats[client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_train = list(range(client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["train"].values()))
        )
        stats["sample per client"] = {
            "std": num_samples.mean(),
            "stddev": num_samples.std(),
        }

        # process data of test clients
        stats["test"] = {}
        for client_cnt, ori_id in enumerate(test["users"]):
            stats[client_cnt] = {"x": None, "y": None}
            data = np.stack(
                [
                    np.asarray(Image.open(raw_data_dir / img_name))
                    for img_name in test["user_data"][ori_id]["x"]
                ],
                axis=0,
            )
            targets = test["user_data"][ori_id]["y"]
            if all_data == []:
                all_data = data
            else:
                all_data = np.concatenate([all_data, data])
            if all_targets == []:
                all_targets = targets
            else:
                all_targets = np.concatenate([all_targets, targets])
            partition["data_indices"][client_cnt] = {
                "train": [],
                "val": [],
                "test": list(range(data_cnt, data_cnt + len(data))),
            }
            stats["test"][client_cnt]["x"] = test["num_samples"][client_cnt]
            stats["test"][client_cnt]["y"] = Counter(targets)

            data_cnt += len(data)
            client_cnt += 1

        clients_4_test = list(range(len(clients_4_train), client_cnt))

        num_samples = np.array(
            list(map(lambda stat_i: stat_i["x"], stats["test"].values()))
        )
        stats["sample per client"] = {
            "std": num_samples.mean().item(),
            "stddev": num_samples.std().item(),
        }

    np.save(DATA_ROOT / "celeba" / "data", all_data)
    np.save(DATA_ROOT / "celeba" / "targets", all_targets)

    partition["separation"] = {
        "train": clients_4_train,
        "val": [],
        "test": clients_4_test,
        "total": client_cnt,
    }
    partition["data_indices"] = [indices for indices in data_indices.values()]
    args.client_num = client_cnt
    return CelebA(
        root=DATA_ROOT / "celeba",
        args=None,
        test_data_transform=None,
        test_target_transform=None,
        train_data_transform=None,
        train_target_transform=None,
    )


def generate_synthetic_data(args, partition: dict, stats: dict):
    def softmax(x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex / sum_ex

    # All codes below are modified from https://github.com/litian96/FedProx/tree/master/data
    class_num = 10 if args.classes <= 0 else args.classes

    samples_per_user = (
        np.random.lognormal(4, 2, args.client_num).astype(int) + 50
    ).tolist()
    # samples_per_user = [10 for _ in range(args.client_num)]
    w_global = np.zeros((args.dimension, class_num))
    b_global = np.zeros(class_num)

    mean_w = np.random.normal(0, args.gamma, args.client_num)
    mean_b = mean_w
    B = np.random.normal(0, args.beta, args.client_num)
    mean_x = np.zeros((args.client_num, args.dimension))

    diagonal = np.zeros(args.dimension)
    for j in range(args.dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for client_id in range(args.client_num):
        if args.iid:
            mean_x[client_id] = np.ones(args.dimension) * B[client_id]  # all zeros
        else:
            mean_x[client_id] = np.random.normal(B[client_id], 1, args.dimension)

    if args.iid:
        w_global = np.random.normal(0, 1, (args.dimension, class_num))
        b_global = np.random.normal(0, 1, class_num)

    all_data = []
    all_targets = []
    data_cnt = 0

    for client_id in range(args.client_num):
        w = np.random.normal(mean_w[client_id], 1, (args.dimension, class_num))
        b = np.random.normal(mean_b[client_id], 1, class_num)

        if args.iid != 0:
            w = w_global
            b = b_global

        data = np.random.multivariate_normal(
            mean_x[client_id], cov_x, samples_per_user[client_id]
        )
        targets = np.zeros(samples_per_user[client_id], dtype=np.int32)

        for j in range(samples_per_user[client_id]):
            true_logit = np.dot(data[j], w) + b
            targets[j] = np.argmax(softmax(true_logit))

        all_data.append(data)
        all_targets.append(targets)

        partition["data_indices"][client_id] = list(
            range(data_cnt, data_cnt + len(data))
        )

        data_cnt += len(data)

        stats[client_id] = {}
        stats[client_id]["x"] = samples_per_user[client_id]
        stats[client_id]["y"] = Counter(targets.tolist())

    np.save(DATA_ROOT / "synthetic" / "data", np.concatenate(all_data))
    np.save(DATA_ROOT / "synthetic" / "targets", np.concatenate(all_targets))

    num_samples = np.array(list(map(lambda stat_i: stat_i["x"], stats.values())))
    stats["sample per client"] = {
        "std": num_samples.mean().item(),
        "stddev": num_samples.std().item(),
    }
    return Synthetic(root=DATA_ROOT / "synthetic")


def exclude_domain(
    client_num: int,
    targets: np.ndarray,
    domain_map: dict[str, int],
    domain_indices_bound: dict,
    ood_domains: set[str],
    partition: dict,
    stats: dict,
):
    ood_domain_num = 0
    data_indices = np.arange(len(targets), dtype=np.int64)
    for domain in ood_domains:
        if domain not in domain_map:
            Warning(f"One of `args.ood_domains` {domain} is unrecongnized and ignored.")
        else:
            ood_domain_num += 1

    def _idx_2_domain_label(index):
        # get the domain label of a specific data index
        for domain, bound in domain_indices_bound.items():
            if bound["begin"] <= index < bound["end"]:
                return domain_map[domain]

    domain_targets = np.vectorize(_idx_2_domain_label)(data_indices)

    id_label_set = set(
        label for domain, label in domain_map.items() if domain not in ood_domains
    )

    train_clients = list(range(client_num - ood_domain_num))
    ood_clients = list(range(client_num - ood_domain_num, client_num))
    partition["separation"] = {
        "train": train_clients,
        "val": [],
        "test": ood_clients,
        "total": client_num,
    }
    for ood_domain, client_id in zip(ood_domains, ood_clients):
        indices = np.where(domain_targets == domain_map[ood_domain])[0]
        partition["data_indices"][client_id] = {
            "train": np.array([], dtype=np.int64),
            "val": np.array([], dtype=np.int64),
            "test": indices,
        }
        stats[client_id] = {
            "x": len(indices),
            "y": {domain_map[ood_domain]: len(indices)},
        }

    return id_label_set, domain_targets, len(train_clients)


def plot_distribution(client_num: int, label_counts: np.ndarray, save_path: str):
    plt.figure()
    ax = plt.gca()
    left = np.zeros(client_num)
    client_ids = np.arange(client_num)
    for y, cnts in enumerate(label_counts):
        ax.barh(client_ids, width=cnts, label=str(y), left=left)
        left += cnts
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(bbox_to_anchor=(1.2, 1))
    plt.savefig(save_path, bbox_inches="tight")
