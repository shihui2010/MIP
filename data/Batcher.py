import numpy as np
import torch
from torch.utils.data import Dataset
import os
import json
import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process


def _generate_sequence(data, n_items, seq_len, neg_len, pos_len):
    """Create multiple sequences from the user engagement sequence
    :param data: Dictionary [user: user history]
    :param n_items: universal item count (as maximum item id)
    :param seq_len: int, input sequence length
    :param neg_len: int, negative samples
    :param pos_len: int, positive samples
    """
    worker_id = current_process()
    res = list()
    for user in tqdm(data, desc=f"{worker_id} processing"):
        sequence, timestamps = data[user]
        unseen = list(set(range(n_items)).difference(set(sequence)))
        replace = len(unseen) < neg_len
        for idx in np.arange(0, len(sequence) - seq_len - pos_len,
                             step=seq_len + pos_len):
            this_time = timestamps[idx: idx + seq_len].astype(float)
            this_seq = sequence[idx: idx + seq_len].astype(int)
            this_pos = sequence[idx + seq_len:
                                idx + seq_len + pos_len].astype(int)
            this_neg = np.random.choice(
                unseen, neg_len, replace=replace).astype(int)
            res.append([this_seq, this_time, this_pos, this_neg])
    return res


class _BatcherBase:
    dirname = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, test_size, val_size):
        self._items = set(self.item2id.values())

        print(f"============ Data Information ==========")
        pool = np.arange(len(self.data))
        np.random.seed(2020)
        np.random.shuffle(pool)
        if test_size + val_size >= len(pool):
            raise ValueError(f"Inadequate samples: {len(pool)}")
        self.test_pool = pool[:test_size]
        self.val_pool = pool[test_size: test_size + val_size]
        self.train_pool = pool[test_size + val_size:]
        print(f"Train size {len(self.train_pool)}, "
              f"Test size {len(self.test_pool)}, "
              f"Validation size {len(self.val_pool)}")
        self._save_processed()

    def _preprocess(self, data, seq_len, pos_len, neg_len):
        """"""
        # split sequence for balanced sampling
        self.data = list()
        count = dict()
        self.user2id = dict()
        self.item2id = dict()
        self.popularity = dict()

        # preprocess
        for user in data:
            seq, times = data[user]
            # print(len(seq))
            if len(seq) < seq_len + pos_len:
                continue
            self.user2id[user] = len(self.user2id)
            for item in seq:
                count[item] = count.setdefault(item, 0) + 1

        rare_items = {item for item in count if count[item] < 10}
        print("rare items", len(rare_items))

        # split user dict for multiprocessing
        data_splits = [dict() for _ in range(2 * cpu_count())]
        idx = 0
        for user in tqdm(self.user2id, desc='gather new user_sequence'):
            seq, times = data[user]
            s_new, t_new = list(), list()
            for s, t in zip(seq, times):
                if s not in rare_items:
                    s_new.append(s)
                    t_new.append(t)
            if len(s_new) < seq_len + pos_len:
                continue
            for item in s_new:
                self.item2id.setdefault(item, len(self.item2id))
                self.popularity.setdefault(self.item2id[item], 0)
                self.popularity[self.item2id[item]] += 1
                assert self.item2id[item] <= len(self.item2id), "check here"
            s_new = [self.item2id[i] for i in s_new]
            indices = np.argsort(t_new)
            timestamps = np.take(t_new, indices).astype(np.float64)
            # Normalize timestamps
            timestamps -= timestamps[0]
            timestamps /= 60 * 60 * 24  # second to day
            sequence = np.take(s_new, indices)
            data_splits[idx % len(data_splits)][user] = (sequence, timestamps)
            idx += 1

        with Pool(cpu_count() * 2) as pool:
            args = [(s, len(self.item2id), seq_len, neg_len, pos_len)
                    for s in data_splits]
            for res in tqdm(pool.starmap(_generate_sequence, args),
                            desc="multiprocessing for sequences"):
                self.data.extend(res)

        # for user in tqdm(self.user2id, desc='generate sequence'):
        #     sequence, timestamps = data[user]
        #     unseen = list(self._items.difference(set(sequence)))
        #     replace = len(unseen) < neg_len
        #     for idx in np.arange(0, len(sequence) - seq_len - pos_len,
        #                          step=seq_len + pos_len):
        #         this_time = timestamps[idx: idx + seq_len].astype(float)
        #         this_seq = sequence[idx: idx + seq_len].astype(int)
        #         this_pos = sequence[idx + seq_len:
        #                             idx + seq_len + pos_len].astype(int)
        #         this_neg = np.random.choice(
        #             unseen, neg_len, replace=replace).astype(int)
        #         self.data.append((this_seq, this_time, this_pos, this_neg))
        self.data = np.array(self.data, dtype=np.object)

    def _save_processed(self):
        for name, pool in [("train", self.train_pool),
                           ("test", self.test_pool), ("val", self.val_pool)]:
            print(f"dumping {name} to {self.dirname}/{name}")
            samples = np.take(self.data, pool, axis=0)
            torch.save(torch.from_numpy(samples[:, 0].astype(np.int)).int(),
                       f"{self.datapath}/{name}_seq.pt")
            torch.save(torch.from_numpy(samples[:, 1].astype(np.float)).float(),
                       f"{self.datapath}/{name}_tem.pt")
            torch.save(torch.from_numpy(samples[:, 2].astype(np.int)).int(),
                       f"{self.datapath}/{name}_pos.pt")
            torch.save(torch.from_numpy(samples[:, 3].astype(np.int)).int(),
                       f"{self.datapath}/{name}_neg.pt")
        with open(f"{self.datapath}/item2id.json", "w") as fp:
            json.dump(self.item2id, fp)
        with open(f"{self.datapath}/user2id.json", "w") as fp:
            json.dump(self.user2id, fp)
        with open(f"{self.datapath}/popularity.json", "w") as fp:
            json.dump(self.popularity, fp)

    def take_sample(self, batch_size, pool):
        for idx in np.arange(len(pool), step=batch_size):
            batch = np.take(self.data, pool[idx: idx + batch_size], axis=0)
            seq = np.stack(batch[:, 0]).astype(int)
            tem = np.stack(batch[:, 1]).astype(float)
            pos = np.stack(batch[:, 2]).astype(int)
            neg = np.stack(batch[:, 3]).astype(int)
            yield seq, tem, pos, neg

    @property
    def item_count(self):
        return len(self.item2id)


class _TaobaoBatcher(_BatcherBase):
    def __init__(self, seq_len, pos_len, neg_len, test_size, val_size):
        self.item2cat = dict()
        data = self._load()
        self._preprocess(data, seq_len, pos_len, neg_len)
        self.datapath = os.path.join(self.dirname, "Taobao")
        super(_TaobaoBatcher, self).__init__(test_size, val_size)

    def _load(self):
        """File format: user_id, item_id, category_id, action, timestamp
                action:
                    pv--page view, click
                    buy
                    cart--add to cart
                    fav--add to favorite"""
        fp = open(os.path.join(self.dirname, "Taobao/UserBehavior.csv"))
        data = dict()
        this_seq = []
        this_times = []
        last_uid = None
        for line in fp:
            uid, item, cat, act, timestamp = line.strip().split(",")
            if act != "pv":
                continue
            if last_uid is not None and uid != last_uid:
                data[last_uid] = [this_seq, this_times]
                data.setdefault(uid, [[], []])
                this_seq, this_times = data[uid]
            this_seq.append(item)
            this_times.append(int(timestamp))
            last_uid = uid
        data[uid] = [this_seq, this_times]
        fp.close()
        return data


class _AmazonBatcher(_BatcherBase):
    def __init__(self, dataset, seq_len, pos_len, neg_len, test_size, val_size):
        data = self._load(dataset)
        self._preprocess(data, seq_len, pos_len, neg_len)
        self.datapath = os.path.join(self.dirname, "Amazon")
        super(_AmazonBatcher, self).__init__(test_size, val_size)

    def _load(self, dataset):
        """File format: user_id, item_id, category_id, rate, timestamp"""
        fp = open(os.path.join(self.dirname, "Amazon", f"{dataset}.csv"))
        last_uid, item, _, timestamp = fp.readline().strip().split(",")
        data = dict()
        this_seq = [0]
        this_times = [int(timestamp)]
        for line in fp:
            uid, item, _, timestamp = line.strip().split(",")
            if uid != last_uid:
                data[last_uid] = [this_seq, this_times]
                data.setdefault(uid, [[], []])
                this_seq, this_times = data[uid]
            this_seq.append(item)
            this_times.append(int(timestamp))
            last_uid = uid
        data[uid] = [this_seq, this_times]
        fp.close()
        return data


class _MovieLensBatcher(_BatcherBase):
    def __init__(self, seq_len, pos_len, neg_len, test_size, val_size):
        data = self._load()
        self._preprocess(data, seq_len, pos_len, neg_len)
        self.datapath = os.path.join(self.dirname, "MovieLens")
        super(_MovieLensBatcher, self).__init__(test_size, val_size)

    def __parse_time(self, date_string):
        time_obj = datetime.datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
        return time_obj.timestamp()

    def _load(self):
        """File format: user_id, item_id, category_id, rate, timestamp"""
        fp = open(os.path.join(self.dirname, "MovieLens", f"rating.csv"))
        fp.readline()
        last_uid, item, _, timestamp = fp.readline().strip().split(",")
        data = dict()
        this_seq = [0]
        this_times = [self.__parse_time(timestamp)]
        for line in fp:
            uid, item, _, timestamp = line.strip().split(",")
            if uid != last_uid:
                data[last_uid] = [this_seq, this_times]
                data.setdefault(uid, [[], []])
                this_seq, this_times = data[uid]
            this_seq.append(item)
            this_times.append(self.__parse_time(timestamp))
            last_uid = uid
        data[uid] = [this_seq, this_times]
        fp.close()
        return data


class CFDataset(Dataset):
    def __init__(self, name, split, cuda=True):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if name == "taobao":
            self.dir = os.path.join(base_dir, "Taobao")
        elif name == "amazon":
            self.dir = os.path.join(base_dir, "Amazon")
        elif name == "movie":
            self.dir = os.path.join(base_dir, "MovieLens")
        else:
            raise ValueError(f"Unknown dataset {name}")
        assert split in ['train', 'test', 'val'], "Unknown split"

        if torch.cuda.is_available() and cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.input_seq = torch.load(f"{self.dir}/{split}_seq.pt").to(device)
        self.temp = torch.load(f"{self.dir}/{split}_tem.pt").to(device)
        self.pos = torch.load(f"{self.dir}/{split}_pos.pt").to(device)
        self.neg = torch.load(f"{self.dir}/{split}_neg.pt").to(device)
        with open(f"{self.dir}/item2id.json") as fp:
            self.item2id = json.load(fp)
        with open(f"{self.dir}/popularity.json") as fp:
            self.popularity = json.load(fp)

    @property
    def item_count(self):
        return len(self.item2id)

    def __len__(self):
        return self.input_seq.shape[0]

    def __getitem__(self, i):
        return self.input_seq[i], self.temp[i].unsqueeze(dim=-1), self.pos[i], self.neg[i]


if __name__ == "__main__":
    # preprocess and dump features
    _AmazonBatcher('book', 50, 50, 50, 5000, 5000)
    _TaobaoBatcher(50, 50, 50, 10000, 10000)
    _MovieLensBatcher(50, 50, 50, 5000, 5000)
