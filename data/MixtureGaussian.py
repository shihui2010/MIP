import torch
from torch.utils.data import Dataset
from itertools import product
import numpy as np
import os

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


class Gaussian:
    global_id = 0

    def __init__(self, mean, cov, weight=0.5):
        self.mean = mean
        self.cov = cov
        self.dim = len(mean)
        self._local_id = self.global_id
        Gaussian.global_id += 1
        self.weight = weight

    def draw(self, size, norm=True):
        samples = np.random.multivariate_normal(
            self.mean, self.cov, size)
        if not norm:
            # for debugging
            return samples
        return samples / np.linalg.norm(samples, axis=1, keepdims=True)

    def __del__(self):
        Gaussian.global_id -= 1

    def __repr__(self):
        return f"<Gaussian Distrib. id=" \
            f"{self._local_id}, mean={self.mean}>"

    def sim(self, x):
        return np.sum(self.weight * x * self.mean / np.linalg.norm(x))


class UnweightedInterests:
    __thresh = None

    def __init__(self, positives, negatives, *args, **kwarg):
        self.positives = positives
        self.negatives = negatives
        self.dim = positives[0].dim
        self.n = len(positives)
        self._p = [1.0 / self.n] * self.n
        self.negative_samples = None
        self.input_seq = None
        self.positive_samples = None

    def _set_thresh(self, n_samples=1000):
        x, _ = self.sampling(n_samples)
        sims = np.array([max(c.sim(p) for c in self.positives) for p in x])
        UnweightedInterests.__thresh = np.percentile(sims, 10)
        print(f"Positive threshold set to {self.__thresh}")

    def sampling(self, seq_len, return_cid=False, pred=False):
        if self.positive_samples is not None and pred:
            x, timestamp, cids = self.positive_samples
        elif self.input_seq is not None and not pred:
            x, timestamp, cids = self.input_seq
        else:
            x = np.empty([seq_len, self.dim], dtype=np.float)
            cids = np.empty([seq_len], dtype=np.int)
            for i in range(seq_len):
                gid = np.random.choice(
                    np.arange(self.n), replace=False, p=self._p)
                x[i] = self.positives[gid].draw(1)
                cids[i] = gid
            timestamp = np.arange(seq_len)
            if pred:
                self.positive_samples = (x, timestamp, cids)
            else:
                self.input_seq = (x, timestamp, cids)
        if return_cid:
            return x, timestamp, cids
        return x, timestamp

    def negative_sampling(self, size):
        if self.negative_samples is not None:
            return self.negative_samples
        res = list()
        reject_count = 0
        while len(res) < size:
            x = np.random.random(self.dim)
            x /= max(1e-7, np.linalg.norm(x))
            x = np.expand_dims(x, axis=0)
            res.append(x)
        self.negative_samples = np.array(res).squeeze(axis=1)
        return self.negative_samples

class Universe:
    def __init__(self, n_components, nd, diffusion_factor):
        print("creating Universe with nd=", nd, "n_components=", n_components)
        if nd == 2:
            self.__init_2d(n_components, diffusion_factor)
        else:
            self.__init_nd(n_components, nd)
        self._p = [1.0 / n_components] * n_components
        self.n = n_components

    def __init_2d(self, n_components, diffusion_factor=1):
        delta = 2 * np.pi / n_components
        phi_0 = np.random.uniform(0, delta)
        std = np.sin(delta / n_components * diffusion_factor)
        angles = [phi_0 + i * delta for i in range(n_components)]
        x_y = [(np.cos(a), np.sin(a)) for a in angles]
        cov = np.zeros([2, 2])
        np.fill_diagonal(cov, np.ones(2) * (std ** 2))
        self.clusters = [Gaussian(mean, cov) for mean in x_y]

    def __init_nd(self, n_components, n_dim):
        split = int(np.ceil(np.power(n_components, 1.0 / n_dim)))
        if split < 2:
            split = 2
        delta_x = 2.0 / (split - 1)
        var = 0.5
        centers = np.arange(start=-1, stop=1.0001, step=delta_x)
        var_dim = min(16, n_dim)
        centers = list(product(centers, repeat=var_dim))
        selected = np.random.choice(range(len(centers)), size=n_components)
        centers = np.take(centers, selected, axis=0)

        invar_dims = n_dim - var_dim
        invar_centers = np.random.choice([-1, 1],
                                         size=[n_components, invar_dims])
        centers = np.concatenate([centers, invar_centers], axis=-1)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        cov = np.zeros([n_dim, n_dim])
        np.fill_diagonal(cov, np.ones(n_dim) * var)
        self.clusters = [Gaussian(mean, cov) for mean in centers]

    def draw(self, size, return_cid=False):
        res = np.empty([size, self.dim], dtype=np.float)
        cids = np.empty([size], dtype=np.int)
        for i in range(size):
            gid = np.random.choice(np.arange(self._n), replace=False, p=self._p)
            res[i] = self.clusters[gid].draw(1)
            cids[i] = gid
        if return_cid:
            return res, cids
        return res

    def sample_user(self, n_users, n_interests):
        user_clusters = np.random.multinomial(n_interests, self._p, n_users)
        users = list()
        for uid in range(n_users):
            pos_cids = np.where(user_clusters[uid])[0]
            neg_cids = np.where(user_clusters[uid] == 0)[0]
            pos_interests = np.take(self.clusters, pos_cids)
            neg_interests = np.take(self.clusters, neg_cids)
            users.append(UnweightedInterests(pos_interests, neg_interests))
        return users


class _SyntheticDataset(Dataset):
    def __init__(self, features):
        self._features = features

    def __len__(self):
        return len(self._features[0])

    def __getitem__(self, idx):
        features = (self._features[0][idx],  # input sequence
                    self._features[1][idx],  # input timestamp
                    self._features[3][idx],  # positive sequence
                    self._features[5][idx])  # negative sequence
        if torch.cuda.is_available():
            features = [x.to("cuda") for x in features]
        return features


class Batcher:
    def __init__(self, n_dim, n_components, interests_per_user,
                 train_size, test_size, val_size, seq_len):
        path = os.path.join(os.path.dirname(__file__),
                            f"synthetic/d{n_dim}c{n_components}")
        self._global = Universe(n_components, n_dim, diffusion_factor=2)
        self.seq_len = seq_len
        self.n_dim = n_dim

        if not os.path.exists(path + 'train'):
            train_users = self._global.sample_user(
                train_size, interests_per_user)
            self.train_dataset = self.dump_feature(f"{path}train", train_users)

            test_users = self._global.sample_user(
                test_size, interests_per_user)
            self.test_dataset = self.dump_feature(f"{path}test", test_users)
            val_users = self._global.sample_user(
                val_size, interests_per_user)
            self.val_dataset = self.dump_feature(f"{path}val", val_users)
        else:
            self.train_dataset = self.load_feature(f"{path}train")
            self.test_dataset = self.load_feature(f"{path}test")
            self.val_dataset = self.load_feature(f"{path}val")

    def dump_feature(self, path, users):
        all_input_seq = torch.empty([len(users), self.seq_len, self.n_dim])
        all_input_t = torch.empty([len(users), self.seq_len, 1])
        all_input_cid = torch.empty([len(users), self.seq_len])
        all_pos = torch.empty_like(all_input_seq)
        all_pos_cid = torch.empty_like(all_input_cid)
        all_neg = torch.empty_like(all_pos)
        for i, user in enumerate(users):
            input_seq, t, cids = user.sampling(self.seq_len, return_cid=True)
            pos_seq, _, pos_cid = user.sampling(self.seq_len, return_cid=True,
                                                pred=True)
            neg_seq = user.negative_sampling(self.seq_len)
            all_input_seq[i] = torch.from_numpy(input_seq)
            all_input_t[i] = torch.from_numpy(np.expand_dims(t, axis=-1))
            all_input_cid[i] = torch.from_numpy(cids)
            all_pos[i] = torch.from_numpy(pos_seq)
            all_pos_cid[i] = torch.from_numpy(pos_cid)
            all_neg[i] = torch.from_numpy(neg_seq)
        os.makedirs(path, exist_ok=True)
        print(f"data dumped to {path}")
        torch.save(all_input_seq, f"{path}/input_seq.pt")
        torch.save(all_input_t, f"{path}/input_t.pt")
        torch.save(all_input_cid, f"{path}/input_cid.pt")
        torch.save(all_pos, f"{path}/pos_seq.pt")
        torch.save(all_pos_cid, f"{path}/pos_cid.pt")
        torch.save(all_neg, f"{path}/neg_seq.pt")
        return _SyntheticDataset((all_input_seq, all_input_t, all_input_cid,
                                  all_pos, all_pos_cid, all_neg))

    def load_feature(self, path):
        all_input_seq = torch.load(f"{path}/input_seq.pt")
        all_input_t = torch.load(f"{path}/input_t.pt")
        all_input_cid = torch.load(f"{path}/input_cid.pt")
        all_pos = torch.load(f"{path}/pos_seq.pt")
        all_pos_cid = torch.load(f"{path}/pos_cid.pt")
        all_neg = torch.load(f"{path}/neg_seq.pt")
        return _SyntheticDataset((all_input_seq, all_input_t, all_input_cid,
                                  all_pos, all_pos_cid, all_neg))

