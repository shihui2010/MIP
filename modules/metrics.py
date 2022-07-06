import tensorflow as tf
import numpy as np
from sklearn.metrics import ndcg_score


class _LabelBalance:
    def __init__(self):
        self._total = 0
        self._pos = 0
        self._finalize = False

    def update_states(self, y_true):
        if self._finalize:
            return
        self._total += tf.size(y_true)
        self._pos += tf.reduce_sum(tf.where(y_true != 0, 1, 0))

    def finalize(self):
        self._finalize = True

    def reset_states(self):
        self._total = 0
        self._pos = 0
        self._finalize = False

    def result(self):
        if self._total == 0:
            return 0
        return self._pos / self._total


def _safe_devide(hit, total):
    return hit / max(total, 1)


class _MetricAt:
    def __init__(self, k):
        self.precision = tf.keras.metrics.Precision(top_k=k)
        self.recall = tf.keras.metrics.Recall(top_k=k)
        self._ndcg_total = 0
        self._hit_total = 0
        self._count = 0
        self._k = k

    def update_states(self, pred, label):
        if pred.shape[1] < self._k:
            return
        self.precision.update_state(y_true=label, y_pred=pred)
        self.recall.update_state(y_true=label, y_pred=pred)
        score = ndcg_score(label.numpy().astype(float),
                           pred.numpy().astype(float), k=self._k)
        self._ndcg_total += score
        self._count += label.shape[0]
        top_idx = np.argsort(pred, axis=-1)
        top_labels = np.stack(
            [np.take(x, idx) for x, idx in zip(label, top_idx)],
            axis=0)[:, :self._k]
        self._hit_total += len(
            np.where(np.sum(top_labels, axis=-1))[0])

    @property
    def ndcg(self):
        return _safe_devide(self._ndcg_total, self._count)

    @property
    def hit_rate(self):
        return _safe_devide(self._hit_total, self._count)

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
        self._ndcg_total = 0
        self._hit_total = 0
        self._count = 0


class Metrics:
    def __init__(self):
        self._at5 = _MetricAt(5)
        self._at10 = _MetricAt(10)
        self._at20 = _MetricAt(20)
        self._at50 = _MetricAt(50)
        self._AUC = tf.keras.metrics.AUC()
        self._nll_total = 0
        self._nll_count = 0
        self._label_balance = _LabelBalance()
        self._history_best = 0
        self._history_best_epoch = -1

    def update_states(self, pos_logits, neg_logits, nll):
        assert pos_logits.shape[0] == neg_logits.shape[0], \
                "batch size different"
        assert len(pos_logits.shape) == 2, "prediction shape error"
        assert len(neg_logits.shape) == 2, "prediction shape error"
        pos_logits = pos_logits.detach().cpu().numpy()
        neg_logits = neg_logits.detach().cpu().numpy()
        pos_label = tf.ones_like(pos_logits, dtype=tf.int32)
        neg_label = tf.zeros_like(neg_logits, dtype=tf.int32)

        predictions = tf.concat([pos_logits, neg_logits], axis=-1)
        # print(pos_label.dtype, neg_label.dtype)
        truths = tf.concat([pos_label, neg_label], axis=-1)
        # print("truth shape", bin_truths.shape, "pred shape", bin_preds.shape)

        self._at5.update_states(predictions, truths)
        self._at10.update_states(predictions, truths)
        self._at20.update_states(predictions, truths)
        self._at50.update_states(predictions, truths)

        self._AUC.update_state(truths, predictions)
        self._label_balance.update_states(truths)
        if nll is None:
            return
        self._nll_total += nll * predictions.shape[0]
        self._nll_count += predictions.shape[0]

    def reset_states(self):
        self._AUC.reset_states()
        self._label_balance.reset_states()
        self._nll_total = 0
        self._nll_count = 0
        self._at5.reset_states()
        self._at10.reset_states()
        self._at20.reset_states()
        self._at50.reset_states()

    @property
    def AUC(self):
        return self._AUC.result().numpy()

    @property
    def NLL(self):
        return _safe_devide(self._nll_total, self._nll_count)

    @property
    def label_balance(self):
        return self._label_balance.result()

    @property
    def precision_at_5(self):
        return self._at5.precision.result().numpy()

    @property
    def precision_at_10(self):
        return self._at10.precision.result().numpy()

    @property
    def precision_at_20(self):
        return self._at20.precision.result().numpy()

    @property
    def precision_at_50(self):
        return self._at50.precision.result().numpy()

    @property
    def recall_at_5(self):
        return self._at5.recall.result().numpy()

    @property
    def recall_at_10(self):
        return self._at10.recall.result().numpy()

    @property
    def recall_at_20(self):
        return self._at20.recall.result().numpy()

    @property
    def recall_at_50(self):
        return self._at50.recall.result().numpy()

    @property
    def ndcg_5(self):
        return self._at5.ndcg

    @property
    def ndcg_10(self):
        return self._at10.ndcg

    @property
    def ndcg_20(self):
        return self._at20.ndcg

    @property
    def ndcg_50(self):
        return self._at50.ndcg

    @property
    def hit_rate_5(self):
        return self._at5.hit_rate

    @property
    def hit_rate_10(self):
        return self._at10.hit_rate

    @property
    def hit_rate_20(self):
        return self._at20.hit_rate

    @property
    def hit_rate_50(self):
        return self._at50.hit_rate

    def __repr__(self):
        res = f"[P/R/NDCG/HR] @5 " \
              f"{self.precision_at_5: .4f}/{self.recall_at_5: .4f}/" \
              f"{self.ndcg_5: .4f}/{self.hit_rate_5: .4f} | " \
              f"@10 {self.precision_at_10: .4f}/{self.recall_at_10: .4f}/" \
              f"{self.ndcg_10: .4f}/{self.hit_rate_10: .4f} | " \
              f"@20 {self.precision_at_20: .4f}/{self.recall_at_20: .4f}/" \
              f"{self.ndcg_20: .4f}/{self.hit_rate_20: .4f} | " \
              f"@50 {self.precision_at_50: .4f}/{self.recall_at_50: .4f}/" \
              f"{self.ndcg_50: .4f}/{self.hit_rate_50: .4f} | " \
              f"AUC {self.AUC: .4f} | NLL {self.NLL: .4f}"
        return res

    def write_record(self, smy_writer, tag, epoch):
        smy_writer.add_scalar(f"{tag}AUC", self.AUC, epoch)
        smy_writer.add_scalar(f"{tag}NLL", self.NLL, epoch)
        smy_writer.add_scalar(f"{tag}Precision@5", self.precision_at_5, epoch)
        smy_writer.add_scalar(f"{tag}Precision@10", self.precision_at_10, epoch)
        smy_writer.add_scalar(f"{tag}Precision@20", self.precision_at_20, epoch)
        smy_writer.add_scalar(f"{tag}Precision@50", self.precision_at_50, epoch)
        smy_writer.add_scalar(f"{tag}Recall@5", self.recall_at_5, epoch)
        smy_writer.add_scalar(f"{tag}Recall@10", self.recall_at_10, epoch)
        smy_writer.add_scalar(f"{tag}Recall@20", self.recall_at_20, epoch)
        smy_writer.add_scalar(f"{tag}Recall@50", self.recall_at_50, epoch)
        smy_writer.add_scalar(f"{tag}NDCG@5", self.ndcg_5, epoch)
        smy_writer.add_scalar(f"{tag}NDCG@10", self.ndcg_10, epoch)
        smy_writer.add_scalar(f"{tag}NDCG@20", self.ndcg_20, epoch)
        smy_writer.add_scalar(f"{tag}NDCG@50", self.ndcg_50, epoch)
        smy_writer.add_scalar(f"{tag}HitRate@5", self.hit_rate_5, epoch)
        smy_writer.add_scalar(f"{tag}HitRate@10", self.hit_rate_10, epoch)
        smy_writer.add_scalar(f"{tag}HitRate@20", self.hit_rate_20, epoch)
        smy_writer.add_scalar(f"{tag}HitRate@50", self.hit_rate_50, epoch)
        if self.AUC > self._history_best and tag == 'Val':
            self._history_best = self.AUC
            self._history_best_epoch = epoch

    def early_stop(self, epoch, tol=10):
        return epoch >= self._history_best_epoch + tol

