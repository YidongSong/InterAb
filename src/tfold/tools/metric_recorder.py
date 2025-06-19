"""Recorder for evaluation metrics w/ Horovod support."""

from collections import defaultdict

import horovod.torch as hvd

from tfold.utils import get_rand_str


class MetricRecorder():
    """Recorder for evaluation metrics w/ Horovod support."""

    def __init__(self, max_len=-1):
        """Constructor function."""

        # configurations
        self.max_len = max_len

        # initialize the buffer of evaluation metrics
        self.metrics_dict = {}
        self.entry_names = []  # used for indexing <self.metrics_dict>

        # check whether hovorod has been initialized
        self.use_hvd = True
        try:
            self.n_workers = hvd.size()
        except ValueError:
            self.use_hvd = False


    def reset(self):
        """Reset the recorder."""

        self.entry_names = []
        self.metrics_dict = {}


    def add(self, metrics, name=None):
        """Add evaluation metrics into the buffer."""

        # add the dict of evaluation metrics into the buffer
        if name is None:
            name = get_rand_str()
        self.entry_names.append(name)
        self.metrics_dict[name] = metrics

        # pop the earliest entry to maintain the buffer size
        if (self.max_len != -1) and (len(self.entry_names) > self.max_len):
            self.metrics_dict.pop(self.entry_names[0])
            self.entry_names = self.entry_names[1:]


    def get(self, rtn_raw=False):
        """Get the dict of averaged evaluation metrics."""

        # check whether the buffer is empty
        assert len(self.metrics_dict) != 0, 'buffer of evaluation metrics must be non-empty'

        # gather buffers of evaluation metrics from all the workers
        if not self.use_hvd:
            metrics_dict = self.metrics_dict
        else:
            metrics_dict = {}
            entry_names = set()  # record all the entry names for duplicate items removal
            metrics_dict_list = hvd.allgather_object(self.metrics_dict)
            for metrics_dict_sel in metrics_dict_list:
                for name, metrics in metrics_dict_sel.items():
                    if name not in entry_names:
                        entry_names.add(name)
                        metrics_dict[name] = metrics

        # calculate the averaged value for each evaluation metric
        metrics_cnt = defaultdict(int)
        metrics_sum = defaultdict(float)
        for metrics in metrics_dict.values():
            for key, val in metrics.items():
                metrics_cnt[key] += 1
                metrics_sum[key] += val
        metrics_avg = {k: metrics_sum[k] / metrics_cnt[k] for k in metrics_sum}

        # convert evaluation metrics into a string
        keys = sorted(list(metrics_avg.keys()))
        metrics_str = ', '.join([f'{key}={metrics_avg[key]:.4f}' for key in keys])

        # return the raw dict of evaluation metrics, indexed by entry names
        if rtn_raw:
            return metrics_avg, metrics_str, metrics_dict

        return metrics_avg, metrics_str
