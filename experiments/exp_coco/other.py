'''def show_by_contexts(self, metric, rng_contexts, n_observed_shifts=None, n_confounder_shifts=None):
    return self._show_by(metric, rng_contexts, 0, None, n_observed_shifts, n_confounder_shifts)


def show_by_observed_shifts(self, metric, rng_observed_shifts, n_contexts, n_confounder_shifts=None):
    return self._show_by(metric, rng_observed_shifts, 1, n_contexts, None, n_confounder_shifts)


def show_by_confounder_shifts(self, metric, rng_confounder_shifts, n_contexts, n_observed_shifts=None):
    return self._show_by(metric, rng_confounder_shifts, 2, n_contexts, n_observed_shifts, None)


def _show_by(self, metric, rng, i_range, n_contexts=None, n_observed_shifts=None, n_confounder_shifts=None):
    frame = pd.DataFrame({'f1': [], 'acc_null': []})
    for r in rng:
        f1s, null_accs = [], []
        for key in self.result_pairs[metric]:
            k = np.int64(key.split('_'))
            if k[i_range] != r:
                continue
            if n_contexts is not None and k[0] != n_contexts:
                continue
            if n_observed_shifts is not None and (k[1] != n_observed_shifts):
                continue
            if n_confounder_shifts is not None and (k[2] != n_confounder_shifts):
                continue
            f1, null_acc = self.result_pairs[metric][key]['f1'], self.result_pairs[metric][key]['null_acc']
            f1s.append(f1)
            null_accs.append(null_acc)

        res = [None, None]
        if len(f1s) > 0:
            res = [mean(f1s), mean(null_accs)]
        frame.loc[len(frame)] = res
    return frame
'''