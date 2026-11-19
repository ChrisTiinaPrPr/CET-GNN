"""GPU-resident prepared traffic, preserving the original CPU batch semantics."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import IndexDataset


class GPUPreparedTraffic:
    def __init__(self, data, device, chunk_rows=8192):
        self.device = torch.device(device)
        self.windows = tuple(data.windows)
        self.num_samples = data.num_samples
        self.max_history = data.src_history.shape[1]
        arrays = {name: getattr(data, name) for name in (
            'edge_features', 'src_profiles', 'dst_profiles', 'src_history', 'dst_history',
            'src_history_direction', 'dst_history_direction', 'labels')}
        self.estimated_bytes = sum(value.nbytes for value in arrays.values()) + 2 * self.num_samples * self.max_history * 4
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            if self.estimated_bytes + 3 * 1024**3 > free:
                raise MemoryError('GPU cache needs its data size plus 3 GiB for training; use the CPU backend or free GPU memory')
        for name, value in arrays.items():
            setattr(self, name, torch.from_numpy(np.ascontiguousarray(value)).to(self.device))
        self.src_lag = self._lag_table(data.src_history, chunk_rows)
        self.dst_lag = self._lag_table(data.dst_history, chunk_rows)

    def _lag_table(self, history, chunk_rows):
        table = torch.empty(history.shape, dtype=torch.float32, device=self.device)
        # NumPy log1p and float32 division exactly match the existing collator.
        # Computing log1p with CUDA instead would introduce avoidable rounding differences.
        for start in range(0, self.num_samples, chunk_rows):
            stop = min(start + chunk_rows, self.num_samples)
            selected = history[start:stop]
            mask = selected >= 0
            safe = selected.clip(min=0).astype(np.int64, copy=False)
            lag = np.arange(start, stop, dtype=np.int64)[:, None] - safe
            lag = np.log1p(np.maximum(lag, 0)).astype(np.float32) / 10.0
            lag[~mask] = 0.0
            table[start:stop].copy_(torch.from_numpy(lag))
        return table

    def collator(self, window, history_size):
        selected_windows = (window,) if isinstance(window, (int, np.integer)) else tuple(window)
        if (not selected_windows or len(set(selected_windows)) != len(selected_windows)
                or any(w not in self.windows for w in selected_windows)
                or not 0 < history_size <= self.max_history):
            raise ValueError('Window/history request exceeds resident data')
        if selected_windows == self.windows:
            profiles_src, profiles_dst = self.src_profiles, self.dst_profiles
        elif len(selected_windows) == 1:
            first = self.windows.index(selected_windows[0]) * 9
            profiles_src = self.src_profiles[:, first:first+9]
            profiles_dst = self.dst_profiles[:, first:first+9]
        else:
            columns = [i for w in selected_windows for i in
                       range(self.windows.index(w) * 9, self.windows.index(w) * 9 + 9)]
            profiles_src = self.src_profiles[:, columns]
            profiles_dst = self.dst_profiles[:, columns]

        def history_batch(indices, selected_table, direction_table, lag_table):
            selected = selected_table[indices, :history_size]
            direction = direction_table[indices, :history_size].long()
            safe = selected.clamp(min=0).long()
            neighbor = torch.where((direction == 0).unsqueeze(-1), profiles_dst[safe], profiles_src[safe])
            return {'edge': self.edge_features[safe], 'neighbour': neighbor,
                    'direction': direction, 'lag': lag_table[indices, :history_size], 'mask': selected >= 0}

        def collate(batch):
            indices = torch.as_tensor(np.asarray(batch, dtype=np.int64), device=self.device)
            return {'index': indices, 'edge': self.edge_features[indices],
                    'src_profile': profiles_src[indices], 'dst_profile': profiles_dst[indices],
                    'src_history': history_batch(indices, self.src_history, self.src_history_direction, self.src_lag),
                    'dst_history': history_batch(indices, self.dst_history, self.dst_history_direction, self.dst_lag),
                    'label': self.labels[indices]}

        return collate

    def loader(self, indices, batch_size, shuffle, window, history_size):
        # Keep the CPU RandomSampler/DataLoader so its permutation and RNG draws
        # remain identical to the original loader. Only gathering/storage changes.
        return DataLoader(IndexDataset(indices), batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, collate_fn=self.collator(window, history_size),
                          pin_memory=False, persistent_workers=False)
