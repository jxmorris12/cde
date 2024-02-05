from typing import Dict, Iterable, Union

import torch

class TensorRunningAverages:
    _store_sum: Dict[str, torch.Tensor]
    _store_total: Dict[str, torch.Tensor]

    def __init__(self):
        self._store_sum = {}
        self._store_total = {}
    
    def __iter__(self) -> Iterable[str]:
        return iter(self._store_sum.keys())

    def update(self, key: str, val: Union[int, float, torch.Tensor]) -> None:
        if key not in self._store_sum:
            self.clear(key)
        if isinstance(val, torch.Tensor):
            val = val.item() # tensor -> num
        self._store_sum[key] += val
        self._store_total[key] += 1

    def get(self, key: str) -> float:
        total = max(self._store_total.get(key).item(), 1.0)
        return (self._store_sum[key] / float(total)).item() or 0.0
    
    def clear(self, key: str) -> None:
        self._store_sum[key] = torch.tensor(0.0, dtype=torch.float32)
        self._store_total[key] = torch.tensor(0, dtype=torch.int32)
    
    def clear_all(self) -> None:
        for key in self._store_sum:
            self.clear(key)

    def get_and_clear_all(self) -> Dict[str, float]:
        metrics = {}
        for key in self:
            metrics[key] = self.get(key)
            self.clear(key)
        return metrics