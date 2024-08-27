from typing import List, Optional

from sklearn.decomposition import TruncatedSVD
import numpy as np
import os
import torch

from cde.lib import DenseEncoder, print0, tqdm_if_main_worker
from cde.lib.embed import embed_with_cache
from cde.lib.misc import get_cache_location_from_kwargs



class ClusterPackingMixin:
    _encoder: Optional[DenseEncoder] = None
    def _compute_centroid_embeddings(self, clusters: List[List[int]]) -> torch.Tensor:
        model_name = "sentence-transformers/gtr-t5-base" # TODO: Argparse for this.
        if self._encoder is None:
            self._encoder = DenseEncoder(model_name, max_seq_length=128)

        all_idxs = [idx for cluster in clusters for idx in cluster]
        subdataset = self.dataset.dataset.select(all_idxs)
        embeddings_dataset = embed_with_cache(
            model_name, 
            subdataset._fingerprint, 
            subdataset,
            "document",
            model=self._encoder,
            save_to_disk=True,
            batch_size=8192,
        )
        centroids = []
        j = 0
        for cluster_idxs in clusters:
            embeddings = embeddings_dataset.select(range(j, j + len(cluster_idxs)))["embeds"]
            centroids.append(embeddings.mean(dim=0))
            j += len(cluster_idxs)
        return torch.stack(centroids)

    def _tsp_greedy_uncached(self, np_gen: np.random.Generator, clusters: List[List[int]]) -> List[int]:
        if len(clusters) == 1:
            print(f"Got only one cluster, returning it without ordering.")
            return [0]

        # Embed all the clusters
        cluster_embeddings = self._compute_centroid_embeddings(clusters)
        
        # Downscale
        scaler = TruncatedSVD(n_components=256, random_state=42)
        cluster_embeddings = scaler.fit_transform(cluster_embeddings.numpy())
        cluster_embeddings = torch.tensor(cluster_embeddings, dtype=torch.float)
        cluster_embeddings = cluster_embeddings.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Solve TSP
        cluster_mask = torch.zeros(len(clusters), dtype=torch.bool, device=cluster_embeddings.device)
        for _ in tqdm_if_main_worker(range(len(clusters) - 1), desc="Solving TSP greedily"):
            if _ == 0:
                idx = np_gen.integers(low=0, high=len(clusters))
                cluster_mask[idx] = 1
                order = [idx]
            distances = (cluster_embeddings[None, idx] - cluster_embeddings).norm(p=2, dim=1)
            distances = distances + cluster_mask * 10**9
            idx = distances.argmin().item()
            cluster_mask[idx] = 1
            order.append(idx)
        assert len(order) == len(clusters)
        assert set(order) == set(range(len(clusters)))
        return order
    
    def tsp_greedy(self, np_gen: np.random.Generator, clusters: List[List[int]], cluster_id: int) -> List[int]:
        cache_location = get_cache_location_from_kwargs(
            method="tsp_greedy",
            dataset_fingerprint=self.dataset._fingerprint, 
            cluster_size=self.cluster_size,
            batch_size=self.batch_size, 
            seed=self.seed,
            epoch=self.epoch,
            batch_packing_strategy=self.batch_packing_strategy,
            cluster_id=cluster_id,
        ) 
        if os.path.exists(cache_location):
            print0(f"[tsp_greedy] Loading TSP solution from {cache_location} (cluster_id={cluster_id})")
            order = list(map(int, open(cache_location).readlines()))
        else:
            print0(f"[tsp_greedy] Saving TSP order to {cache_location} (cluster_id={cluster_id})")
            order = self._tsp_greedy_uncached(np_gen, clusters)
            with open(cache_location, "w") as f:
                f.write("\n".join(map(str, order)))
            print0(f"[tsp_greedy] Saved TSP order to {cache_location} (cluster_id={cluster_id})")
        assert len(order) == len(clusters)
        return [clusters[j] for j in order]

    def _order_batches(self, np_gen: np.random.Generator, batches: List[List[int]], cluster_id: int) -> List[List[int]]:
        if self.batch_packing_strategy == "random":
            np_gen.shuffle(batches)
            return batches
        elif self.batch_packing_strategy == "tsp_greedy":
            return self.tsp_greedy(np_gen, batches, cluster_id=cluster_id)
        else:    
            raise ValueError(f"unknown batch_packing_strategy {self.batch_packing_strategy}")
