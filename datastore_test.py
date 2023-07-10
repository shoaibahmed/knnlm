import torch

from dataset import Dataset
from transformers import AutoModel, AutoTokenizer


class InMemoryDataStore:
    """
    In memory variant of the datastore introduced in the kNN-LM paper (https://arxiv.org/abs/1911.00172).
    It stores key-value pairs, and returns the log-probability of the next token leveraging both
        the predicted next token probabilities from the LM (parametric memory) as well as the next
        token probability from the nearest neighbors storted in the datastore (non-parametric memory).
    """
    def __init__(self, dist_metric) -> None:
        self.k = None
        self.v = None
        assert dist_metric in ["l2", "squared_l2", "cosine"]
        self.dist_metric = dist_metric

    def add_item_to_store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert k.shape == v.shape, f"{k.shape} != {v.shape}"
        if len(k.shape) == 1:
            assert len(v.shape) == 1, v.shape
            k = k.expand_dim(dim=0)
            v = v.expand_dim(dim=0)

        if self.dist_metric == "cosine":
            k = torch.nn.functional.normalize(k, p=2.0, dim=1)  # l2 normalize the key
        if self.k is None:
            assert self.v is None
            self.k = k
            self.v = v
        else:
            self.k = torch.concatenate([self.k, k], dim=0)
            self.v = torch.concatenate([self.v, v], dim=0)
        print(f"!! New item added in memory store / k: {k.shape} / v: {v.shape}")

    def get_knn_log_probs(self, q: torch.Tensor, lm_log_probs: torch.Tensor,
                          num_nn: int=1024) -> torch.Tensor:
        if len(q.shape) == 1:
            q = q.expand_dim(dim=0)

        log_prob_vector = torch.zeros_like(lm_log_probs)
        if self.k is not None:  # zero probability for every token otherwise
            # self.k shape: B x D
            if self.dist_metric in ["l2", "squared_l2"]:
                dist = ((self.k - q) ** 2).sum(dim=1)
                if self.dist_metric == "l2":
                    dist = torch.sqrt(dist)
            elif self.dist_metric == "cosine":
                q = torch.nn.functional.normalize(q, p=2.0, dim=1)  # l2 normalize the query
                sim = torch.matmul(q, self.k.T)  # (1 x D) x (N x D).T -> 1 x N
                dist = 1.0 - sim  # cosine distance
            else:
                raise RuntimeError(f"Unknown distance metric: {self.dist_metric}")

            # Compute nearest neighbors based on the distance
            dist_idx_sorted = torch.argsort(dist, dim=1, descending=False)
            nearest_neighbors = dist_idx_sorted[:, :num_nn]

            # Select the nearest neighbors
            selected_dists = torch.gather(dist, dim=1, index=nearest_neighbors)
            selected_vals = torch.gather(self.v, dim=1, index=nearest_neighbors)

            # Compute the normalized log-probs
            unnormalized_log_prob = -selected_dists  # -dist since we are now in log-space
            normalized_log_prob = unnormalized_log_prob - torch.logsumexp(unnormalized_log_prob, dim=1, keepdim=True)

            # Compute aggregation of probabilities for the same word
            for idx in range(normalized_log_prob.shape[1]):
                log_prob_vector.scatter_add_(dim=1, index=selected_vals[:, idx:idx+1], src=normalized_log_prob[:, idx:idx+1])

        # return the final prob vector
        return log_prob_vector

    def get_combined_prob(self, lm_log_probs: torch.Tensor, lm_features: torch.Tensor,
                          lambda_val: float, num_nn: int=1024) -> torch.Tensor:
        # Get the log probs from the kNN
        knn_log_probs = self.get_knn_log_probs(lm_features, lm_log_probs, num_nn=num_nn)

        # LogSumExp is used since the two probabities are added, which necessitates exponentiation due to log_probs
        combined_log_probs = torch.logsumexp(torch.stack([lm_log_probs + torch.log(1. - lambda_val),
                                                          knn_log_probs + torch.log(lambda_val)]), dim=0)
        return combined_log_probs

    def print_datastore_stats(self) -> None:
        print(f"Datastore stats / Keys: {self.k.shape} / Values: {self.v.shape} / Distance metric: {self.dist_metric}")


class kNN_LM:
    def __init__(self, model_name: str, representation_layer_hook: str='last_ffn_layer') -> None:
        # Load the model
        self.model, self.tokenizer = kNN_LM.load_model(model_name)
        self.representation_layer_hook = representation_layer_hook

        # TODO: Attach the hook to the model

        # Create the datastore
        self.dstore = InMemoryDataStore(dist_metric="squared_l2")

    @staticmethod
    def load_model(model_name: str):
        if model_name == "gpt2-small":
            url = "gpt2"  # 124M params
        elif model_name == "gpt2-medium":
            url = "gpt2-medium"  # 355M params
        elif model_name == "gpt2-large":
            url = "gpt2-large"  # 774M params
        elif model_name == "gpt2-xl":
            url = "gpt2-xl"  # 1.5B params
        else:
            raise RuntimeError(f"Unknown model: {model_name}")

        # Load the model
        model = AutoModel.from_pretrained(url)
        tokenizer = AutoTokenizer.from_pretrained(url)

        return model, tokenizer

    def train(self, dataset: Dataset, adaptive_mem: bool=False) -> None:
        if adaptive_mem:
            raise NotImplementedError

        # dataset should give both extra context (from the previous sequence and the actual sequence itself)
        for extra_context, sequence in dataset:
            pass
        raise NotImplementedError


def main():
    # TODO: Load the dataset here
    pass

    # TODO: Create the kNN with datastore
    knn_lm = kNN_LM(model_name="gpt2-small")

    # TODO: Train the LM
    knn_lm.train()


if __name__ == "__main__":
    main()
