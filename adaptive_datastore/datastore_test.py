import sys
import torch

from datasets import Dataset
from transformers import AutoModel, AutoTokenizer

from gpt2_updated import GPT2LMHeadModel

sys.path.append("..")
from fairseq.data.lm_context_window_dataset import LMContextWindowDataset
from fairseq_cli.eval_lm_adaptive import InMemoryDataStore


class kNN_LM:
    def __init__(self, model_name: str) -> None:
        # Load the model
        self.model, self.tokenizer = kNN_LM.load_model(model_name)

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
        # model = AutoModel.from_pretrained(url)
        model = GPT2LMHeadModel.from_pretrained(url)
        tokenizer = AutoTokenizer.from_pretrained(url)

        return model, tokenizer

    def train(self, dataset: Dataset, adaptive_mem: bool=False) -> None:
        if adaptive_mem:
            raise NotImplementedError

        # dataset should give both extra context (from the previous sequence and the actual sequence itself)
        for extra_context, sequence in dataset:
            # TODO: Convert to a batched implementation
            full_input = extra_context + sequence

            extra_context_input_ids = self.tokenizer(extra_context, return_tensors="pt")["input_ids"]
            full_input_tok = self.tokenizer(full_input, return_tensors="pt")

            output, last_layer_ffn_input = self.model(**full_input_tok)
            selected_last_layer_ffn_input = last_layer_ffn_input[:, extra_context_input_ids-1:, :]
            assert selected_last_layer_ffn_input.shape[1] == len(sequence), selected_last_layer_ffn_input.shape

            # Convert the whole sequence into set of target words
            sequence_tokenized = full_input_tok["input_ids"][torch.arange(), extra_context_input_ids.shape[1]:]
            for batch_idx in range(len(sequence_tokenized)):
                for i, tokenized_target_word in enumerate(sequence_tokenized[batch_idx]):
                    key = selected_last_layer_ffn_input[batch_idx, i, :]
                    self.dstore.add_item_to_store((key, tokenized_target_word))
        raise NotImplementedError


def main():
    # TODO: Load the dataset here
    dataset = None

    # Create the kNN with datastore and train
    knn_lm = kNN_LM(model_name="gpt2-small")
    knn_lm.train(dataset)


if __name__ == "__main__":
    main()
