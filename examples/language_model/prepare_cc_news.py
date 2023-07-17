#!/bin/python

import os
import natsort
from datasets import load_dataset, DatasetDict


# Download the dataset
dataset_name = "cc_news"
dataset = load_dataset(dataset_name)

# Split the dataset into train, val and test
train_testvalid = dataset['train'].train_test_split(test_size=0.1)              # 90% train, 10% test + validation
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)            # Split the 10% test + valid in half test, half valid

train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'valid': test_valid['train'],
    'test': test_valid['test'],
})
print(train_test_valid_dataset)

# Save the different tokenized version
output_dir = "./cc_news_tokenized/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Output directory created:", output_dir)

for split in ["train", "valid", "test"]:
    with open(os.path.join(output_dir, f"{split}.tokens"), "w") as f:
        for ex_idx, example in enumerate(train_test_valid_dataset[split]):
            f.write(f"= {example['title']} =\n\n")
            f.write(f"{example['text']}\n\n\n")
