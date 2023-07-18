#!/bin/python

import os
import random
import natsort

import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict


def filter_without_date(example):
    return example['date'] != ''


def add_day(example):
    example['day'] = example['date'].split()[0]  # assuming date format is 'yyyy-mm-dd hh:mm:ss'
    return example


# Download the dataset
dataset_name = "cc_news"
dataset = load_dataset(dataset_name)["train"]  # there is no other split

# Remove all elements without any date
dataset = dataset.filter(filter_without_date)

# Add 'day' column
dataset = dataset.map(add_day)

# Unique days
unique_days = natsort.natsorted(list(set(dataset['day'])))
print("Unique days:", unique_days[:5])

train_datasets = []
test_datasets = []
valid_datasets = []

# Split data for each day and append to respective datasets
day_np_arr = np.array(list(dataset['day']))
use_filter = False
max_articles_per_day = 250
for day in unique_days:
    if use_filter:
        day_data = dataset.filter(lambda x: x['day'] == day)
    else:
        selected_idx = np.where(day_np_arr == day)[0]
        day_data = dataset.select(selected_idx)

    if max_articles_per_day > 0 and len(day_data) > max_articles_per_day:
        random_idx = np.random.choice(np.arange(len(day_data)), size=(max_articles_per_day,), replace=False)
        prev_size = len(day_data)
        day_data = day_data.select(random_idx)
        print(f"! Chopping down {prev_size} to {len(day_data)}")

    print(f"Date: {day} / records: {len(day_data)}")
    if len(day_data) == 1:
        prob = random.random()
        if prob < 0.33:
            train_datasets.append(day_data)
        elif prob > 0.66:
            test_datasets.append(day_data)
        else:
            valid_datasets.append(day_data)
        continue

    train_testvalid = day_data.train_test_split(test_size=0.02)              # 98% train, (1% validation + 1% test)
    train_datasets.append(train_testvalid['train'])

    if len(train_testvalid['test']) == 1:  # assign the example randomly to either validation or test
        assign_to_test = random.random() >= 0.5
        if assign_to_test:
            test_datasets.append(train_testvalid['test'])
        else:
            valid_datasets.append(train_testvalid['test'])
    elif len(train_testvalid['test']) > 1:
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)     # Split the 2% (validation + test) in half test, half valid
        test_datasets.append(test_valid['test'])
        valid_datasets.append(test_valid['train'])

# Concatenating all splits
train_dataset = concatenate_datasets(train_datasets)
test_dataset = concatenate_datasets(test_datasets)
valid_dataset = concatenate_datasets(valid_datasets)

train_test_valid_dataset = DatasetDict({
    'train': train_dataset,
    'valid': valid_dataset,
    'test': test_dataset,
})
print(train_test_valid_dataset)

# Save the different tokenized version
output_dir = "./cc_news_tokenized/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("Output directory created:", output_dir)

for split in ["train", "valid", "test"]:
    with open(os.path.join(output_dir, f"cc_news.{split}.tokens"), "w") as f:
        for ex_idx, example in enumerate(train_test_valid_dataset[split]):
            f.write(f"= {example['title']} ({example['date']}) =\n\n")
            f.write(f"{example['text']}\n\n\n")
