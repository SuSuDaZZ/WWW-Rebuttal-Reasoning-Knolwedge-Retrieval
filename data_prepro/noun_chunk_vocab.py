# Lint as: python3
"""Builds the concept vocabulary from the preprocessed corpus."""
import os
import json
import collections

from tqdm import tqdm 

DATA_ROOT = "projects/MCTS_GenericsKB/data/GenericsKB"
INPUT_JSON_PATH = "GKB_large.jsonl"
OUTPUT_VOCAB_PATH = "GKB_large.vocab.txt"

def main():
    data_lines = []
    with open(os.path.join(DATA_ROOT, INPUT_JSON_PATH), "r") as f:
        data_lines = f.read().split("\n")
    print(len(data_lines))
    concept_freq = collections.defaultdict(lambda : 0)
    for line in tqdm(data_lines):
        instance = json.loads(line)
        noun_chunks = instance["noun_chunks"]
        for nc in noun_chunks:
            concept_freq[nc[0].lower()] += 1

    print(len(concept_freq))
    concept_freq_list = [(k, v) for k, v in sorted(concept_freq.items(), key=lambda item: item[1], reverse=True) if v>=1]
    print(len(concept_freq_list))

    with open(os.path.join(DATA_ROOT, OUTPUT_VOCAB_PATH), "w") as f:
        f.write("\n".join([k for k, v in concept_freq_list]))


if __name__ == "__main__":
    main()