import json

from collections import defaultdict
from typing import Dict
from typing import List

EXPERIMENT_DOMAINS = set(["hotel", "train", "restaurant", "attraction", "taxi"])
EXCLUDE_DOMAINS = set(["hospital", "police"])


def get_slot_information(ontology: Dict[str, List[str]]) -> List:
    ontology_domains = dict(
        [(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS]
    )
    SLOTS = [
        k.replace(" ", "").lower() if ("book" not in k) else k.lower()
        for k in ontology_domains.keys()
    ]

    return SLOTS


def collate_fn(tokenizer, converter):
    def _collate(batch):
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [d[key] for d in batch]

        input_batch = tokenizer(
            batch_data["intput_text"],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            verbose=False,
            truncation=True,
            max_length=1000,
        )

        if "output_text" not in batch_data:
            batch_data["output_text"] = [converter.state_to_sum(x) for x in batch_data["slot_values"]]

        batch_data["encoder_input"] = input_batch["input_ids"]
        batch_data["attention_mask"] = input_batch["attention_mask"]
        batch_data["decoder_output"] = tokenizer(
            batch_data["output_text"],
            padding=True,
            return_tensors="pt", # non-padded return List[List[Int]]
            return_attention_mask=False,
            truncation=True,
            max_length=200,
        ).input_ids

        return batch_data
    return _collate


def normalize_ontology(ontology: Dict[str, List[str]]) -> Dict[str, List[str]]:
    keys = [k for k in ontology]
    for k in keys:
        for i in range(len(ontology[k])):
            ontology[k][i] = ontology[k][i].replace("do n't care", "dontcare")
            ontology[k][i] = ontology[k][i].replace("'s", " s")

        ontology[
            k.replace(" ", "").lower() if ("book" not in k) else k.lower()
        ] = ontology.pop(k)

    return ontology

