import torch

ROBERTA_MAX_SEQUENCE_LENGTH = 512
STAFF_TYPE_MAP = {
    "class-0": 0,
    "class-1": 1,
    "class-2": 2,
}


def collate_function(data: list[tuple[str, int]], tokenizer):
    texts, labels = zip(*data, strict=True)

    labels = [STAFF_TYPE_MAP[label] for label in labels]

    encoding = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=ROBERTA_MAX_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return (
        {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        },
        torch.tensor(labels, dtype=torch.long),
    )
