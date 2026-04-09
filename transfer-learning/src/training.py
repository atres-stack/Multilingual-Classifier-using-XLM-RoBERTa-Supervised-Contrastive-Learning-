from typing import TypedDict

import torch


class TrainStepOutput(TypedDict):
    loss: float
    grad_norm: float


def train_step(model, optimizer, criterion, batch_X, batch_y) -> TrainStepOutput:
    model.train()
    device = next(model.parameters()).device

    optimizer.zero_grad(set_to_none=True)

    input_ids = batch_X["input_ids"].to(device) # dimension (BATCH_SIZE, 512)
    attention_mask = batch_X["attention_mask"].to(device) # dimension (BATCH_SIZE, 512)
    labels = batch_y.to(device) # dimension = (BATCH_SIZE)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

    return TrainStepOutput(loss=loss.item(), grad_norm=grad_norm.item())
