from typing import TypedDict

import torch


class ValidStepOutput(TypedDict):
    logits: torch.Tensor
    loss: float


@torch.no_grad()
def valid_step(model, criterion, batch_X, batch_y) -> ValidStepOutput:
    model.eval()

    device = next(model.parameters()).device

    input_ids = batch_X["input_ids"].to(device) # dimension (BATCH_SIZE, 512)
    attention_mask = batch_X["attention_mask"].to(device) # dimension (BATCH_SIZE, 512)
    labels = batch_y.to(device) # dimension = (BATCH_SIZE)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

    return ValidStepOutput(logits=logits.detach().cpu(), loss=loss.item())
