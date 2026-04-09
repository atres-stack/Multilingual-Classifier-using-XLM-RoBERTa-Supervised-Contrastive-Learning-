from torch import Tensor, nn
from torch.nn import functional
from transformers import XLMRobertaModel

from pooler import Pooler


class RobertaClassifier(nn.Module):
    def __init__(self, encoder: XLMRobertaModel, num_classes: int) -> None:
        super().__init__()
        self.encoder: XLMRobertaModel = encoder
        self.pooler = Pooler()
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(input_ids, attention_mask=attention_mask)
        mean_embedding = self.pooler(attention_mask=attention_mask, outputs=out)
        norm_feat = functional.normalize(mean_embedding)
        logits = self.classifier(norm_feat)
        return logits
