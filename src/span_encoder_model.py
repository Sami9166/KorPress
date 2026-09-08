"""KLUE-RoBERTa contextual span classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass
class SpanEncoderConfig:
    model_name: str = "klue/roberta-base"
    pooling: str = "mean_max"
    dropout: float = 0.1
    classifier_hidden_size: int = 256

    def validate(self) -> None:
        if self.pooling not in {"mean", "max", "mean_max"}:
            raise ValueError(f"Unsupported pooling: {self.pooling}")


class ContextualSpanClassifier(nn.Module):
    """Encode context once, pool arbitrary token positions, classify each span.

    ``span_token_masks[b]`` has shape ``[number_of_spans, sequence_length]``.
    It may contain non-contiguous True positions, so dependency-tree spans do not
    need to be represented by a single start/end interval.
    """

    def __init__(
        self,
        config: SpanEncoderConfig,
        encoder_config_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.span_config = config
        if encoder_config_path is None:
            self.encoder = AutoModel.from_pretrained(config.model_name)
        else:
            base_config = AutoConfig.from_pretrained(encoder_config_path)
            self.encoder = AutoModel.from_config(base_config)
        hidden_size = self.encoder.config.hidden_size
        pooled_size = hidden_size * (2 if config.pooling == "mean_max" else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Dropout(config.dropout),
            nn.Linear(pooled_size, config.classifier_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_size, 2),
        )

    @staticmethod
    def _scatter_pool(
        token_hidden: torch.Tensor,
        mask: torch.Tensor,
        pooling: str,
    ) -> torch.Tensor:
        """Pool without constructing a [spans, tokens, hidden] tensor."""
        span_ids, token_ids = mask.nonzero(as_tuple=True)
        if span_ids.numel() == 0:
            raise ValueError("Every context must contain at least one non-empty span")

        members = token_hidden[token_ids]
        span_count = mask.shape[0]
        hidden_size = token_hidden.shape[-1]
        expanded_ids = span_ids[:, None].expand(-1, hidden_size)
        outputs: list[torch.Tensor] = []

        if pooling in {"mean", "mean_max"}:
            sums = token_hidden.new_zeros((span_count, hidden_size))
            sums.scatter_add_(0, expanded_ids, members)
            counts = torch.bincount(span_ids, minlength=span_count).clamp_min(1)
            outputs.append(sums / counts[:, None].to(token_hidden.dtype))

        if pooling in {"max", "mean_max"}:
            maxima = token_hidden.new_full(
                (span_count, hidden_size), torch.finfo(token_hidden.dtype).min
            )
            maxima.scatter_reduce_(
                0, expanded_ids, members, reduce="amax", include_self=True
            )
            outputs.append(maxima)

        return torch.cat(outputs, dim=-1) if len(outputs) == 2 else outputs[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_token_masks: Sequence[torch.Tensor],
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoded.last_hidden_state
        pooled = [
            self._scatter_pool(hidden[b], masks, self.span_config.pooling)
            for b, masks in enumerate(span_token_masks)
        ]
        span_vectors = torch.cat(pooled, dim=0)
        logits = self.classifier(span_vectors)
        result = {
            "logits": logits,
            "drop_probabilities": logits.softmax(dim=-1)[:, 1],
            "span_vectors": span_vectors,
        }
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "span_config": asdict(self.span_config),
                "state_dict": self.state_dict(),
            },
            output / "span_classifier.pt",
        )
        self.encoder.config.save_pretrained(output)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: str | Path, map_location: str | torch.device = "cpu"
    ) -> "ContextualSpanClassifier":
        payload = torch.load(
            Path(checkpoint_dir) / "span_classifier.pt",
            map_location=map_location,
            weights_only=False,
        )
        model = cls(
            SpanEncoderConfig(**payload["span_config"]),
            encoder_config_path=checkpoint_dir,
        )
        model.load_state_dict(payload["state_dict"])
        return model
