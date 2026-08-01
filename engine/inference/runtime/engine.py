from typing import Any, Iterator, List, Optional

import torch.nn as nn

from engine.inference.config import GenerationConfig
from engine.inference.logits.penalties import (
    FrequencyPenaltyProcessor,
    PresencePenaltyProcessor,
    RepetitionPenaltyProcessor,
)
from engine.inference.logits.processor import LogitsProcessorList
from engine.inference.runtime.generation_loop import AutoregressiveGenerationLoop
from engine.inference.runtime.stopping_criteria import (
    CompositeStoppingCriteria,
    StopOnEOS,
    StopOnTokens,
)


class GenerationEngine:
    """High-level orchestration engine coordinating token iterations, caches, and stopping criteria."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def execute(
        self,
        prompt_token_ids: List[int],
        config: GenerationConfig,
        cache: Optional[Any] = None,
        eos_token_id: Optional[int] = None,
    ) -> Iterator[int]:
        # Build stopping criteria
        criteria = []
        if eos_token_id is not None:
            criteria.append(StopOnEOS(eos_token_id))
        criteria.append(StopOnTokens(len(prompt_token_ids) + config.max_new_tokens))
        stopping = CompositeStoppingCriteria(criteria)

        # Build penalty processors
        processors = LogitsProcessorList()
        if config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyProcessor(config.repetition_penalty))
        if config.frequency_penalty != 0.0:
            processors.append(FrequencyPenaltyProcessor(config.frequency_penalty))
        if config.presence_penalty != 0.0:
            processors.append(PresencePenaltyProcessor(config.presence_penalty))

        # Run loop
        return AutoregressiveGenerationLoop.generate(
            model=self.model,
            prompt_token_ids=prompt_token_ids,
            config=config,
            cache=cache,
            processors=processors,
            stopping_criteria=stopping,
        )
