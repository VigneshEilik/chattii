from typing import Any, Iterator, List, Optional

import torch
import torch.nn as nn

from engine.inference.config import GenerationConfig
from engine.inference.logits.processor import LogitsProcessorList
from engine.inference.runtime.state import GenerationRequestState
from engine.inference.runtime.stopping_criteria import CompositeStoppingCriteria
from engine.inference.runtime.token_generator import SingleTokenGenerator


class AutoregressiveGenerationLoop:
    """Executes the autoregressive generation loop, passing only new tokens when cache is enabled."""

    @staticmethod
    def generate(
        model: nn.Module,
        prompt_token_ids: List[int],
        config: GenerationConfig,
        cache: Optional[Any] = None,
        processors: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[CompositeStoppingCriteria] = None,
    ) -> Iterator[int]:
        processors = processors or LogitsProcessorList()
        params = list(model.parameters())
        device = params[0].device if params else torch.device("cpu")

        state = GenerationRequestState(prompt_token_ids)
        current_input_ids = torch.tensor(
            [prompt_token_ids], dtype=torch.long, device=device
        )

        step = 0
        while True:
            # Step evaluation: if caching is enabled and we are past step 0, only pass the last token
            if config.use_cache and cache is not None and step > 0:
                step_ids = current_input_ids[:, -1:]
            else:
                step_ids = current_input_ids

            # Forward pass
            logits = SingleTokenGenerator.step(
                model, step_ids, cache=cache, use_cache=config.use_cache
            )

            # Apply penalties and processors
            processed_logits = processors(current_input_ids, logits)

            # Decode next token
            if config.do_sample:
                probs = torch.softmax(processed_logits, dim=-1)
                probs = probs / torch.sum(probs, dim=-1, keepdim=True)
                next_token_tensor = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_token_tensor = torch.argmax(processed_logits, dim=-1)

            next_token = next_token_tensor.item()
            state.add_token(next_token)

            if step == 0:
                state.record_first_token()

            yield next_token

            # Update history and stop check
            current_input_ids = torch.cat(
                [current_input_ids, next_token_tensor.unsqueeze(0)], dim=1
            )

            if stopping_criteria is not None and stopping_criteria(
                state.total_sequence, next_token
            ):
                break

            step += 1
            if step >= config.max_new_tokens:
                break
