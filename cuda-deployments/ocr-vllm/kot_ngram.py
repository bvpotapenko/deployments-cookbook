"""DeepSeek's OCR no-repeat-ngram rule as a vLLM V1 LogitsProcessor.

Wired into the engine through ``--logits-processors``; ``kot_serve.sh``
copies this module into the engine venv's site-packages so the FQCN
resolves in every worker. The rule is DeepSeek's own (their
DeepSeek-OCR2-vllm run scripts; sglang ships the same values as its native
default): a token that would complete an n-gram that already occurred
within the lookback window is banned from sampling, except the table
tokens, so table structure may repeat.

The logprobs kot stores stay raw: ``logprobs_mode`` defaults to
``raw_logprobs``, which computes them from the logits before this stage
runs.
"""

from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = frozenset({128821, 128822})  # <td>, </td>


class KotNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Ban tokens that would complete a repeated n-gram, per request."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device
        # batch row -> (prompt ids, live reference to the running output ids)
        self._requests: dict = {}

    @staticmethod
    def _add_request(params, prompt_tok_ids, output_tok_ids):
        return (list(prompt_tok_ids or []), output_tok_ids)

    def is_argmax_invariant(self):
        return False  # the rule changes which token greedy sampling picks

    def update_state(self, batch_update: BatchUpdate | None):
        process_dict_updates(self._requests, batch_update, self._add_request)

    def apply(self, logits):
        for index, (prompt_ids, output_ids) in self._requests.items():
            banned = _banned_tokens(prompt_ids + list(output_ids))
            if banned:
                logits[index, banned] = -float("inf")
        return logits


def _banned_tokens(ids) -> list:
    if len(ids) < NGRAM_SIZE:
        return []
    current_prefix = tuple(ids[-(NGRAM_SIZE - 1):])
    search_start = max(0, len(ids) - NGRAM_WINDOW)
    search_end = len(ids) - NGRAM_SIZE + 1
    banned = set()
    for i in range(search_start, search_end):
        ngram = tuple(ids[i:i + NGRAM_SIZE])
        if ngram[:-1] == current_prefix:
            banned.add(ngram[-1])
    return sorted(banned - NGRAM_WHITELIST)
