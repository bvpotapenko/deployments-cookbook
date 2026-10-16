"""DeepSeek's OCR no-repeat-ngram rule as an sglang CustomLogitProcessor.

Copied into the engine venv's site-packages by ``kot_serve.sh`` (beside
sitecustomize.py) so dill resolves the class by name inside the scheduler
processes, whose environment carries no PYTHONPATH. The adapter serializes
the class with ``to_str()`` and passes it per request as
``custom_logit_processor``; the request's ``sampling_params`` carries an
empty ``custom_params`` dict, which the scheduler fills with ``__req__``.

sglang applies custom processors inside ``Sampler.forward``, after the
stash sitecustomize.py writes at that call's entry, so the stashed rows
stay the raw pre-processing distributions; this rule changes which token
is sampled, never what is reported. The values are DeepSeek's own (their
DeepSeek-OCR2-vllm run scripts; sglang ships the same as its native
default): a token that would complete a 30-gram that already occurred
within a 90-token lookback is banned, except the table tokens.
"""

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = frozenset({128821, 128822})  # <td>, </td>


class KotOCR2NoRepeatNGram(CustomLogitProcessor):
    """Ban tokens that would complete a repeated n-gram, per request row."""

    def __call__(self, logits, custom_param_list=None):
        for index, params in enumerate(custom_param_list or []):
            request = (params or {}).get("__req__")
            if request is None:
                continue
            ids = list(request.origin_input_ids) + list(request.output_ids)
            banned = _banned_tokens(ids)
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
