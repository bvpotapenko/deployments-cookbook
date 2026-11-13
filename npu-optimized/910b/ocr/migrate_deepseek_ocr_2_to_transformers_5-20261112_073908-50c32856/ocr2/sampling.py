"""Sampling for the decode path, mirroring transformers 4.46.3's generate()
semantics for the fields the contract sends, plus DeepSeek's own OCR
anti-repetition rule.

The stored log-probability rows are always the log-softmax of the raw logits
before any processing.  Sampling applies, in the order 4.46.3 applied them:
the logits processors (the no-repeat n-gram ban, the EOS ban while below
min_new_tokens, repetition penalty), then the warpers (temperature, top_k,
top_p, min_p), then torch.multinomial.
"""
import torch

NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = {128821, 128822}  # <td>, </td>


class NoRepeatNGramLogitsProcessor:
    """A token that would complete an n-gram already seen within the last
    window tokens is banned from sampling, except the table tokens, so table
    structure may repeat.  Verbatim from the example deployment (scores here
    are the one row being sampled: [vocab])."""

    def __init__(self, ngram_size=NGRAM_SIZE, window_size=NGRAM_WINDOW,
                 whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = NGRAM_WHITELIST if whitelist_token_ids is None else whitelist_token_ids

    def __call__(self, ids, scores):
        if len(ids) < self.ngram_size:
            return scores
        current_prefix = tuple(ids[-(self.ngram_size - 1):])
        search_start = max(0, len(ids) - self.window_size)
        search_end = len(ids) - self.ngram_size + 1
        banned = set()
        for index in range(search_start, search_end):
            ngram = tuple(ids[index:index + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned.add(ngram[-1])
        banned -= self.whitelist_token_ids
        if banned:
            scores = scores.clone()
            for token in banned:
                scores[token] = -float("inf")
        return scores


class Sampler:
    """One sampler per request, holding its sampling record and generator."""

    def __init__(self, sampling, seed, eos_token_id, ngram=True):
        self.temperature = float(sampling.get("temperature", 1.0))
        self.top_p = float(sampling.get("top_p", 1.0))
        self.top_k = int(sampling.get("top_k", 0))
        self.min_p = float(sampling.get("min_p", 0.0))
        self.repetition_penalty = float(sampling.get("repetition_penalty", 1.0))
        self.presence_penalty = float(sampling.get("presence_penalty", 0.0))
        if self.presence_penalty != 0.0:
            raise ValueError(
                "unsupported sampling value: presence_penalty must be 0 (no engine "
                "knob applies it, and a value this deployment cannot apply is an "
                "error, never ignored)"
            )
        if self.temperature == 0.0:
            raise ValueError("temperature 0 is not a sampling path this model runs")
        self.eos_token_id = eos_token_id
        self.ngram = NoRepeatNGramLogitsProcessor() if ngram else None
        self._seed = seed
        self.generators = {}

    def _generator(self, device):
        key = str(device)
        if key not in self.generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._seed)
            self.generators[key] = generator
        return self.generators[key]

    def process(self, logits, consumed_ids, allow_eos, min_new_tokens, n_generated):
        """Apply processors + warpers to one fp32 logits row [vocab]."""
        scores = logits
        if self.ngram is not None:
            scores = self.ngram(consumed_ids, scores)
        if not allow_eos and n_generated < min_new_tokens:
            scores = scores.clone()
            scores[self.eos_token_id] = -float("inf")
        if self.repetition_penalty != 1.0:
            scores = scores.clone()
            for token in set(consumed_ids):
                score = scores[token]
                scores[token] = score / self.repetition_penalty if score >= 0 else score * self.repetition_penalty
        scores = scores / self.temperature
        if self.top_k and self.top_k > 0:
            kth = torch.topk(scores, k=min(self.top_k, scores.shape[-1]), dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < kth, -float("inf"))
        if self.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
            keep = cumulative - torch.softmax(sorted_scores, dim=-1) < self.top_p
            keep[..., 0] = True
            remove = ~keep
            sorted_scores[remove] = -float("inf")
            scores = sorted_scores.scatter(-1, sorted_indices, sorted_scores)
        if self.min_p and self.min_p > 0.0:
            probs = torch.softmax(scores, dim=-1)
            keep = probs >= probs.max(dim=-1, keepdim=True).values * self.min_p
            scores = scores.masked_fill(~keep, -float("inf"))
        return scores

    def sample(self, scores):
        """torch.multinomial over the processed distribution, as generate() did."""
        probs = torch.softmax(scores, dim=-1)
        index = torch.multinomial(probs, num_samples=1, generator=self._generator(probs.device))
        return int(index.item())
