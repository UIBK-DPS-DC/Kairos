"""Provider ("space") predictors: where does a task run next?

Each series (one per task) is the chronological sequence of providers the task
ran on. The Markov model predicts the next provider from the last ``order``
providers and backs off to shorter contexts and finally to the most frequent
provider, so it captures round-robin rotation as well as a static home
provider that a task returns to after a failure.

The pattern Markov model is task- and provider-agnostic: providers in a context
are relabelled by order of first appearance (e.g. B, C, B -> 0, 1, 0), so it
learns patterns like "rotate" or "return home" that transfer to unseen tasks.
It backs the pretrained fallback model (see fallback_model.py).
"""

from collections import Counter
from typing import Dict, List, Optional


def _most_common(counts: Counter) -> Optional[str]:
    """Highest count; ties broken alphabetically so predictions are deterministic."""
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def train_most_frequent(sequences: Dict[str, List[str]]) -> Dict[str, Counter]:
    """Count how often each provider was used per series."""
    return {key: Counter(providers) for key, providers in sequences.items()}


def predict_most_frequent(model: Dict[str, Counter], key: str) -> Optional[str]:
    return _most_common(model.get(key, Counter()))


def train_markov(sequences: Dict[str, List[str]], order: int = 2) -> Dict:
    """Count next-provider transitions for every context of length 1..order per series."""
    series = {}
    for key, providers in sequences.items():
        contexts: Dict[tuple, Counter] = {}
        for i in range(1, len(providers)):
            for n in range(1, min(order, i) + 1):
                contexts.setdefault(tuple(providers[i - n : i]), Counter())[providers[i]] += 1
        series[key] = {"contexts": contexts, "frequency": Counter(providers)}
    return {"order": order, "series": series}


def predict_markov(
    model: Dict, key: str, recent_providers: List[str], backoff: bool = True
) -> Optional[str]:
    """Predict the next provider of series ``key`` given its most recent providers.

    With ``backoff=False`` only the full-order context is used, i.e. None unless
    exactly this context was seen in training.
    """
    info = model["series"].get(key)
    if info is None:
        return None
    longest = min(model["order"], len(recent_providers))
    for n in range(longest, 0 if backoff else longest - 1, -1):
        counts = info["contexts"].get(tuple(recent_providers[-n:]))
        if counts:
            return _most_common(counts)
    return _most_common(info["frequency"]) if backoff else None


NEW = "new"  # pattern label: next provider does not occur in the context


def _canonical(context: List[str]) -> tuple[tuple[str, ...], List[str]]:
    """Relabel providers by first appearance; return (pattern, label -> provider)."""
    labels: List[str] = []
    for p in context:
        if p not in labels:
            labels.append(p)
    return tuple(str(labels.index(p)) for p in context), labels


def train_pattern_markov(sequences: List[List[str]], order: int = 4) -> Dict:
    """Count next-provider labels for every canonical context of length 1..order."""
    patterns: Dict[str, Counter] = {}
    for providers in sequences:
        for i in range(1, len(providers)):
            for n in range(1, min(order, i) + 1):
                pattern, labels = _canonical(providers[i - n : i])
                nxt = providers[i]
                label = str(labels.index(nxt)) if nxt in labels else NEW
                patterns.setdefault(",".join(pattern), Counter())[label] += 1
    return {"order": order, "patterns": patterns}


def predict_pattern_markov(model: Dict, recent_providers: List[str]) -> Optional[str]:
    """Predict the next provider; None if the best guess is a provider not in the context."""
    for n in range(min(model["order"], len(recent_providers)), 0, -1):
        pattern, labels = _canonical(recent_providers[-n:])
        counts = model["patterns"].get(",".join(pattern))
        if counts:
            label = _most_common(Counter(counts))
            return None if label == NEW else labels[int(label)]
    return None
