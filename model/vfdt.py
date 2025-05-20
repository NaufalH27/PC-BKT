from __future__ import annotations

from typing import Dict, Any, Optional
import math
from collections import Counter, defaultdict
from data_object import StatsTracker, StudentAnswer, Label, Difficulty


class StatsTracker:
    def __init__(self) -> None:
        self.correct_counts: Dict[bool, Counter[Label]] = defaultdict(Counter)
        self.difficulty_counts: Dict[Difficulty, Counter[Label]] = defaultdict(Counter)
        self.label_counts: Counter[Label] = Counter(Label)
        # duration: int

    def update(self, answer: StudentAnswer):
        self.correct_counts[answer.is_correct][answer.label] += 1
        self.difficulty_counts[answer.difficulty][answer.label] += 1

    def get_all_features(self) -> Dict[str, Dict[Any, Counter[Label]]]:
        return {
            "is_correct": self.correct_counts,
            "difficulty": self.difficulty_counts,
        }


def gini(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def hoeffding_bound(R: float, n: int, delta: float) -> float:
    """Calculate Hoeffding bound (epsilon) for splitting confidence."""
    return math.sqrt((R ** 2 * math.log(1.0 / delta)) / (2 * n))


class VFDTNode:
    def __init__(self) -> None:
        self.stats: StatsTracker = StatsTracker()
        self.children: Dict[Any, VFDTNode] = {}

    def try_split(self, nmin: int, delta: float) -> bool:
        n = sum(self.stats.label_counts.values())
        if n < nmin:
            return False

        gini_scores: Dict[str, float] = {}
        for feature, value_bins in self.stats.get_all_features().items():
            total = sum(sum(c.values()) for c in value_bins.values())
            weighted_gini = 0.0
            for val_counts in value_bins.values():
                count = sum(val_counts.values())
                weighted_gini += (count / total) * gini(val_counts)
            gini_scores[feature] = weighted_gini

        if len(gini_scores) < 2:
            return False

        sorted_features = sorted(gini_scores.items(), key=lambda x: x[1])
        best_feature, g_best = sorted_features[0]
        _, g_second = sorted_features[1]

        epsilon = hoeffding_bound(R=1.0, n=n, delta=delta)

        if (g_second - g_best > epsilon) or (epsilon < 0.05):
            self.split_feature = best_feature
            for feature_val in self.stats.get_all_features()[best_feature]:
                self.children[feature_val] = VFDTNode()
            return True

        return False


class VFDT:
    def __init__(self, nmin: int = 100, delta: float = 0.01):
        self.root = VFDTNode()
        self.nmin = nmin
        self.delta = delta

    def _traverse(self, answer: StudentAnswer) -> VFDTNode:
        node = self.root
        while node.children:
            feature = node.split_feature
            value = getattr(answer, feature)
            if value in node.children:
                node = node.children[value]
            else:
                break
        return node

    def update(self, answer: StudentAnswer):
        leaf = self._traverse(answer)
        leaf.stats.update(answer)
        leaf.stats.label_counts[answer.label] += 1
        leaf.try_split(self.nmin, self.delta)

    def predict(self, answer: StudentAnswer) -> Optional[Label]:
        leaf = self._traverse(answer)
        if not leaf.stats.label_counts:
            return None
        return max(leaf.stats.label_counts.items(), key=lambda x: x[1])[0]
