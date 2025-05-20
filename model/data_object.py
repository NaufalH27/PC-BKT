from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any
from collections import Counter, defaultdict


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Label(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class StudentAnswer:
    is_correct: bool
    # duration: int
    difficulty: Difficulty
    label: Label


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
