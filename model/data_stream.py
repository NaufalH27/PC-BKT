import random
from data_object import StatsTracker, StudentAnswer, Label, Difficulty


def stream_data():
    randomize_is_correct: bool = random.choices([True, False], weights=[0.8, 0.2])[0]
    randomize_duration: float = round(random.uniform(0.5, 15.0), 2)
    randomize_difficulty: Difficulty = random.choice(diff)
    label = label_answer(randomize_is_correct, randomize_duration, randomize_difficulty)

    return StudentAnswer(
            is_correct = randomize_is_correct,
            difficulty = ,
            )


def label_answer(is_correct: bool, duration:float, difficulty: Difficulty) -> Label:
    if not is_correct:
        return Label.LOW
    if difficulty == "hard":
        if duration < 120:
            return "high"
        else:
            return "medium"
    elif difficulty == "medium":
        if duration < 60:
            return "high"
        else:
            return "medium"
    elif difficulty == "easy":
        if duration < 30:
            return "high"
        else:
            return "medium"
    return "low"
