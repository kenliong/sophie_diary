from pydantic import BaseModel
import json
from typing import List, Set, Union, Dict
from random import randint

sahha_scores_path: str = 'sample_sahha_scores.json'
activity_factors: Set[str] = set(['steps', 'active_hours', 'active_calories', 'intense_activity_duration', 'extended_inactivity', 'floors_climbed'])
sleep_factors: Set[str] = set(['sleep_duration', 'sleep_regularity', 'sleep_continuity', 'sleep_debt', 'circadian_alignment', 'physical_recovery', 'mental_recovery'])


class SahhaScore(BaseModel):
    """
    Represents a single health or activity factor with its associated metrics.

    Attributes:
        name (str): The name of the health or activity factor.
        score (Union[float, int, None]): The computed score for this factor
        state (Union[str, None]): The qualitative state associated with the score
        unit (str): The unit of measurement for the value (e.g., 'count', 'minute', 'kcal').
        value (Union[float, None]): The actual measured value for this factor, or None if not measured.
        goal (int): The target or goal value for this factor.
    """
    name: str
    score: Union[float, int, None]
    state: Union[str, None]
    unit: str
    value: Union[float, None]
    goal: int

    def __hash__(self):
        return hash((self.name, self.score, self.state, self.unit, self.value, self.goal))

    def to_dict(self):
        return {
            "name": self.name,
            "score": self.score,
            "state": self.state,
            "unit": self.unit,
            "value": self.value,
            "goal": self.goal
        }

class SahhaUser(BaseModel):
    """
    Represents a user's wellbeing profile with various health and activity scores.

    Attributes:
        id (str): Unique identifier for the user.
        profileId (str): Identifier linking to the user's profile.
        type (str): Type of profile, typically describing the category of wellbeing (e.g., 'wellbeing').
        score (float): Overall wellbeing score calculated for the user.
        state (str): Qualitative assessment of the user's overall wellbeing state
        factors (List[SahhaScore]): List of SahhaScore objects representing individual factors contributing to the overall score.
    """
    id: str
    profileId: str
    type: str
    score: float  # wellbeing score
    state: str
    factors: List[SahhaScore]

    def to_dict(self):
        return {
            "id": self.id,
            "profileId": self.profileId,
            "type": self.type,
            "score": self.score,
            "state": self.state,
            # "factors": [factor.to_dict() for factor in self.factors]
        }

def get_user_details(file_path: str) -> list:
    # Read json data
    with open(file_path, 'r') as file:
        data: dict = json.load(file)

    user_details: List[SahhaUser, Set[SahhaScore], Set[SahhaScore]] = []
    # Parse json data
    for user_obj in data:
        user: SahhaUser = SahhaUser(**user_obj)
        activity_scores: Set[SahhaScore] = {factor for factor in user.factors if factor.name in activity_factors}
        sleep_scores: Set[SahhaScore] = {factor for factor in user.factors if factor.name in sleep_factors}
        user_details.append([user, activity_scores, sleep_scores])

    return user_details

def main():
    # Read the score.json

    user_list: List[SahhaUser, Set[SahhaScore], Set[SahhaScore]] = get_user_details(sahha_scores_path)
    # Sahha User Overview, Sahha User Activity Score, Sahha User Sleep Score

    # Return a random User
    random_user: List[SahhaUser, Set[SahhaScore], Set[SahhaScore]] = user_list[randint(0,len(user_list)-1)]

    # Convert random_user to a dictionary
    random_user_dict: dict = {
        "user": random_user[0].to_dict(),
        "activity_scores": [score.to_dict() for score in random_user[1]],
        "sleep_scores": [score.to_dict() for score in random_user[2]]
    }

    # Serialize to JSON
    random_user_json = json.dumps(random_user_dict, indent=4)
    print(random_user_json)

if __name__ == "__main__":
    main()

