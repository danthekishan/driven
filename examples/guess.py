import random

from pydantic import BaseModel

from driven import tool, Extension


class GuessInput(BaseModel):
    guess: int


class NewGameInput(BaseModel):
    min_number: int = 1
    max_number: int = 100


class HintInput(BaseModel):
    target: int


class NumberGuessExtension(Extension):
    name = "guess"
    description = "Number guessing game tools."

    def __init__(self):
        self._target: int = 0
        self._min: int = 1
        self._max: int = 100
        self._attempts: int = 0
        super().__init__()

    async def start(self):
        self._target = random.randint(self._min, self._max)

    @tool(description="Start a new number guessing game.")
    async def new_game(self, input: NewGameInput) -> dict:
        self._min = input.min_number
        self._max = input.max_number
        self._target = random.randint(self._min, self._max)
        self._attempts = 0
        return {
            "message": f"Guess a number between {self._min} and {self._max}!",
            "range": [self._min, self._max],
        }

    @tool(description="Make a guess.")
    async def guess(self, input: GuessInput) -> dict:
        self._attempts += 1
        if input.guess == self._target:
            return {"result": "correct", "number": self._target, "attempts": self._attempts}
        if input.guess < self._target:
            return {"result": "too_low", "hint": "go higher", "attempts": self._attempts}
        return {"result": "too_high", "hint": "go lower", "attempts": self._attempts}

    @tool(description="Get a hint about the target number.")
    async def hint(self, input: HintInput) -> dict:
        diff = abs(input.target - self._target)
        if diff == 0:
            proximity = "exact"
        elif diff <= 5:
            proximity = "very_close"
        elif diff <= 15:
            proximity = "close"
        elif diff <= 30:
            proximity = "warm"
        else:
            proximity = "cold"
        return {"proximity": proximity, "attempts": self._attempts}
