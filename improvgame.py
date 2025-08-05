from pydantic import BaseModel, Field


class ImprovGame(BaseModel):
    game_name: str = Field(description="The name of the game")

    description: str = Field(description="A short description of the game")

    rules: list[str] = Field(description="The rules of the game")

    context: str = Field(description="The context of why you picked the game")

    def __str__(self):
        return f"Game: {self.game_name}\nDescription: {self.description}\nRules: {self.rules}\nContext: {self.context}"