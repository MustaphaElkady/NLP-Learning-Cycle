from pydantic import BaseModel

class PredictRequest(BaseModel):
    text: str
    max_tokens: int = 1


class PredictResponse(BaseModel):
    input_text: str
    predicted_tokens: list[str]
