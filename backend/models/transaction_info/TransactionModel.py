from pydantic import BaseModel
from datetime import datetime

class TransactionModel(BaseModel):
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool
    total_points_home: float
    previous_points_home: float
    rank_home: float
    rank_change_home: float
    total_points_away: float
    previous_points_away: float
    rank_away: float
    rank_change_away: float

