from sqlalchemy import Column, Integer, String, Text, DateTime
from database import Base
import datetime

class HCPInteraction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    hcp_name = Column(String(255))
    interaction_type = Column(String(100))
    date = Column(String(100))
    topics = Column(Text)
    sentiment = Column(String(50))
    outcomes = Column(Text)
    follow_ups = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)