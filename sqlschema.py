from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
import datetime

Base = declarative_base()

class HCPInteraction(Base):
    __tablename__ = "hcp_interactions"

    id = Column(Integer, primary_key=True, index=True)
    hcp_name = Column(String(255), nullable=False) # [cite: 24]
    interaction_type = Column(String(50)) # [cite: 27]
    date_time = Column(DateTime, default=datetime.datetime.utcnow) # [cite: 26, 29]
    attendees = Column(JSON) # [cite: 38]
    topics_discussed = Column(Text) # [cite: 40]
    sentiment = Column(String(20)) # [cite: 45]
    outcomes = Column(Text) # [cite: 47]
    follow_up_actions = Column(Text) # [cite: 49]
    materials_shared = Column(JSON) # [cite: 42]
    samples_distributed = Column(JSON) # [cite: 44]