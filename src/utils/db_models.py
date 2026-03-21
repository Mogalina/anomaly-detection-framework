import time

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, func
from sqlalchemy.dialects.postgresql import JSONB

from utils.db import Base


class CoordinatorState(Base):
    """
    Singleton row representing the coordinator's durable state.
    Written after every successful round via checkpoint.
    """
    __tablename__ = 'coordinator_state'

    id = Column(Integer, primary_key=True, default=1)
    current_round = Column(Integer, nullable=False, default=0)
    global_model_path = Column(String(512), nullable=True)
    updated_at = Column(Float, nullable=False, default=time.time)

    def __repr__(self) -> str:
        return (
            f"<CoordinatorState round={self.current_round} "
            f"model={self.global_model_path}>"
        )


class RegisteredClient(Base):
    """
    One row per client that has ever registered with the coordinator.
    Upserted on every Register RPC so clients survive coordinator restarts.
    """
    __tablename__ = 'registered_clients'

    client_id   = Column(String(255), primary_key=True)
    metadata_json = Column(Text, nullable=True)
    registered_at = Column(Float, nullable=False, default=time.time)
    last_seen = Column(Float, nullable=False, default=time.time)
    rounds_participated = Column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return (
            f"<RegisteredClient id={self.client_id} "
            f"rounds={self.rounds_participated}>"
        )


class RoundHistory(Base):
    """
    One row per completed FL round. Useful for auditing, plotting, and resuming 
    a training run after coordinator restart.
    """
    __tablename__ = 'round_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    round_num = Column(Integer, nullable=False, index=True)
    num_clients = Column(Integer, nullable=False)
    total_samples = Column(Integer, nullable=False)
    aggregation_time = Column(Float, nullable=False)
    timestamp = Column(Float, nullable=False, default=time.time)

    def __repr__(self) -> str:
        return (
            f"<RoundHistory round={self.round_num} "
            f"clients={self.num_clients} samples={self.total_samples}>"
        )
