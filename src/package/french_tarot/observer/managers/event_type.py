from enum import Enum, auto


class EventType(Enum):
    DUMMY = auto()
    KILL_ALL = auto()
    OBSERVATION = auto()
    ACTION = auto()
    ACTION_RESULT = auto()
    MODEL_UPDATE = auto()
    RESET_ENVIRONMENT = auto()
