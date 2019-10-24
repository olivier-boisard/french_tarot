from attr import dataclass

from french_tarot.observer.managers.manager import EventType


@dataclass
class Message:
    event_type: EventType
    data: any
