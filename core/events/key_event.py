from core.events.event import (
    Event,
    EventType,
    EventCategory
)


class KeyEvent(Event):

    CATEGORIES = (
        EventCategory.INPUT
        | EventCategory.KEYBOARD
    )

    def __init__(self, key):
        super().__init__()

        self.key = key


class KeyPressedEvent(KeyEvent):

    EVENT_TYPE = EventType.KEY_PRESSED

    def __init__(self, key, repeat=False):
        super().__init__(key)

        self.repeat = repeat

    def __str__(self):
        return (
            f"KeyPressedEvent: "
            f"{self.key}, "
            f"repeat={self.repeat}"
        )


class KeyReleasedEvent(KeyEvent):

    EVENT_TYPE = EventType.KEY_RELEASED

    def __str__(self):
        return (
            f"KeyReleasedEvent: "
            f"{self.key}"
        )


class KeyTypedEvent(KeyEvent):

    EVENT_TYPE = EventType.KEY_TYPED

    def __init__(self, codepoint):
        super().__init__(codepoint)

        self.codepoint = codepoint

    @property
    def character(self):
        return chr(self.codepoint)

    def __str__(self):
        return (
            f"KeyTypedEvent: "
            f"{self.character}"
        )