from core.events.event import (
    Event,
    EventType,
    EventCategory
)


class WindowCloseEvent(Event):

    EVENT_TYPE = EventType.WINDOW_CLOSE
    CATEGORIES = EventCategory.APPLICATION


class WindowResizeEvent(Event):

    EVENT_TYPE = EventType.WINDOW_RESIZE
    CATEGORIES = EventCategory.APPLICATION

    def __init__(
        self,
        width: int,
        height: int
    ):

        super().__init__()

        self.width = width
        self.height = height

    def __str__(self):

        return (
            f"WindowResizeEvent: "
            f"{self.width}x{self.height}"
        )


class WindowFocusEvent(Event):

    EVENT_TYPE = EventType.WINDOW_FOCUS
    CATEGORIES = EventCategory.APPLICATION

    def __init__(self):

        super().__init__()

    def __str__(self):

        return "WindowFocusEvent"


class WindowLostFocusEvent(Event):

    EVENT_TYPE = EventType.WINDOW_LOST_FOCUS
    CATEGORIES = EventCategory.APPLICATION

    def __init__(self):

        super().__init__()

    def __str__(self):

        return "WindowLostFocusEvent"