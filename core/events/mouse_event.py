from core.events.event import (
    Event,
    EventType,
    EventCategory
)


class MouseMovedEvent(Event):

    EVENT_TYPE = EventType.MOUSE_MOVED

    CATEGORIES = (
        EventCategory.INPUT
        | EventCategory.MOUSE
    )

    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y

    def __str__(self):
        return (
            f"MouseMovedEvent: "
            f"{self.x}, {self.y}"
        )


class MouseScrolledEvent(Event):

    EVENT_TYPE = EventType.MOUSE_SCROLLED

    CATEGORIES = (
        EventCategory.INPUT
        | EventCategory.MOUSE
    )

    def __init__(self, x_offset, y_offset):
        super().__init__()

        self.x_offset = x_offset
        self.y_offset = y_offset

    def __str__(self):
        return (
            f"MouseScrolledEvent: "
            f"{self.x_offset}, "
            f"{self.y_offset}"
        )


class MouseButtonEvent(Event):

    CATEGORIES = (
        EventCategory.INPUT
        | EventCategory.MOUSE
        | EventCategory.MOUSE_BUTTON
    )

    def __init__(self, button):
        super().__init__()

        self.button = button


class MouseButtonPressedEvent(MouseButtonEvent):

    EVENT_TYPE = EventType.MOUSE_BUTTON_PRESSED

    def __str__(self):
        return (
            f"MouseButtonPressedEvent: "
            f"{self.button}"
        )


class MouseButtonReleasedEvent(MouseButtonEvent):

    EVENT_TYPE = EventType.MOUSE_BUTTON_RELEASED

    def __str__(self):
        return (
            f"MouseButtonReleasedEvent: "
            f"{self.button}"
        )