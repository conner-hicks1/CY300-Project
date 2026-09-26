from enum import Enum, IntFlag, auto


class EventType(Enum):
    NONE = 0

    WINDOW_CLOSE = auto()
    WINDOW_RESIZE = auto()
    WINDOW_FOCUS = auto()
    WINDOW_LOST_FOCUS = auto()

    KEY_PRESSED = auto()
    KEY_RELEASED = auto()
    KEY_TYPED = auto()

    MOUSE_BUTTON_PRESSED = auto()
    MOUSE_BUTTON_RELEASED = auto()
    MOUSE_MOVED = auto()
    MOUSE_SCROLLED = auto()


class EventCategory(IntFlag):
    NONE = 0

    APPLICATION = auto()
    INPUT = auto()
    KEYBOARD = auto()
    MOUSE = auto()
    MOUSE_BUTTON = auto()


class Event:

    EVENT_TYPE = EventType.NONE
    CATEGORIES = EventCategory.NONE

    def __init__(self):
        self.handled = False

    @property
    def event_type(self):
        return self.EVENT_TYPE

    @property
    def categories(self):
        return self.CATEGORIES

    def is_in_category(self, category):
        return bool(self.categories & category)

    def __str__(self):
        return self.__class__.__name__


class EventDispatcher:

    def __init__(self, event):
        self.event = event

    def dispatch(self, event_class, handler):

        if isinstance(self.event, event_class):

            result = handler(self.event)

            if result:
                self.event.handled = True

            return True

        return False