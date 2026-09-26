from core.uuid import UUID


class Resource:

    def __init__(
        self,
        name: str
    ):

        self.uuid = UUID()

        self.name = name

    def delete(self):
        """
        Override for resources that own external resources,
        such as OpenGL objects.
        """
        pass