import ctypes

from OpenGL.GL import GL_FLOAT

from core.assertions import engine_assert


class VertexAttribute:

    def __init__(
        self,
        count,
        gl_type,
        normalized=False
    ):

        self.count = count
        self.gl_type = gl_type
        self.normalized = normalized

        self.offset = 0

    @property
    def size(self):

        return (
            self.count
            * get_gl_type_size(
                self.gl_type
            )
        )


class VertexLayout:

    def __init__(self):

        self.attributes = []
        self.stride = 0

    def add(
        self,
        count,
        gl_type=GL_FLOAT,
        normalized=False
    ):

        engine_assert(
            1 <= count <= 4,
            (
                "Vertex attribute component count "
                "must be between 1 and 4."
            )
        )

        engine_assert(
            gl_type == GL_FLOAT,
            (
                "Current VertexBuffer implementation "
                "only supports GL_FLOAT attributes."
            )
        )

        attribute = VertexAttribute(
            count,
            gl_type,
            normalized
        )

        attribute.offset = (
            self.stride
        )

        self.attributes.append(
            attribute
        )

        self.stride += (
            attribute.size
        )

        return self


def get_gl_type_size(
    gl_type
):

    if gl_type == GL_FLOAT:

        return ctypes.sizeof(
            ctypes.c_float
        )

    raise ValueError(
        f"Unsupported OpenGL type: {gl_type}"
    )