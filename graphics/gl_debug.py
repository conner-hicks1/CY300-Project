from OpenGL.GL import (
    glGetError,

    GL_NO_ERROR,
    GL_INVALID_ENUM,
    GL_INVALID_VALUE,
    GL_INVALID_OPERATION,
    GL_INVALID_FRAMEBUFFER_OPERATION,
    GL_OUT_OF_MEMORY
)

from core.exceptions import GraphicsError


_GL_ERROR_NAMES = {

    GL_INVALID_ENUM:
        "GL_INVALID_ENUM",

    GL_INVALID_VALUE:
        "GL_INVALID_VALUE",

    GL_INVALID_OPERATION:
        "GL_INVALID_OPERATION",

    GL_INVALID_FRAMEBUFFER_OPERATION:
        "GL_INVALID_FRAMEBUFFER_OPERATION",

    GL_OUT_OF_MEMORY:
        "GL_OUT_OF_MEMORY"
}


def check_gl_error(
    operation=""
):

    errors = []

    while True:

        error = glGetError()

        if error == GL_NO_ERROR:
            break

        errors.append(
            _GL_ERROR_NAMES.get(
                error,
                f"UNKNOWN_GL_ERROR({error})"
            )
        )

    if not errors:
        return

    description = ", ".join(
        errors
    )

    if operation:

        raise GraphicsError(
            f"OpenGL error after "
            f"'{operation}': "
            f"{description}"
        )

    raise GraphicsError(
        f"OpenGL error: {description}"
    )