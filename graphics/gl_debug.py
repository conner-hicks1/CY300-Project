import ctypes

from OpenGL.GL import (
    GLDEBUGPROC,
    GL_DEBUG_OUTPUT,
    GL_DEBUG_OUTPUT_SYNCHRONOUS,
    GL_DEBUG_SEVERITY_HIGH,
    GL_DEBUG_SEVERITY_LOW,
    GL_DEBUG_SEVERITY_MEDIUM,
    GL_DEBUG_SEVERITY_NOTIFICATION,
    GL_DEBUG_TYPE_ERROR,
    glDebugMessageCallback,
    glEnable,
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

# =========================================================
# Debug Output (OpenGL 4.3+)
# =========================================================
#
# With a debug context the driver reports errors, invalid
# usage and performance warnings through a callback, at the
# call that caused them, instead of us polling glGetError.

_SEVERITY_NAMES = {
    GL_DEBUG_SEVERITY_HIGH: "high",
    GL_DEBUG_SEVERITY_MEDIUM: "medium",
    GL_DEBUG_SEVERITY_LOW: "low",
}

# The callback object must outlive the context, or ctypes
# frees it while the driver still calls it.
_callback = None


def install_debug_callback() -> bool:
    """
    Route driver messages to the engine log. Returns False
    if the context has no debug output.
    """

    global _callback

    from core.logger import Logger

    def on_message(
        source,
        message_type,
        message_id,
        severity,
        length,
        message,
        user_param
    ):

        # Notifications (buffer placement hints etc.) are
        # noise.
        if severity == GL_DEBUG_SEVERITY_NOTIFICATION:
            return

        text = (
            message[:length].decode("utf-8", errors="replace")
            if isinstance(message, bytes)
            else ctypes.string_at(message, length).decode("utf-8", errors="replace")
        )

        log = (
            Logger.error
            if message_type == GL_DEBUG_TYPE_ERROR
            or severity == GL_DEBUG_SEVERITY_HIGH
            else Logger.warning
        )

        log(
            "[OpenGL] (%s, id %d) %s",
            _SEVERITY_NAMES.get(severity, "?"),
            int(message_id),
            text.strip()
        )

    try:

        _callback = GLDEBUGPROC(on_message)

        glEnable(GL_DEBUG_OUTPUT)

        # Report on the offending call's thread, so the log
        # order matches the code.
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS)

        glDebugMessageCallback(_callback, None)

    except Exception:

        Logger.exception(
            "[OpenGL] Debug output unavailable."
        )

        _callback = None

        return False

    Logger.info(
        "[OpenGL] Debug output enabled."
    )

    return True
