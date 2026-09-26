class EngineError(Exception):
    """Base class for engine runtime errors."""
    pass


class WindowError(EngineError):
    """Window or graphics-context failure."""
    pass


class GraphicsError(EngineError):
    """Graphics subsystem failure."""
    pass


class ShaderError(GraphicsError):
    """Shader loading, compilation, linking, or usage failure."""
    pass


class ResourceError(EngineError):
    """Resource loading or resource lifetime failure."""
    pass


class ConfigurationError(EngineError):
    """Invalid engine configuration."""
    pass