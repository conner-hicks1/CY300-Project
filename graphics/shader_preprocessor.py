import re

from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import ShaderError


# =========================================================
# Shader Preprocessor
# =========================================================
#
# Adds two things GLSL lacks:
#
#   #include "relative/path.glsl"
#       Pasted in place. Paths resolve relative to the
#       including file. Each file is included at most once
#       per shader (implicit include guard, like
#       #pragma once), which also makes include cycles
#       harmless.
#
#   Injected #defines
#       Engine constants (e.g. MAX_POINT_LIGHTS) are
#       inserted right after #version, so Python and GLSL
#       share one source of truth.
#
# #line directives are emitted around includes so driver
# error messages point at the right file and line. Each
# file gets a source-string number; `files[n]` is the
# path for source number n.

_INCLUDE_PATTERN = re.compile(
    r'^\s*#\s*include\s+"([^"]+)"\s*$'
)

_VERSION_PATTERN = re.compile(
    r'^\s*#\s*version\b'
)

_DEFINE_NAME_PATTERN = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_]*$'
)


@dataclass(slots=True)
class PreprocessedShader:

    source: str

    # Source-string number -> file. Index 0 is the root.
    files: list[Path] = field(
        default_factory=list
    )

    def describe_files(
        self
    ) -> str:

        return "\n".join(
            f"  {index}: {path}"
            for index, path in enumerate(self.files)
        )


def preprocess_shader(
    path,
    defines: dict[str, object] | None = None
) -> PreprocessedShader:

    root = Path(
        path
    ).resolve()

    files: list[Path] = []

    body = _expand(
        root,
        files,
        include_site=None
    )

    # -----------------------------------------------------
    # Split off #version (must be first)
    # -----------------------------------------------------

    lines = body.split(
        "\n"
    )

    version_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip()
            and not line.strip().startswith("//")
        ),
        None
    )

    if (
        version_index is None
        or not _VERSION_PATTERN.match(lines[version_index])
    ):

        raise ShaderError(
            f"Shader must start with a #version directive: {root}"
        )

    header = lines[
        :version_index + 1
    ]

    rest = lines[
        version_index + 1:
    ]

    # -----------------------------------------------------
    # Injected Defines
    # -----------------------------------------------------

    define_lines = []

    for name, value in (defines or {}).items():

        if not _DEFINE_NAME_PATTERN.match(
            str(name)
        ):

            raise ShaderError(
                f"Invalid shader define name: {name!r}"
            )

        define_lines.append(
            f"#define {name} {_format_define_value(value)}"
        )

    # Restore the root file's line numbering after the
    # injected lines.

    define_lines.append(
        f"#line {version_index + 2} 0"
    )

    return PreprocessedShader(
        source="\n".join(
            header
            + define_lines
            + rest
        ),
        files=files
    )


# =========================================================
# Include Expansion
# =========================================================

def _expand(
    path: Path,
    files: list[Path],
    include_site: str | None
) -> str:

    try:

        text = path.read_text(
            encoding="utf-8"
        )

    except OSError as error:

        location = (
            f" (included from {include_site})"
            if include_site
            else ""
        )

        raise ShaderError(
            f"Failed to read shader '{path}'{location}: {error}"
        ) from error

    source_number = len(
        files
    )

    files.append(
        path
    )

    output: list[str] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        match = _INCLUDE_PATTERN.match(
            line
        )

        if match is None:

            output.append(
                line
            )

            continue

        included = (
            path.parent
            / match.group(1)
        ).resolve()

        # Implicit include guard.

        if included in files:

            output.append(
                f"// (already included: {match.group(1)})"
            )

            continue

        included_source = _expand(
            included,
            files,
            include_site=f"{path.name}:{line_number}"
        )

        if _has_version_directive(
            included_source
        ):

            raise ShaderError(
                "Included shader files cannot contain "
                f"#version: {included}"
            )

        included_number = files.index(
            included
        )

        output.append(
            f"#line 1 {included_number}"
        )

        output.append(
            included_source
        )

        output.append(
            f"#line {line_number + 1} {source_number}"
        )

    return "\n".join(
        output
    )


def _has_version_directive(
    source: str
) -> bool:

    return any(
        _VERSION_PATTERN.match(line)
        for line in source.split("\n")
    )


def _format_define_value(
    value
) -> str:

    if isinstance(
        value,
        bool
    ):

        return "1" if value else "0"

    if isinstance(
        value,
        float
    ):

        # GLSL needs a decimal point for float literals.

        text = repr(value)

        return (
            text
            if any(c in text for c in ".eE")
            else f"{text}.0"
        )

    return str(
        value
    )
