import pytest

from core.exceptions import ShaderError
from graphics.shader_preprocessor import preprocess_shader


def write(path, text):

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

    return path


def test_defines_are_injected_after_version(tmp_path):

    root = write(tmp_path / "a.glsl", "#version 330 core\nvoid main() {}\n")

    result = preprocess_shader(root, {"MAX_LIGHTS": 8, "SCALE": 2.0, "FLAG": True})

    lines = result.source.split("\n")

    assert lines[0] == "#version 330 core"
    assert lines[1:4] == ["#define MAX_LIGHTS 8", "#define SCALE 2.0", "#define FLAG 1"]
    assert lines[4] == "#line 2 0"


def test_include_is_expanded_with_line_directives(tmp_path):

    write(tmp_path / "include" / "common.glsl", "float helper() { return 1.0; }")

    root = write(
        tmp_path / "main.glsl",
        '#version 330 core\n#include "include/common.glsl"\nvoid main() {}\n'
    )

    result = preprocess_shader(root)

    assert "float helper()" in result.source
    assert "#line 1 1" in result.source
    assert "#line 3 0" in result.source
    assert [p.name for p in result.files] == ["main.glsl", "common.glsl"]


def test_each_file_is_included_once(tmp_path):

    write(tmp_path / "b.glsl", '#include "a.glsl"\nfloat b;')
    write(tmp_path / "a.glsl", '#include "b.glsl"\nfloat a;')

    root = write(
        tmp_path / "main.glsl",
        '#version 330 core\n#include "a.glsl"\n#include "b.glsl"\n'
    )

    result = preprocess_shader(root)

    # Mutual includes neither loop nor duplicate.
    assert result.source.count("float a;") == 1
    assert result.source.count("float b;") == 1


def test_missing_include_names_the_including_file(tmp_path):

    root = write(tmp_path / "main.glsl", '#version 330 core\n#include "nope.glsl"\n')

    with pytest.raises(ShaderError, match=r"main\.glsl:2"):
        preprocess_shader(root)


def test_missing_version_is_rejected(tmp_path):

    root = write(tmp_path / "main.glsl", "void main() {}\n")

    with pytest.raises(ShaderError, match="#version"):
        preprocess_shader(root)


def test_included_version_is_rejected(tmp_path):

    write(tmp_path / "inc.glsl", "#version 330 core\n")

    root = write(tmp_path / "main.glsl", '#version 330 core\n#include "inc.glsl"\n')

    with pytest.raises(ShaderError, match="cannot contain"):
        preprocess_shader(root)


def test_invalid_define_name(tmp_path):

    root = write(tmp_path / "main.glsl", "#version 330 core\n")

    with pytest.raises(ShaderError):
        preprocess_shader(root, {"1BAD": 1})


def test_engine_shaders_preprocess(tmp_path):

    # Every shader shipped with the engine resolves its includes.
    from pathlib import Path

    from graphics.uniform_blocks import ENGINE_SHADER_DEFINES

    shaders = Path(__file__).resolve().parent.parent / "assets" / "shaders"

    roots = [p for p in shaders.glob("*.glsl")]

    assert roots

    for root in roots:
        preprocess_shader(root, ENGINE_SHADER_DEFINES)
