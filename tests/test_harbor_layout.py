from pathlib import Path

from harness_rl.tasks.harbor_layout import (
    convert_task,
    leaked_paths,
    without_prebuilt_image,
)

TASK_TOML = """[environment]
docker_image = "harbor-swe-base:2"
cpus = 1
"""


def make_task(root: Path) -> Path:
    (root / "environment" / "workdir" / "src").mkdir(parents=True)
    (root / "environment" / "workdir" / "src" / "a.c").write_text("int a;")
    (root / "environment" / "setup.sh").write_text("true")
    (root / "solution").mkdir()
    (root / "solution" / "edits.json").write_text("[]")
    (root / "tests").mkdir()
    (root / "tests" / "test.sh").write_text("exit 0")
    (root / "instruction.md").write_text("fix it")
    (root / "task.toml").write_text(TASK_TOML)
    return root


def test_prebuilt_image_line_is_removed() -> None:
    assert without_prebuilt_image(TASK_TOML) == "[environment]\ncpus = 1\n"


def test_answers_and_hidden_tests_count_as_leaks() -> None:
    paths = ["workdir/src/a.c", "solution/edits.json", "tests/h0/x.c"]
    assert leaked_paths(paths) == ["solution/edits.json", "tests/h0/x.c"]


def test_converted_task_keeps_solution_out_of_build_context(tmp_path: Path) -> None:
    target = tmp_path / "out"
    convert_task(make_task(tmp_path / "src"), target)
    environment_files = {
        p.relative_to(target / "environment").as_posix()
        for p in (target / "environment").rglob("*")
        if p.is_file()
    }
    assert environment_files == {"Dockerfile", "setup.sh", "workdir/src/a.c"}
    assert (target / "solution" / "edits.json").exists()  # oracle still available
    assert "docker_image" not in (target / "task.toml").read_text()
