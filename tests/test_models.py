import pytest
from pydantic import ValidationError

from flight_recorder.models import CommandSucceeds, FileExists, TaskSpec

TASK_YAML = """\
id: sample
prompt: Do the thing.
assertions:
  - kind: file_exists
    path: out.txt
  - kind: command_succeeds
    command: grep -q done out.txt
timeout_s: 120
"""


def test_task_from_yaml(tmp_path):
    task_file = tmp_path / "task.yaml"
    task_file.write_text(TASK_YAML)
    task = TaskSpec.from_yaml(task_file)
    assert task.id == "sample"
    assert task.timeout_s == 120
    assert isinstance(task.assertions[0], FileExists)
    assert isinstance(task.assertions[1], CommandSucceeds)


def test_defaults():
    task = TaskSpec(id="t", prompt="p")
    assert task.template == "base"
    assert task.assertions == []


def test_unknown_assertion_kind_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {"id": "t", "prompt": "p", "assertions": [{"kind": "nonsense"}]}
        )
