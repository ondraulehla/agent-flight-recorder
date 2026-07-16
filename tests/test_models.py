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
setup_files:
  - path: data.csv
    content: "a,b\\n1,2\\n"
setup_commands:
  - md5sum data.csv > .checksum
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
    assert task.setup_files[0].path == "data.csv"
    assert task.setup_files[0].content == "a,b\n1,2\n"
    assert task.setup_commands == ["md5sum data.csv > .checksum"]


def test_defaults():
    task = TaskSpec(id="t", prompt="p")
    assert task.template == "base"
    assert task.assertions == []


def test_unknown_assertion_kind_rejected():
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {"id": "t", "prompt": "p", "assertions": [{"kind": "nonsense"}]}
        )
