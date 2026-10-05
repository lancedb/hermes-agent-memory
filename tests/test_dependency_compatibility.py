from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    ("package", "tested_version", "next_major"),
    [
        ("openai", "2.24.0", "3.0.0"),
        ("requests", "2.33.0", "3.0.0"),
        ("botocore", "1.42.97", "2.0.0"),
    ],
)
def test_dependencies_accept_tested_versions_without_next_major(package, tested_version, next_major):
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    requirements = {req.name: req for req in map(Requirement, project["dependencies"])}
    assert tested_version in requirements[package].specifier
    assert next_major not in requirements[package].specifier
