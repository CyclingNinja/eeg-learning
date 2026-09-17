"""Shared fixtures for the tools test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def make_tree(tmp_path):
    """Return a builder that materialises empty files under ``tmp_path``.

    Call ``make_tree([...])`` with POSIX-style relative paths; parent directories
    are created as needed and the created ``Path`` objects are returned. Request
    the ``tmp_path`` fixture alongside it when you need the common root to hand to
    the function under test.

    Contents are irrelevant to the path helpers, which only inspect names,
    suffixes, and directory structure.
    """

    def _make(rel_paths):
        created = []
        for rel in rel_paths:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            created.append(p)
        return created

    return _make
