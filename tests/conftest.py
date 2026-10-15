"""Shared pytest fixtures.

``requires_core``: tests that exercise owner-written CORE functions request this fixture.
If any CORE stub raises ``NotImplementedError`` during setup or the test body, the test
is reported as *skipped* (naming the stub) instead of failed. Once the CORE function is
implemented, the same test runs unchanged.
"""

import types

import numpy as np
import pytest


def _raising_function(tb: types.TracebackType | None) -> str:
    """Qualified name of the function that raised, from the innermost traceback frame."""
    if tb is None:
        return "<unknown>"
    while tb.tb_next is not None:
        tb = tb.tb_next
    frame = tb.tb_frame
    return f"{frame.f_globals.get('__name__', '?')}.{frame.f_code.co_qualname}"


def _maybe_skip(item: pytest.Item, exc: NotImplementedError) -> None:
    if "requires_core" in getattr(item, "fixturenames", ()):
        pytest.skip(f"CORE not implemented: {_raising_function(exc.__traceback__)}")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    try:
        return (yield)
    except NotImplementedError as exc:
        _maybe_skip(item, exc)
        raise


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    try:
        return (yield)
    except NotImplementedError as exc:
        _maybe_skip(item, exc)
        raise


@pytest.fixture
def requires_core() -> None:
    """Mark a test as exercising CORE code: ``NotImplementedError`` turns into a skip."""


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(20261015)
