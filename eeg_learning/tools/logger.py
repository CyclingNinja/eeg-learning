"""Package-wide logging.

Progress reporting in this stack is currently a mix of bare ``print`` calls
(io/, tools/) and ad-hoc ``logging.basicConfig`` (pipeline/decision.py). This
module gives both a single entry point: handlers are attached once to the
``eeg_learning`` logger and every module logs through a child of it.

Typical module use::

    from eeg_learning.tools.logger import get_logger

    log = get_logger(__name__)
    log.info("loaded %d recordings", n)

Entry points (CLI, DVC stages, API backends) configure the destination once::

    from eeg_learning.tools.logger import Logger

    Logger.configure(level="DEBUG", log_file="target/run.log")

The module is named ``logger.py`` rather than ``logging.py`` on purpose: the
latter shadows the stdlib module for anything importing it by path, and ruff
flags it (A005).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import IO

__all__ = ["Logger", "get_logger"]

#: Logger every other logger in the package hangs off.
ROOT_NAME = "eeg_learning"

DEFAULT_LEVEL = "INFO"
DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Handlers this module owns are tagged so repeat configuration replaces them
# instead of stacking duplicates (a real hazard: DVC stages, the CLI and the
# API all configure logging, and a run may hit more than one of them).
_HANDLER_KEY = "_eeg_win_stack_key"
_CONSOLE_KEY = "console"


def _coerce_level(level: str | int) -> int:
    """Turn ``"INFO"`` / ``logging.INFO`` into a numeric level."""
    if isinstance(level, int):
        return level
    resolved = logging.getLevelName(str(level).upper())
    if not isinstance(resolved, int):
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def _qualified_name(name: str | None) -> str:
    """Qualify ``name`` so it is always a child of :data:`ROOT_NAME`."""
    if not name or name == ROOT_NAME:
        return ROOT_NAME
    if name.startswith(f"{ROOT_NAME}."):
        return name
    return f"{ROOT_NAME}.{name}"


class _StderrHandler(logging.StreamHandler):
    """Console handler that resolves ``sys.stderr`` at emit time.

    A plain :class:`logging.StreamHandler` captures the stream object when it
    is built. Modules call :func:`get_logger` at import time, so a captured
    stream would pin every message to whatever ``sys.stderr`` happened to be
    during import — breaking pytest's ``capsys`` and any caller that redirects
    stderr afterwards (DVC stage capture, notebook contexts).
    """

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        # StreamHandler.__init__ and setStream() assign here; the whole point
        # of this subclass is that the destination is looked up, not stored.
        pass


def _owned_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [handler for handler in logger.handlers if hasattr(handler, _HANDLER_KEY)]


def _install(logger: logging.Logger, key: str, factory, formatter: logging.Formatter) -> logging.Handler:
    """Add a tagged handler, or refresh the formatter of the existing one."""
    for handler in _owned_handlers(logger):
        if getattr(handler, _HANDLER_KEY) == key:
            handler.setFormatter(formatter)
            return handler
    handler = factory()
    setattr(handler, _HANDLER_KEY, key)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler


class Logger:
    """A named logger within the ``eeg_learning`` hierarchy.

    Instances are thin wrappers around :class:`logging.Logger`; the handlers
    live on the shared package root, so configuration set by one instance (or
    by :meth:`configure`) applies to every module in the package.

    Parameters
    ----------
    name : str, optional
        Logger name, usually ``__name__``. Names outside the package are
        re-parented under ``eeg_learning``. Defaults to the package root.
    level : str or int, optional
        Threshold for the package root, e.g. ``"DEBUG"`` or ``logging.DEBUG``.
        When omitted an unconfigured package defaults to ``INFO`` and an
        already-configured one is left alone.
    log_file : str or pathlib.Path, optional
        File to append to, in addition to the console. Parent directories are
        created. Passing it re-configures the package root.
    console : bool, default True
        Whether to emit to ``stream``.
    stream : IO, optional
        Console destination, ``sys.stderr`` by default, so log output never
        contaminates piped stdout (metrics JSON, CSV rows).
    fmt, datefmt : str
        :class:`logging.Formatter` arguments.

    Examples
    --------
    >>> log = Logger(__name__)
    >>> log.info("windowing %d recordings", 12)  # doctest: +SKIP
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        level: str | int | None = None,
        log_file: str | Path | None = None,
        console: bool = True,
        stream: IO[str] | None = None,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATEFMT,
    ) -> None:
        self._logger = logging.getLogger(_qualified_name(name))
        root = logging.getLogger(ROOT_NAME)
        unconfigured = not _owned_handlers(root)
        if unconfigured or level is not None or log_file is not None:
            if level is None:
                level = DEFAULT_LEVEL if unconfigured else root.level
            self.configure(
                level=level,
                log_file=log_file,
                console=console,
                stream=stream,
                fmt=fmt,
                datefmt=datefmt,
            )

    # -- configuration ----------------------------------------------------

    @classmethod
    def configure(
        cls,
        level: str | int = DEFAULT_LEVEL,
        log_file: str | Path | None = None,
        console: bool = True,
        stream: IO[str] | None = None,
        fmt: str = DEFAULT_FORMAT,
        datefmt: str = DEFAULT_DATEFMT,
    ) -> logging.Logger:
        """Configure the package root logger and return it.

        Safe to call repeatedly: handlers are keyed by destination, so a second
        call with the same console/file does not duplicate output.
        """
        root = logging.getLogger(ROOT_NAME)
        root.setLevel(_coerce_level(level))
        # Package output is handled here; leave the interpreter root (and
        # anything third-party attached to it, e.g. mlflow) out of it.
        root.propagate = False

        formatter = logging.Formatter(fmt, datefmt)
        if console:
            # No explicit stream => _StderrHandler, which looks sys.stderr up at
            # emit time instead of pinning the object captured during import.
            factory = (lambda: logging.StreamHandler(stream)) if stream is not None else _StderrHandler
            _install(root, _CONSOLE_KEY, factory, formatter)
        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            _install(root, f"file:{path.resolve()}", lambda: logging.FileHandler(path, encoding="utf-8"), formatter)
        return root

    @classmethod
    def from_config(cls, config: dict, name: str | None = None) -> Logger:
        """Build a logger from a loaded ``params.toml``.

        Reads the optional ``[logging]`` section; an absent section, or an
        empty ``file`` value, falls back to console-only at ``INFO``.
        """
        section = config.get("logging", {}) if config else {}
        log_file = section.get("file") or None
        return cls(
            name,
            level=section.get("level", DEFAULT_LEVEL),
            log_file=log_file,
            console=section.get("console", True),
            fmt=section.get("format", DEFAULT_FORMAT),
            datefmt=section.get("datefmt", DEFAULT_DATEFMT),
        )

    @classmethod
    def reset(cls) -> None:
        """Detach and close every handler this module installed.

        Mainly for tests and for long-lived processes that reconfigure logging
        between jobs; handlers owned by other libraries are left untouched.
        """
        root = logging.getLogger(ROOT_NAME)
        for handler in _owned_handlers(root):
            root.removeHandler(handler)
            handler.close()

    def set_level(self, level: str | int) -> None:
        """Set the threshold of this logger only (not the package root)."""
        self._logger.setLevel(_coerce_level(level))

    def add_file_handler(self, path: str | Path) -> None:
        """Also append package output to ``path``."""
        self.configure(level=logging.getLogger(ROOT_NAME).level, log_file=path)

    # -- accessors --------------------------------------------------------

    @property
    def name(self) -> str:
        return self._logger.name

    @property
    def logger(self) -> logging.Logger:
        """The wrapped stdlib logger, for handing to third-party APIs."""
        return self._logger

    def child(self, suffix: str) -> Logger:
        """Return a logger one level below this one, e.g. ``log.child("io")``."""
        return type(self)(f"{self._logger.name}.{suffix}")

    def is_enabled_for(self, level: str | int) -> bool:
        """Whether ``level`` would be emitted (guard for expensive messages)."""
        return self._logger.isEnabledFor(_coerce_level(level))

    # -- emitting ---------------------------------------------------------

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """Log an error with the active traceback; call from an except block."""
        self._logger.exception(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def log(self, level: str | int, msg: str, *args, **kwargs) -> None:
        self._logger.log(_coerce_level(level), msg, *args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._logger.name!r}, level={logging.getLevelName(self._logger.getEffectiveLevel())})"


def get_logger(name: str | None = None) -> Logger:
    """Return a :class:`Logger` for ``name``, configuring defaults if needed.

    The module-level convenience form of ``Logger(name)``: call it at import
    time in any module with ``get_logger(__name__)``.
    """
    return Logger(name)
