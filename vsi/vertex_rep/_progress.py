"""
Progress reporting for per-state VSI computations.

Defines a tiny :class:`ProgressSink` protocol that the engine's
``_compute_value_set`` methods speak, plus two implementations:

* :class:`NullSink` — silent default; used in tests and when no UI is
  desired.
* :class:`RichPerStateProgress` — context manager that owns a
  :class:`rich.progress.Progress` bar with one task per state and
  hands out a :class:`ProgressSink` for each via :meth:`sink_for`.

All Rich-specific code lives here so the algorithm modules stay UI-free.
"""
from __future__ import annotations

import threading
from typing import Protocol

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeRemainingColumn,
)


class ProgressSink(Protocol):
    """A sink for per-state progress updates from a compute method."""

    def set_total(self, n: int) -> None:
        """Set/reset the total count for the current phase."""
        ...

    def advance(self) -> None:
        """Advance the current phase by one unit."""
        ...

    def set_label(self, label: str) -> None:
        """Update the human-readable label for the current task."""
        ...

    def finish(self, n_vertices: int) -> None:
        """Mark the task as complete, recording the vertex count."""
        ...


class NullSink:
    """No-op :class:`ProgressSink` for silent execution."""

    def set_total(self, n: int) -> None:
        pass

    def advance(self) -> None:
        pass

    def set_label(self, label: str) -> None:
        pass

    def finish(self, n_vertices: int) -> None:
        pass


class _RichProgressSink:
    """Per-state sink backed by a shared :class:`rich.progress.Progress`."""

    def __init__(
        self,
        progress: Progress,
        task_id: int,
        state: int,
        lock: threading.Lock,
    ) -> None:
        self._progress = progress
        self._task_id = task_id
        self._state = state
        self._lock = lock

    def set_total(self, n: int) -> None:
        with self._lock:
            self._progress.reset(self._task_id, total=n, completed=0)

    def advance(self) -> None:
        with self._lock:
            self._progress.advance(self._task_id)

    def set_label(self, label: str) -> None:
        with self._lock:
            self._progress.update(
                self._task_id,
                description=f"State {self._state} {label}",
            )

    def finish(self, n_vertices: int) -> None:
        with self._lock:
            task = next(
                task for task in self._progress.tasks
                if task.id == self._task_id
            )
            total = 1 if task.total is None else task.total
            self._progress.start_task(self._task_id)
            self._progress.update(
                self._task_id,
                total=total, completed=total,
                description=(
                    f"State {self._state} [green]done[/] "
                    f"({n_vertices} points)"
                ),
            )
            self._progress.stop_task(self._task_id)


class RichPerStateProgress:
    """
    Context manager that displays a per-state Rich progress bar.

    Use as::

        with RichPerStateProgress(num_states, console) as ui:
            sink = ui.sink_for(s0)
            points = compute_fn(s0, value_sets, sink)
            sink.finish(len(points))

    With ``enabled=False``, no display is created and :meth:`sink_for`
    returns a :class:`NullSink`.
    """

    def __init__(
        self, num_states: int, console: Console | None = None,
        *, enabled: bool = True,
    ) -> None:
        self._num_states = num_states
        self._console = console
        self._enabled = enabled
        self._lock = threading.Lock()
        self._progress: Progress | None = None
        self._tasks: dict[int, int] = {}

    def __enter__(self) -> "RichPerStateProgress":
        if not self._enabled:
            return self
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._progress.__enter__()
        self._tasks = {
            s: self._progress.add_task(f"State {s}", total=None, start=False)
            for s in range(self._num_states)
        }
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled:
            assert self._progress is not None
            self._progress.__exit__(exc_type, exc, tb)

    def sink_for(self, state: int) -> ProgressSink:
        if not self._enabled:
            return NullSink()
        assert self._progress is not None, (
            "RichPerStateProgress.sink_for called outside the context manager."
        )
        return _RichProgressSink(
            self._progress, self._tasks[state], state, self._lock,
        )
