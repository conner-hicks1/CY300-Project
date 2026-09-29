from dataclasses import dataclass

from core.assertions import engine_assert


# =========================================================
# Undo History
# =========================================================
#
# Snapshot-based undo: after each completed edit the
# editor stores the whole serialized scene. Scenes here
# are small (tens of entities, a few KB of JSON), so full
# snapshots are simpler and far less error-prone than
# per-command inverse operations, and every kind of edit
# (inspector, gizmo, hierarchy, add/remove) gets undo for
# free.
#
#     history.reset(snapshot)        # after load / new
#     history.push(snapshot)         # after each edit
#     snapshot = history.undo()      # None if nothing to undo
#     snapshot = history.redo()


@dataclass(frozen=True, slots=True)
class Snapshot:

    # Serialized scene (JSON text).
    scene: str

    # File id of the selected entity, if any, so selection
    # survives undo/redo (entities are recreated).
    selected_id: int | None = None

    # Shown in Edit > Undo "<label>".
    label: str = ""


class UndoHistory:

    DEFAULT_LIMIT = 100

    def __init__(
        self,
        limit: int = DEFAULT_LIMIT
    ):

        engine_assert(
            limit >= 1,
            "Undo history limit must be at least 1."
        )

        self._limit = limit

        # _states[_index] is the current state; earlier
        # entries are undo targets, later ones redo.

        self._states: list[Snapshot] = []
        self._index = -1

    # =====================================================
    # Recording
    # =====================================================

    def reset(
        self,
        snapshot: Snapshot
    ):
        """Forget all history; `snapshot` is the new baseline."""

        self._states = [snapshot]
        self._index = 0

    def push(
        self,
        snapshot: Snapshot
    ) -> bool:
        """
        Record a new current state. Discards any redo
        states. Returns False (and records nothing) if the
        scene did not actually change.
        """

        engine_assert(
            self._index >= 0,
            "UndoHistory.push() before reset()."
        )

        if snapshot.scene == self._states[self._index].scene:
            return False

        del self._states[self._index + 1:]

        self._states.append(
            snapshot
        )

        # Drop the oldest states beyond the limit (the
        # current state is always kept).

        overflow = len(self._states) - (self._limit + 1)

        if overflow > 0:
            del self._states[:overflow]

        self._index = len(self._states) - 1

        return True

    # =====================================================
    # Navigation
    # =====================================================

    @property
    def can_undo(
        self
    ) -> bool:

        return self._index > 0

    @property
    def can_redo(
        self
    ) -> bool:

        return 0 <= self._index < len(self._states) - 1

    @property
    def undo_label(
        self
    ) -> str:

        return (
            self._states[self._index].label
            if self.can_undo
            else ""
        )

    @property
    def redo_label(
        self
    ) -> str:

        return (
            self._states[self._index + 1].label
            if self.can_redo
            else ""
        )

    def undo(
        self
    ) -> Snapshot | None:

        if not self.can_undo:
            return None

        self._index -= 1

        return self._states[self._index]

    def redo(
        self
    ) -> Snapshot | None:

        if not self.can_redo:
            return None

        self._index += 1

        return self._states[self._index]

    @property
    def current(
        self
    ) -> Snapshot | None:

        return (
            self._states[self._index]
            if self._index >= 0
            else None
        )
