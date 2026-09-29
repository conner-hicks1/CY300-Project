from typing import (
    Callable,
    Generic,
    TypeVar
)

from core.assertions import engine_assert
from core.handle import Handle
from core.handle_manager import HandleManager
from core.logger import Logger


T = TypeVar("T")


class ResourceManager(
    Generic[T]
):

    def __init__(
        self,
        name="ResourceManager"
    ):

        self._name = name

        self._handles = HandleManager[T](
            name
        )

        self._key_to_handle: dict[
            str,
            Handle
        ] = {}

        self._handle_to_key: dict[
            Handle,
            str
        ] = {}

        Logger.debug(
            "[%s] Resource manager created.",
            self._name
        )

    # =====================================================
    # Loading
    # =====================================================

    def load(
        self,
        key: str,
        factory: Callable[[], T]
    ) -> Handle:

        normalized_key = self._normalize_key(
            key
        )

        engine_assert(
            bool(normalized_key),
            (
                f"{self._name} resource key "
                f"cannot be empty."
            )
        )

        engine_assert(
            callable(factory),
            (
                f"{self._name} resource factory "
                f"must be callable."
            )
        )

        # -------------------------------------------------
        # Existing Resource
        # -------------------------------------------------

        existing = self._key_to_handle.get(
            normalized_key
        )

        if existing is not None:

            if self._handles.is_valid(
                existing
            ):

                Logger.debug(
                    "[%s] Reusing '%s'.",
                    self._name,
                    normalized_key
                )

                return existing

            # A stale mapping should not normally occur,
            # but remove it defensively.

            self._key_to_handle.pop(
                normalized_key,
                None
            )

            self._handle_to_key.pop(
                existing,
                None
            )

        # -------------------------------------------------
        # Create
        # -------------------------------------------------

        Logger.debug(
            "[%s] Loading '%s'.",
            self._name,
            normalized_key
        )

        resource = factory()

        engine_assert(
            resource is not None,
            (
                f"{self._name} factory returned None "
                f"for '{normalized_key}'."
            )
        )

        handle = self._handles.create(
            resource
        )

        self._key_to_handle[
            normalized_key
        ] = handle

        self._handle_to_key[
            handle
        ] = normalized_key

        Logger.info(
            "[%s] Loaded '%s' as %s.",
            self._name,
            normalized_key,
            handle
        )

        return handle

    # =====================================================
    # Add Existing
    # =====================================================

    def add(
        self,
        key: str,
        resource: T
    ) -> Handle:

        engine_assert(
            resource is not None,
            (
                f"{self._name} cannot add "
                f"a None resource."
            )
        )

        normalized_key = self._normalize_key(
            key
        )

        engine_assert(
            not self.contains(normalized_key),
            (
                f"{self._name} already contains "
                f"resource '{normalized_key}'."
            )
        )

        handle = self._handles.create(
            resource
        )

        self._key_to_handle[
            normalized_key
        ] = handle

        self._handle_to_key[
            handle
        ] = normalized_key

        Logger.info(
            "[%s] Added '%s' as %s.",
            self._name,
            normalized_key,
            handle
        )

        return handle

    # =====================================================
    # Access by Handle
    # =====================================================

    def get(
        self,
        handle: Handle
    ) -> T:

        return self._handles.get(
            handle
        )

    def try_get(
        self,
        handle: Handle
    ) -> T | None:

        return self._handles.try_get(
            handle
        )

    # =====================================================
    # Access by Key
    # =====================================================

    def get_by_key(
        self,
        key: str
    ) -> T:

        return self.get(
            self.get_handle(key)
        )

    def try_get_by_key(
        self,
        key: str
    ) -> T | None:

        handle = self.try_get_handle(
            key
        )

        if handle is None:
            return None

        return self.try_get(
            handle
        )

    # =====================================================
    # Handle Lookup
    # =====================================================

    def get_handle(
        self,
        key: str
    ) -> Handle:

        normalized_key = self._normalize_key(
            key
        )

        handle = self._key_to_handle.get(
            normalized_key
        )

        engine_assert(
            handle is not None,
            (
                f"{self._name} resource "
                f"'{normalized_key}' is not loaded."
            )
        )

        engine_assert(
            self._handles.is_valid(handle),
            (
                f"{self._name} contains a stale handle "
                f"for '{normalized_key}'."
            )
        )

        return handle

    def try_get_handle(
        self,
        key: str
    ) -> Handle | None:

        normalized_key = self._normalize_key(
            key
        )

        handle = self._key_to_handle.get(
            normalized_key
        )

        if handle is None:
            return None

        if not self._handles.is_valid(
            handle
        ):

            return None

        return handle

    # =====================================================
    # Iteration
    # =====================================================

    def items(
        self
    ) -> list[tuple[str, T]]:
        """
        Snapshot of (key, resource) for every loaded
        resource, sorted by key.
        """

        return [
            (key, self._handles.get(handle))
            for key, handle in sorted(
                self._key_to_handle.items()
            )
            if self._handles.is_valid(handle)
        ]

    def key_of(
        self,
        handle: Handle
    ) -> str | None:
        """Key a handle was loaded under (None if unknown)."""

        if not self._handles.is_valid(handle):
            return None

        return self._handle_to_key.get(
            handle
        )

    def handle_items(
        self
    ) -> list[tuple[str, Handle]]:

        return [
            (key, handle)
            for key, handle in sorted(
                self._key_to_handle.items()
            )
            if self._handles.is_valid(handle)
        ]

    # =====================================================
    # Queries
    # =====================================================

    def contains(
        self,
        key: str
    ) -> bool:

        return (
            self.try_get_handle(key)
            is not None
        )

    def is_valid(
        self,
        handle: Handle
    ) -> bool:

        return self._handles.is_valid(
            handle
        )

    # =====================================================
    # Unload
    # =====================================================

    def unload(
        self,
        handle: Handle
    ):

        engine_assert(
            self._handles.is_valid(handle),
            (
                f"{self._name} cannot unload "
                f"invalid or stale handle: {handle}"
            )
        )

        resource = self._handles.get(
            handle
        )

        key = self._handle_to_key.get(
            handle
        )

        # Destroy external/GPU state before invalidating
        # the handle.

        self._delete_resource(
            resource
        )

        if key is not None:

            self._key_to_handle.pop(
                key,
                None
            )

        self._handle_to_key.pop(
            handle,
            None
        )

        self._handles.destroy(
            handle
        )

        Logger.debug(
            "[%s] Unloaded '%s'.",
            self._name,
            (
                key
                if key is not None
                else "<unknown>"
            )
        )

    def unload_by_key(
        self,
        key: str
    ):

        self.unload(
            self.get_handle(key)
        )

    # =====================================================
    # Clear
    # =====================================================

    def clear(self):

        handles = list(
            self._handles.active_handles()
        )

        if handles:

            Logger.debug(
                "[%s] Releasing %d resources.",
                self._name,
                len(handles)
            )

        # Work from a snapshot because unload() modifies
        # manager state.

        for handle in handles:

            self.unload(
                handle
            )

        self._key_to_handle.clear()
        self._handle_to_key.clear()

    # =====================================================
    # Resource Destruction
    # =====================================================

    @staticmethod
    def _delete_resource(
        resource: T
    ):

        delete_method = getattr(
            resource,
            "delete",
            None
        )

        if callable(
            delete_method
        ):

            delete_method()

    # =====================================================
    # Key Normalization
    # =====================================================

    @staticmethod
    def _normalize_key(
        key: str
    ) -> str:

        engine_assert(
            isinstance(key, str),
            "Resource key must be a string."
        )

        return key.strip()

    # =====================================================
    # Information
    # =====================================================

    @property
    def count(self) -> int:

        return len(
            self._handles
        )

    @property
    def manager_id(self) -> int:

        return self._handles.manager_id

    def __len__(self):

        return self.count