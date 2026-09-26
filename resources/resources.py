from core.logger import Logger

from graphics.material import Material
from graphics.mesh import Mesh
from graphics.shader import Shader
from graphics.texture import Texture2D

from resources.resource_manager import ResourceManager


class Resources:

    # =====================================================
    # Construction
    # =====================================================

    def __init__(self):

        Logger.info(
            "[Resources] Initializing."
        )

        self._shutdown = False

        # -------------------------------------------------
        # Resource Managers
        # -------------------------------------------------

        self.shaders = ResourceManager[Shader](
            "Shaders"
        )

        self.textures = ResourceManager[Texture2D](
            "Textures"
        )

        self.meshes = ResourceManager[Mesh](
            "Meshes"
        )

        self.materials = ResourceManager[Material](
            "Materials"
        )

        Logger.info(
            "[Resources] Initialized."
        )

    # =====================================================
    # Shader Hot Reload
    # =====================================================

    def reload_changed_shaders(
        self,
        force: bool = False
    ) -> int:
        """
        Rebuild shaders whose source (or any #include) has
        changed on disk. A shader that fails to compile
        keeps its previous program. Handles stay valid
        because the Shader object is updated in place.

        Returns the number of shaders reloaded.
        """

        reloaded = 0

        for key, shader in self.shaders.items():

            if force or shader.has_changed_on_disk():

                Logger.info(
                    "[Resources] Reloading shader '%s'.",
                    key
                )

                if shader.reload():
                    reloaded += 1

        return reloaded

    # =====================================================
    # Shutdown
    # =====================================================

    def shutdown(self):

        if self._shutdown:
            return

        Logger.info(
            "[Resources] Shutting down."
        )

        # Materials contain handles to shaders/textures,
        # so materials must disappear before those resources.

        self.materials.clear()

        self.meshes.clear()
        self.textures.clear()
        self.shaders.clear()

        self._shutdown = True

        Logger.info(
            "[Resources] Shutdown complete."
        )

    # =====================================================
    # State
    # =====================================================

    @property
    def is_shutdown(self) -> bool:

        return self._shutdown