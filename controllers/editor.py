from ecs.components import BodyComponent

from game.part_editor import LaunchBody

from vessel.design import DesignDraft


# =========================================================
# The Application: Editor
# =========================================================
#
# The part editor: building on the pad, launching,
# cancelling back to what was flown.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

class EditorController:

    def start_editing(self):
        """Build on the launch pad: the current design, there, to edit."""

        launch = self.vessel_system.launch_body

        if launch is None:
            return

        body = self.scene.try_get_component(launch, BodyComponent)

        self._design_before_edit = self.vessel_design
        self._edited_height = 0.0

        found = self.vessel_system.find(self.scene)

        self._flying_before_edit = found[0] if found is not None and found[1].paid else None

        self.free_camera = False
        self.map_view.open = False
        self.time_warp.stop()

        self.part_editor.start(
            DesignDraft.from_design(self.vessel_design),
            LaunchBody(
                self.vessel_system.body_name(self.scene, launch),
                body.surface_pressure_bar if body is not None else 0.0,
                body.surface_gravity if body is not None else 9.80665
            )
        )

    def _update_editor(self):
        """The rocket on the pad follows the draft; the selected part lights up."""

        editor = self.part_editor

        if not editor.changed:
            return

        editor.changed = False

        draft = editor.draft

        self.vessel_system.rebuild(
            self.scene,
            draft.preview() if draft.entries else None,
            self.orbit_system
        )

        self.vessel_system.highlight(self.scene, editor.selected)

        found = self.vessel_system.find(self.scene)

        height = found[1].vessel.height if found is not None else 0.0

        # The whole stack in view, refitted when it grows or
        # shrinks (a zoom of your own stays otherwise).
        if abs(height - self._edited_height) > 0.5:

            self.chase_camera.distance = max(height * 1.8, 15.0)

            self._edited_height = height

    def _launch_design(
        self,
        design
    ):

        self.vessel_design = design

        self.vessel_system.rebuild(self.scene, design, self.orbit_system)

        self.hud.notify(f"{design.name} on the pad: Z for full throttle, Space to launch")

    def _cancel_editing(self):

        if self._design_before_edit is not None:

            self.vessel_design = self._design_before_edit

            self.vessel_system.rebuild(self.scene, self.vessel_design, self.orbit_system)

        # Back to the vessel in flight it was opened from.
        if self._flying_before_edit is not None:

            self.vessel_system.switch(self.scene, self._flying_before_edit, self.orbit_system, self.planet_system)

            self._flying_before_edit = None
