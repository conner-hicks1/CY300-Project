from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import format, style
from game.map_view import PANEL_FLAGS


# =========================================================
# Vessels Panel
# =========================================================
#
# L: every vessel in flight (a tracking station): where each
# is, how far from the one flown; fly another ([ and ]
# step through them), make one the map's target (for a
# rendezvous), end one's mission. Close to another vessel
# (within reach, nearly still to each other: docked in all
# but name) propellant can be moved between their tanks:
# a depot.
#
# Drawing only: clicks come back as actions.

GOOD = style.tone("good")
NOTE = style.tone("muted")
WARN = style.tone("warn")


@dataclass(slots=True)
class VesselRow:

    entity: object
    name: str
    body: str
    situation: str
    distance: float | None          # m from the flown one (None: it)
    active: bool = False
    targeted: bool = False
    crewed: bool = False
    propellant: float = 0.0         # kg
    capacity: float = 0.0
    docked: int = 1                 # vessels docked in one


@dataclass(slots=True)
class VesselsInfo:

    rows: list = field(default_factory=list)

    # A vessel close enough to move propellant with:
    # (entity, name, distance m, relative speed m/s), or None.
    nearby: tuple | None = None

    # The flown one's propellant (kg) and capacity.
    propellant: float = 0.0
    capacity: float = 0.0


class VesselsPanel:

    def __init__(self):

        self.open = False

        # The end-a-mission confirmation pending (entity).
        self._confirm = None

    def toggle(self):

        self.open = not self.open

    def draw(
        self,
        info: VesselsInfo
    ) -> list[tuple]:

        actions: list[tuple] = []

        if not self.open:
            return actions

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x * 0.5, viewport.work_pos.y + viewport.work_size.y * 0.15),
            imgui.Cond_.first_use_ever,
            (0.5, 0.0)
        )

        opened, self.open = imgui.begin("Vessels (L)", self.open, PANEL_FLAGS)

        if not opened:

            imgui.end()

            return actions

        imgui.text_disabled("[ and ] fly the previous / next")

        flags = imgui.TableFlags_.row_bg.value | imgui.TableFlags_.borders_inner_h.value

        if imgui.begin_table("vessels", 5, flags):

            for heading in ("Vessel", "Where", "Distance", "Propellant", ""):
                imgui.table_setup_column(heading)

            imgui.table_headers_row()

            for row in info.rows:

                imgui.table_next_row()

                imgui.table_next_column()

                name = row.name + ("  (crew)" if row.crewed else "") + (f"  ({row.docked} docked)" if row.docked > 1 else "")

                if row.active:
                    imgui.text_colored(GOOD, f"> {name}")
                else:
                    imgui.text(name)

                imgui.table_next_column()
                imgui.text(f"{row.situation}, {row.body}")

                imgui.table_next_column()
                imgui.text("-" if row.distance is None else format.distance(row.distance))

                imgui.table_next_column()

                if row.capacity > 0.0:
                    imgui.text(f"{row.propellant / 1000.0:,.1f} / {row.capacity / 1000.0:,.1f} t")
                else:
                    imgui.text_disabled("-")

                imgui.table_next_column()

                if row.active and row.docked > 1:

                    if imgui.small_button(f"Undock##undock{row.name}"):
                        actions.append(("undock",))

                    imgui.same_line()

                if not row.active:

                    if imgui.small_button(f"Fly##fly{id(row.entity)}{row.name}"):
                        actions.append(("fly", row.entity))

                    imgui.same_line()

                    if imgui.small_button(f"{'Untarget' if row.targeted else 'Target'}##target{id(row.entity)}{row.name}"):
                        actions.append(("target", None if row.targeted else row.entity))

                    imgui.same_line()

                if self._confirm == row.entity:

                    if imgui.small_button(f"Really end it##end{id(row.entity)}{row.name}"):

                        actions.append(("end", row.entity))

                        self._confirm = None

                    imgui.same_line()

                    if imgui.small_button(f"No##keep{id(row.entity)}{row.name}"):
                        self._confirm = None

                elif imgui.small_button(f"End mission##ask{id(row.entity)}{row.name}"):

                    self._confirm = row.entity

            imgui.end_table()

        if info.nearby is not None:

            entity, name, distance, speed = info.nearby

            imgui.separator()

            imgui.text_colored(GOOD, f"Alongside {name}: {format.distance(distance)}, {speed:,.1f} m/s apart")

            imgui.text(f"Move propellant (this vessel: {info.propellant / 1000.0:,.1f} / {info.capacity / 1000.0:,.1f} t):")

            for amount, label in ((1000.0, "1 t"), (10_000.0, "10 t"), (float("inf"), "all")):

                if imgui.small_button(f"Take {label}##take{label}"):
                    actions.append(("transfer", entity, amount))

                imgui.same_line()

            for amount, label in ((1000.0, "1 t"), (10_000.0, "10 t"), (float("inf"), "all")):

                if imgui.small_button(f"Give {label}##give{label}"):
                    actions.append(("transfer", entity, -amount))

                imgui.same_line()

            imgui.new_line()

        imgui.end()

        return actions
