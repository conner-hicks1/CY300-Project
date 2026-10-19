from dataclasses import dataclass, field

from imgui_bundle import imgui

from game import style
from game.map_view import PANEL_FLAGS


# =========================================================
# Programme Panel
# =========================================================
#
# P: the space programme (programme/career.py).
#
#   funds, the monthly budget, reputation, science (and
#   what is left to spend on research: Research...)
#   this vessel: what it costs, what landing it home brings
#                back, whether it is paid for
#   making propellant: what the plant aboard makes here
#   missions: open ones first (with their deadlines on the
#             calendar), then those waiting on others, done,
#             failed
#   launch windows: the next cheap departures to the
#             planets from home's orbit
#
# Drawing only.

GOOD = style.tone("good")
NOTE = style.tone("muted")
WARN = style.tone("warn")
BAD = style.tone("bad")
MONEY = style.tone("money")


@dataclass(slots=True)
class ProgrammeInfo:

    funds: float = 0.0
    budget: float = 0.0
    reputation: float = 0.0
    science: float = 0.0
    today: str = ""

    # Science left to spend on research (None: a sandbox).
    spend: float | None = None

    # Mars's weather: (text, tone).
    mars: tuple | None = None

    # Flight data: (part name, what is flown, reliability
    # text, tone) for each part flown.
    flight_data: list = field(default_factory=list)

    # This vessel: (name, cost, recovery value, paid).
    vessel: tuple | None = None

    # Propellant plant: (kg a day now, what it is made from,
    # made so far in all kg).
    plant: tuple | None = None

    # (title, about, state, deadline text or "", reward text).
    missions: list = field(default_factory=list)

    # (target, departure date text, in days, flight days,
    # excess m/s).
    windows: list = field(default_factory=list)


class ProgrammePanel:

    def __init__(self):

        self.open = False

        # Research... clicked (the game opens the tech tree).
        self.research_requested = False

    def toggle(self):

        self.open = not self.open

    def draw(
        self,
        info: ProgrammeInfo
    ):

        if not self.open:
            return

        viewport = imgui.get_main_viewport()

        imgui.set_next_window_pos(
            (viewport.work_pos.x + viewport.work_size.x - 10.0, viewport.work_pos.y + viewport.work_size.y * 0.2),
            imgui.Cond_.first_use_ever,
            (1.0, 0.0)
        )

        imgui.set_next_window_size_constraints((380.0, 0.0), (560.0, viewport.work_size.y * 0.75))

        opened, self.open = imgui.begin("Space programme (P)", self.open, PANEL_FLAGS)

        if not opened:

            imgui.end()

            return

        imgui.text_colored(MONEY, f"${info.funds:,.1f}M")
        imgui.same_line()
        imgui.text_colored(NOTE, f"(+${info.budget:,.1f}M a month)")

        imgui.text(f"Reputation {info.reputation:,.0f}   Science {info.science:,.1f}")

        if info.spend is not None:

            imgui.same_line()
            imgui.text_colored(style.tone("science"), f"({info.spend:,.1f} to spend)")

        if imgui.button("Research..."):
            self.research_requested = True

        imgui.same_line()

        imgui.text_disabled(info.today)

        if info.vessel is not None:

            name, cost, back, paid = info.vessel

            imgui.separator()

            imgui.text(f"{name}: ${cost:,.1f}M")
            imgui.same_line()

            if paid:
                imgui.text_colored(NOTE, f"paid; ${back:,.1f}M back landing home")
            elif cost > info.funds:
                imgui.text_colored(BAD, "cannot afford it")
            else:
                imgui.text_colored(NOTE, "paid at its first stage")

        if info.plant is not None:

            rate, source, made = info.plant

            if rate > 0.0:
                imgui.text_colored(GOOD, f"Propellant plant: {rate:,.0f} kg/day of {source}")
            else:
                imgui.text_colored(NOTE, f"Propellant plant: {source}")

            imgui.text_disabled(f"  {made:,.0f} kg made away from home in all")

        imgui.separator()

        imgui.text("Missions")

        order = {"open": 0, "waiting": 1, "locked": 2, "done": 3, "failed": 4}

        for title, about, state, deadline, reward in sorted(info.missions, key=lambda item: order.get(item[2], 5)):

            if state == "open":

                imgui.text_colored(WARN if deadline else (0.95, 0.95, 0.95, 1.0), f"  {title}")

                imgui.same_line()
                imgui.text_colored(MONEY, reward)

                if about:
                    imgui.text_wrapped(f"      {about}")

                if deadline:
                    imgui.text_colored(WARN, f"      {deadline}")

            elif state == "done":

                imgui.text_colored(GOOD, f"  {title}: done")

            elif state == "failed":

                imgui.text_colored(BAD, f"  {title}: missed")

            elif state == "waiting":

                imgui.text_disabled(f"  {title}: {deadline or 'opens later'}")

            else:

                imgui.text_disabled(f"  {title} (after others)")

        if info.flight_data:

            imgui.separator()

            if imgui.tree_node("Flight data (reliability)"):

                for name, flown, odds, tone in info.flight_data:

                    imgui.text(f"  {name}")
                    imgui.same_line()
                    imgui.text_colored(style.tone(tone), odds)
                    imgui.text_disabled(f"      {flown}")

                imgui.tree_pop()

        if info.mars is not None:

            imgui.separator()

            text, tone = info.mars

            imgui.text_colored(style.tone(tone), text)

        if info.windows:

            imgui.separator()

            imgui.text("Launch windows from home")

            for target, when, days, flight, excess in info.windows:

                text = f"  {target}: {when}" + (f" (in {days:,.0f} days)" if days >= 1.0 else " (now)")

                imgui.text_colored(GOOD if days < 1.0 else (0.95, 0.95, 0.95, 1.0), text)
                imgui.text_disabled(f"      {flight:,.0f} days on the way, {excess:,.0f} m/s beyond escape")

        imgui.end()


def deadline_text(
    deadline,
    today
) -> str:

    if deadline is None:
        return ""

    days = (deadline - today).days

    if days < 0:
        return f"was due {deadline.isoformat()}"

    return f"by {deadline.isoformat()} ({days:,} days left)" if days < 3650 else f"by {deadline.isoformat()}"


def reward_text(
    reward
) -> str:

    parts = [f"${reward.funds:,.0f}M"] if reward.funds else []

    if reward.reputation:
        parts.append(f"{reward.reputation:+.0f} rep")

    if reward.science:
        parts.append(f"{reward.science:+.0f} sci")

    return ", ".join(parts)

