from game.programme_panel import ProgrammeInfo, deadline_text, reward_text
from game.research_panel import ResearchInfo

from planet import weather
from programme import research

from vessel import propellants, reliability

from planet.orbits import date_of

from programme.career import RECOVERY, today_of


# =========================================================
# The Application: Programme
# =========================================================
#
# The space programme's panel, and the tech tree's.
#
# A part of Application (application.py), apart for its
# size: its methods run on the application (self is it).

def mars_forecast(
    now: float
) -> tuple[str, str]:
    """Mars's weather for the panel: (text, tone)."""

    ls = weather.solar_longitude(now)

    head = f"Mars: Ls {ls:,.0f}, {weather.season(ls).split(' / ')[0]}"

    blowing = weather.active(now)

    grand = [(storm, strength) for storm, strength in blowing if storm.global_storm]

    if grand:

        storm, strength = grand[0]

        return f"{head}: a global dust storm (dust {storm.tau * strength:.1f} everywhere); landers' power is down", "bad"

    if blowing:
        return f"{head}, the dust season: {len(blowing)} regional storm{'s' if len(blowing) > 1 else ''}", "warn"

    if weather.dusty_season(ls):
        return f"{head}, the dust season (storms likely until Ls {weather.DUST_SEASON[1]:.0f})", "warn"

    # Ls goes round in about 687 days.
    days = ((weather.DUST_SEASON[0] - ls) % 360.0) * 686.98 / 360.0

    return f"{head}: clear skies; the dust season in about {days:,.0f} days", "muted"


class ProgrammeController:

    def _draw_programme(self):

        if not self.programme_panel.open:
            return

        programme = self.programme_system

        career = programme.career

        today = today_of(date_of(self.orbit_system.time))

        info = ProgrammeInfo(
            funds=career.funds,
            budget=career.budget(programme.campaign),
            reputation=career.reputation,
            science=self.science_system.record.points,
            today=today.strftime("%d %B %Y"),
            spend=None if programme.sandbox else programme.science_to_spend(self.science_system.record)
        )

        found = self.vessel_system.find(self.scene)

        if found is not None:

            vessel = found[1].vessel

            info.vessel = (vessel.name, vessel.cost, vessel.cost * RECOVERY, found[1].paid)

            if any(state.part.isru is not None for state in vessel.parts):

                kind = programme.making_kind

                if programme.making > 0.0:

                    source = programme.resources[0].name if programme.resources else ""

                    source = f"{propellants.of(kind).short} from {source}"

                    if vessel.isru_share < 0.995:
                        source += f" ({vessel.isru_share:.0%} of its power)"

                elif kind.startswith("needs:"):

                    names = " or ".join(propellants.of(item).short for item in kind[6:].split(","))

                    source = f"{programme.resources[0].name} here, but no {names} tanks with room"

                elif vessel.isru_wanted and programme.resources:

                    source = f"{programme.resources[0].name}: no power to spare"
                elif programme.resources:
                    source = f"{programme.resources[0].name} here: land to make propellant"
                else:
                    source = "nothing to make it from here"

                info.plant = (programme.making, source, career.made)

        for mission in programme.campaign.missions:

            state = career.state(mission, today)

            info.missions.append((
                mission.title,
                mission.about,
                state,
                deadline_text(mission.deadline, today) if state != "waiting" else f"opens {mission.opens.isoformat()}",
                reward_text(mission.reward)
            ))

        now = self.orbit_system.time

        info.mars = mars_forecast(now)

        if not programme.sandbox:
            info.astronauts = [
                (a.name, a.role, a.status, a.flights, a.days, a.dose)
                for a in programme.career.corps.astronauts
            ]

        if programme.failures.enabled:

            for part_id, experience in sorted(programme.failures.record.parts.items()):

                part = self.parts.get(part_id)

                if part is None or not experience.ignitions:
                    continue

                if part.engine is not None:
                    chance = reliability.mission_reliability(experience, 300.0)
                    odds = f"{chance:.1%} for a 300 s burn"
                    flown = f"{experience.seconds:,.0f} s burned, lit {experience.ignitions}, {experience.failures} failures"
                else:
                    chance = 1.0 - reliability.chute_failure(experience)
                    odds = f"{chance:.1%} to open"
                    flown = f"opened {experience.ignitions}, {experience.failures} failures"

                info.flight_data.append((part.name, flown, odds, "good" if chance > 0.98 else ("warn" if chance > 0.93 else "bad")))

        for window in programme.windows(self):

            info.windows.append((
                window.target,
                date_of(window.departure).strftime("%d %b %Y"),
                (window.departure - now) / 86_400.0,
                window.flight / 86_400.0,
                window.excess
            ))

        self.programme_panel.draw(info)

        if self.programme_panel.hire_requested is not None:

            why = programme.hire(self.programme_panel.hire_requested)

            self.programme_panel.hire_requested = None

            for news in programme.news:
                self.hud.notify(news)

            programme.news.clear()

            if why:
                self.hud.notify(why)

        if self.programme_panel.research_requested:

            self.programme_panel.research_requested = False

            self.research_panel.open = True

    def _draw_research(self):

        if self.part_editor.research_requested:

            self.part_editor.research_requested = False

            self.research_panel.open = True

        if not self.research_panel.open:
            return

        programme = self.programme_system

        tree = programme.tree

        researched = programme.career.researched

        info = ResearchInfo(
            science=programme.science_to_spend(self.science_system.record),
            sandbox=programme.sandbox,
            nodes=[
                (node, "known" if programme.sandbox else research.state(tree, node, researched))
                for node in tree.nodes
            ],
            depth={node.id: tree.depth(node.id) for node in tree.nodes},
            part_names={part_id: part.name for part_id, part in self.parts.items()},
            node_names={node.id: node.name for node in tree.nodes}
        )

        for action in self.research_panel.draw(info):

            if action[0] == "research":

                why = programme.research(action[1], self.science_system.record)

                if why:
                    self.hud.notify(why)

            for news in programme.news:
                self.hud.notify(news)

            programme.news.clear()
