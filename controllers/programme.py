from game.programme_panel import ProgrammeInfo, deadline_text, reward_text
from game.research_panel import ResearchInfo

from programme import research

from vessel import propellants

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

        for window in programme.windows(self):

            info.windows.append((
                window.target,
                date_of(window.departure).strftime("%d %b %Y"),
                (window.departure - now) / 86_400.0,
                window.flight / 86_400.0,
                window.excess
            ))

        self.programme_panel.draw(info)

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
