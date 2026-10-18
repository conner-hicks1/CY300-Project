from game.programme_panel import ProgrammeInfo, deadline_text, reward_text

from planet.orbits import date_of

from programme.career import RECOVERY, today_of


# =========================================================
# The Application: Programme
# =========================================================
#
# The space programme's panel.
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
            today=today.strftime("%d %B %Y")
        )

        found = self.vessel_system.find(self.scene)

        if found is not None:

            vessel = found[1].vessel

            info.vessel = (vessel.name, vessel.cost, vessel.cost * RECOVERY, found[1].paid)

            if any(state.part.isru is not None for state in vessel.parts):

                if programme.making > 0.0:

                    source = programme.resources[0].name if programme.resources else ""

                    if vessel.isru_share < 0.995:
                        source += f" ({vessel.isru_share:.0%} of its power)"

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
