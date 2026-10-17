import json

from pathlib import Path

from core.logger import Logger

from ecs.components import BodyComponent, VesselComponent

from science.record import ScienceResult

from vessel.records import (
    node_from_json,
    node_to_json,
    state_from_json,
    state_to_json,
    vessel_from_json,
    vessel_to_json
)


# =========================================================
# The Fleet
# =========================================================
#
# Every vessel left in flight, kept across scenes (a revert,
# a trip elsewhere) and sessions: data/programme/fleet.json.
# Each with the clock time it was saved at; restored, it is
# brought up to now on rails (landed: turned with its body;
# in flight: along its conic; down meanwhile: landed under a
# parachute, else lost). A vessel never launched (still on
# the pad, unpaid) is not kept.

FLEET_PATH = Path("data/programme/fleet.json")


def body_profiles(
    scene
) -> dict:
    """Profile id -> body entity in the scene ("sun" -> None)."""

    found = {"sun": None}

    for entity, body in scene.registry.view_with(BodyComponent):
        if body.profile:
            found[body.profile] = entity

    return found


def snapshot(
    scene,
    vessel_system,
    science_system,
    now: float,
    include_active: bool = True
) -> list[dict]:
    """The vessels in flight as records (the active one too, unless not wanted)."""

    active = vessel_system.find(scene)

    active_entity = active[0] if active is not None else None

    records = []

    for entity, component in vessel_system.vessels(scene):

        if component.state.destroyed or not component.paid:
            continue

        if entity == active_entity and not include_active:
            continue

        records.append({
            "vessel": vessel_to_json(component.vessel),
            "state": state_to_json(component.state),
            "body": science_system.body_id(scene, component.body),
            "epoch": now,
            "landed_on": sorted(component.landed_on),
            "science": [[result.subject, result.title, result.cap] for result in component.science],
            "node": node_to_json(component.node),
            "active": entity == active_entity,
        })

    return records


def restore(
    records: list[dict],
    scene,
    vessel_system,
    orbit_system,
    parts: dict,
    now: float,
    activate: bool = False
) -> list[str]:
    """
    Put the records' vessels in the scene, brought up to now.
    activate: fly the one that was flown when saved. Returns
    notices (vessels lost meanwhile).
    """

    profiles = body_profiles(scene)

    notices = []

    for record in records:

        body_id = record.get("body", "")

        if body_id not in profiles:
            continue

        body = profiles[body_id]

        if body is not None and body not in orbit_system.states:
            continue

        try:
            vessel = vessel_from_json(record["vessel"], parts)
            state = state_from_json(record["state"])
        except (KeyError, ValueError, TypeError) as error:
            Logger.warning("[Fleet] A saved vessel could not be restored: %s", error)
            continue

        entity = vessel_system.add_vessel(
            scene,
            vessel,
            state,
            body,
            orbit_system,
            paid=True,
            landed_on=set(record.get("landed_on", [])),
            science=[ScienceResult(*item) for item in record.get("science", [])],
            node=node_from_json(record.get("node"))
        )

        component = scene.get_component(entity, VesselComponent)

        # Up to now, on rails.
        elapsed = now - float(record.get("epoch", now))

        if elapsed > 0.0:

            for event in vessel_system._background(scene, entity, component, elapsed, orbit_system):

                if event.startswith("lost:"):
                    notices.append(f"{vessel.name} was lost while you were away")

        if component.state.destroyed:

            vessel_system.remove(scene, entity)

            continue

        if activate and record.get("active"):
            vessel_system.active = entity

    vessel_system.place(scene, orbit_system)

    return notices


def save(
    records: list[dict],
    path: Path = FLEET_PATH
):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")

    temporary.write_text(json.dumps({"format": 1, "vessels": records}), encoding="utf-8")

    temporary.replace(path)


def load(
    path: Path = FLEET_PATH
) -> list[dict]:

    try:
        return list(json.loads(Path(path).read_text(encoding="utf-8")).get("vessels", []))
    except (OSError, ValueError, AttributeError):
        return []
