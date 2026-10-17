import numpy as np

from vessel.maneuver import ManeuverNode
from vessel.physics import VesselState
from vessel.vessel import PartState, Vessel


# =========================================================
# Vessels in Flight, Saved
# =========================================================
#
# A vessel as it is now (not as designed: staged, burned,
# hot), as plain JSON, and back: every part still attached
# with its own state, the flight state, the plan. The fleet
# (programme/fleet.py) keeps these for the vessels left in
# flight.

def vessel_to_json(
    vessel: Vessel
) -> dict:

    return {
        "name": vessel.name,
        "next_stage": vessel.next_stage,
        "parts": [
            {
                "part": state.part.id,
                "stage": state.stage,
                "propellant": state.propellant,
                "ignited": state.ignited,
                "armed": state.armed,
                "deployed": state.deployed,
                "count": state.count,
                "group": state.group,
                "offset": state.offset,
                "source": list(state.source),
                "temperature": state.temperature,
            }
            for state in vessel.parts
        ],
    }


def vessel_from_json(
    data: dict,
    parts: dict
) -> Vessel:
    """parts: the catalog (vessel/parts.py load_parts). KeyError for a part no longer in it."""

    states = []

    for raw in data["parts"]:

        states.append(PartState(
            part=parts[raw["part"]],
            stage=raw.get("stage"),
            propellant=float(raw.get("propellant", 0.0)),
            ignited=bool(raw.get("ignited", False)),
            armed=bool(raw.get("armed", False)),
            deployed=float(raw.get("deployed", 0.0)),
            count=int(raw.get("count", 1)),
            group=raw.get("group"),
            offset=float(raw.get("offset", 0.0)),
            source=tuple(raw.get("source", ())),
            temperature=float(raw.get("temperature", 290.0)),
        ))

    vessel = Vessel(str(data.get("name", "Vessel")), states, int(data.get("next_stage", 0)))

    vessel._restack()

    return vessel


def state_to_json(
    state: VesselState
) -> dict:

    return {
        "position": [float(x) for x in state.position],
        "velocity": [float(x) for x in state.velocity],
        "orientation": [float(x) for x in np.asarray(state.orientation, dtype=np.float64)],
        "angular_velocity": [float(x) for x in state.angular_velocity],
        "landed": state.landed,
        "thrust_dv": [float(x) for x in state.thrust_dv],
    }


def state_from_json(
    data: dict
) -> VesselState:

    return VesselState(
        position=np.array(data["position"], dtype=np.float64),
        velocity=np.array(data["velocity"], dtype=np.float64),
        orientation=np.array(data["orientation"], dtype=np.float64),
        angular_velocity=np.array(data.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64),
        landed=bool(data.get("landed", False)),
        thrust_dv=np.array(data.get("thrust_dv", [0.0, 0.0, 0.0]), dtype=np.float64),
    )


def node_to_json(
    node: ManeuverNode | None
) -> dict | None:

    if node is None:
        return None

    return {
        "time": node.time,
        "prograde": node.prograde,
        "normal": node.normal,
        "radial": node.radial,
        "vector": [float(x) for x in node.vector],
        "baseline": [float(x) for x in node.baseline],
    }


def node_from_json(
    data: dict | None
) -> ManeuverNode | None:

    if not data:
        return None

    return ManeuverNode(
        time=float(data["time"]),
        prograde=float(data.get("prograde", 0.0)),
        normal=float(data.get("normal", 0.0)),
        radial=float(data.get("radial", 0.0)),
        vector=np.array(data.get("vector", [0.0, 0.0, 0.0]), dtype=np.float64),
        baseline=np.array(data.get("baseline", [0.0, 0.0, 0.0]), dtype=np.float64),
    )
