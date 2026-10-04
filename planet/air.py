import math

from dataclasses import dataclass


# =========================================================
# The Air Column
# =========================================================
#
# Temperature and pressure with height, enough to place
# cloud decks where they condense, to show a descending
# camera the conditions around it, and to light the deep
# air of the giants by its own heat.
#
#   solid worlds   the temperature falls by the lapse rate
#                  up to the tropopause, where it reaches the
#                  radiative skin temperature (equilibrium /
#                  2^(1/4): Earth ~215 K, Titan ~70 K), then
#                  stays; pressure from hydrostatic balance
#   giants         above the visible cloud tops a stable,
#                  near-isothermal layer; below them the gas
#                  is convecting: temperature rises along the
#                  dry adiabat (Jupiter ~2 K per km), and
#                  pressure with it, P ~ T^(cp/R) (the Galileo
#                  probe found ~22 bar and 425 K at 150 km)
#
# Altitudes are km above the reference radius (sea level, or
# a giant's cloud tops), negative below.

GAS_CONSTANT = 8.314                    # J / (mol K)


@dataclass(frozen=True, slots=True)
class AirColumn:

    surface_pressure: float             # bar at altitude 0
    surface_temperature: float          # K at altitude 0
    molar_mass: float                   # g / mol
    gravity: float                      # m / s^2
    specific_heat: float                # J / (kg K)
    lapse_rate: float                   # K / km near the surface (solid worlds)
    giant: bool = False
    skin_temperature: float = 0.0       # K above the tropopause (0 = 60% of the surface)

    @classmethod
    def from_scale_height(
        cls,
        surface_pressure: float,
        surface_temperature: float,
        scale_height_km: float,
        gravity: float,
        adiabatic_exponent: float,
        lapse_rate: float,
        giant: bool = False,
        skin_temperature: float = 0.0
    ) -> "AirColumn":
        """
        From what a scene's components keep: the scale height
        gives the gas constant (H = R T / g), cp / R the heat
        capacity.
        """

        gas_constant = max(scale_height_km, 1e-3) * 1000.0 * gravity / max(surface_temperature, 1.0)

        return cls(
            surface_pressure=surface_pressure,
            surface_temperature=surface_temperature,
            molar_mass=GAS_CONSTANT / gas_constant * 1000.0,
            gravity=gravity,
            specific_heat=adiabatic_exponent * gas_constant,
            lapse_rate=lapse_rate,
            giant=giant,
            skin_temperature=skin_temperature
        )

    @property
    def gas_constant(
        self
    ) -> float:
        """Specific gas constant (J / kg K)."""

        return GAS_CONSTANT / (self.molar_mass / 1000.0)

    @property
    def adiabatic_lapse(
        self
    ) -> float:
        """Dry adiabatic lapse rate g / cp (K / km)."""

        return self.gravity / self.specific_heat * 1000.0

    @property
    def adiabatic_exponent(
        self
    ) -> float:
        """cp / R: on an adiabat P ~ T^(cp/R), density ~ T^(cp/R - 1)."""

        return self.specific_heat / self.gas_constant

    @property
    def scale_height(
        self
    ) -> float:
        """At altitude 0 (km)."""

        return self.gas_constant * self.surface_temperature / self.gravity / 1000.0

    @property
    def _lapse(
        self
    ) -> float:

        return self.adiabatic_lapse if self.giant else max(self.lapse_rate, 0.0)

    @property
    def _tropopause_temperature(
        self
    ) -> float:

        if self.skin_temperature > 0.0:
            return min(self.skin_temperature, self.surface_temperature)

        return 0.6 * self.surface_temperature

    # -----------------------------------------------------

    def temperature(
        self,
        altitude_km: float
    ) -> float:

        t0 = self.surface_temperature

        if self.giant:

            # Convecting below the tops; ~isothermal above.
            if altitude_km >= 0.0:
                return t0

            return t0 - self._lapse * altitude_km

        return max(t0 - self._lapse * altitude_km, self._tropopause_temperature)

    def pressure(
        self,
        altitude_km: float
    ) -> float:
        """bar."""

        p0 = self.surface_pressure
        t0 = self.surface_temperature

        if self.giant and altitude_km >= 0.0:
            return p0 * math.exp(-altitude_km / self.scale_height)

        lapse = self._lapse

        if lapse <= 0.0:
            return p0 * math.exp(-altitude_km / self.scale_height)

        # Linear temperature: P = P0 (T / T0)^(g / (R lapse)).
        exponent = self.gravity / (self.gas_constant * lapse / 1000.0)

        if self.giant:
            return p0 * (self.temperature(altitude_km) / t0) ** exponent

        tropopause = (t0 - self._tropopause_temperature) / lapse

        if altitude_km <= tropopause:
            return p0 * (self.temperature(altitude_km) / t0) ** exponent

        p_tropopause = p0 * (self._tropopause_temperature / t0) ** exponent

        height = self.gas_constant * self._tropopause_temperature / self.gravity / 1000.0

        return p_tropopause * math.exp(-(altitude_km - tropopause) / height)

    def altitude_of_pressure(
        self,
        pressure_bar: float
    ) -> float:
        """km (bisection: pressure falls with height)."""

        low, high = -5_000.0, 2_000.0

        for _ in range(80):

            middle = 0.5 * (low + high)

            if self.pressure(middle) > pressure_bar:
                low = middle
            else:
                high = middle

        return 0.5 * (low + high)
