from ecs.components import ClockComponent


# =========================================================
# Time Warp
# =========================================================
#
# How fast simulated time runs, in steps (the "," and "."
# keys), as in most spaceflight games: real time, then
# faster by steps up to a year in under a minute. Paused is
# separate (the pause menu): the warp is kept for after it.
#
# The rate is the clock's (ClockComponent.rate: simulated
# seconds per real second); OrbitSystem moves the bodies by
# it.

WARP_RATES: tuple[float, ...] = (
    1.0,
    5.0,
    10.0,
    50.0,
    100.0,
    1_000.0,
    10_000.0,
    100_000.0,
    1_000_000.0,
)


class TimeWarp:

    def __init__(
        self,
        rates: tuple[float, ...] = WARP_RATES
    ):

        self.rates = rates

        self.index = 0

        self.paused = False

        # Fastest allowed now (the vessel's limit: physics
        # in the air or under thrust); inf: none.
        self.max_rate = float("inf")

    @property
    def rate(
        self
    ) -> float:
        """Simulated seconds per real second (0 while paused)."""

        return 0.0 if self.paused else self.rates[self.index]

    @property
    def warping(
        self
    ) -> bool:

        return self.index > 0

    def faster(self):

        self.index = min(self.index + 1, len(self.rates) - 1)

        self.limit(self.max_rate)

    def limit(
        self,
        max_rate: float
    ) -> bool:
        """
        Hold the warp to at most `max_rate` (real time always
        allowed). True when it had to slow down.
        """

        self.max_rate = max_rate

        lowered = False

        while self.index > 0 and self.rates[self.index] > max_rate:

            self.index -= 1

            lowered = True

        return lowered

    @property
    def capped(
        self
    ) -> bool:
        """The next step up is not allowed now."""

        return self.index + 1 < len(self.rates) and self.rates[self.index + 1] > self.max_rate

    def slower(self):

        self.index = max(self.index - 1, 0)

    def stop(self):
        """Back to real time."""

        self.index = 0

    def apply(
        self,
        clock: ClockComponent | None
    ):
        """Set the clock to run at the warp's rate."""

        if clock is not None:
            clock.rate = self.rate

    def label(
        self
    ) -> str:

        if self.paused:
            return "Paused"

        rate = self.rates[self.index]

        return "Real time" if rate == 1.0 else f"{rate:,.0f}x"
