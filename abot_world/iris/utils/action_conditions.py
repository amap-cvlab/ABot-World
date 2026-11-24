"""Text-conditioned basic actions for standalone and interactive inference."""

BODY_TEXT = {
    "Ctrl": "The character bends their knees, lowers their body, and maintains a stable crouched posture close to the ground",
    "Shift": "The character leans forward, accelerates into a fast sprint, and moves quickly through the scene",
    "Space": "The character bends their knees, pushes off the ground, rises into the air, and lands back on their feet",
}
# Captions remain independent: the runtime never encodes skill/action products.
BASIC_ACTION_TEXT = {
    "MouseRight": "The character raises the equipped weapon, carefully lines it up with the target, and maintains a steady aiming posture",
    "MouseLeft": "The character readies the equipped weapon, directs it toward the target, and performs a forceful attack",
    "Alt": "The character lowers their body, tucks into a roll, rolls across the ground, and returns to a stable standing posture",
}
CONTROL_TEXT = {**BODY_TEXT, **BASIC_ACTION_TEXT}
# Match approved combinations before single actions. Among combinations, keep
# the lead action's existing priority: jump, attack, then crouch.
ACTION_COMBINATIONS = {
    ("Shift", "Space"): "The character sprints forward, pushes off the ground into a running jump, travels through the air, and lands back on their feet",
    ("MouseRight", "MouseLeft"): "The character keeps the equipped weapon aimed at the target and performs an attack while maintaining the aiming posture",
    ("Ctrl", "Shift"): "The character stays low in a crouched posture and moves quickly forward with bent knees, keeping their body close to the ground",
}

PULSE_DURATIONS = {"Alt": 2, "MouseLeft": 2, "Space": 1}
ACTION_PRIORITY = ("Alt", "Space", "MouseLeft", "MouseRight", "Ctrl", "Shift")


def select_action(keys):
    for combo in ACTION_COMBINATIONS:
        if keys.issuperset(combo):
            return combo
    return next(((key,) for key in ACTION_PRIORITY if key in keys), ())


class ActionConditions:
    """Consume one held/tapped snapshot per generated block."""

    def __init__(self):
        self.selected = ()
        self.pulse = None
        self.remaining = 0

    def step(self, held, tapped=()):
        held = set(held) & CONTROL_TEXT.keys()
        tapped = set(tapped) & CONTROL_TEXT.keys()
        sustained = held - PULSE_DURATIONS.keys()
        if tapped:
            # Match the online controls: a new press interrupts the previous
            # pulse and can pair with a held key in a supported combination.
            self.pulse, self.remaining = None, 0
            self.selected = next((combo for combo in ACTION_COMBINATIONS
                                  if set(combo) <= sustained | tapped and tapped.intersection(combo)),
                                 select_action(tapped))
        candidates = sustained.intersection(self.selected)
        if not tapped and not self.remaining:
            # A completed pulse must not suppress keys that are still held.
            candidates = sustained
        if tapped:
            candidates.update(self.selected)
        if self.remaining:
            candidates.add(self.pulse)
        action = select_action(candidates)
        pulse = next((key for key in action if key in PULSE_DURATIONS), None)
        if pulse is not None and (pulse != self.pulse or pulse in tapped):
            self.remaining = PULSE_DURATIONS[pulse]
        self.pulse = pulse
        self.remaining = max(0, self.remaining - 1) if pulse else 0
        if not action:
            return (), ""
        return action, ACTION_COMBINATIONS[action] if len(action) > 1 else CONTROL_TEXT[action[0]]
