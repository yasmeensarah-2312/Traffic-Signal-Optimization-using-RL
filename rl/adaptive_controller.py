"""Temporary queue-based adaptive traffic controller.
This is not a trained DRL agent.
Actions: 0->phase 0, 1->phase 2, 2->phase 4, 3->phase 6.
"""

GREEN_PHASES = (0, 2, 4, 6)


class AdaptiveController:
    def __init__(self, min_green_steps=5):
        self.min_green_steps = min_green_steps
        self.last_action = None
        self.steps_on_action = 0

    def select_action(self, queue_lengths, waiting_times, current_action=None):
        if len(queue_lengths) != 4 or len(waiting_times) != 4:
            raise ValueError("Provide four queue lengths and four waiting times.")

        queues = [max(0.0, float(x)) for x in queue_lengths]
        waits = [max(0.0, float(x)) for x in waiting_times]

        scores = [queues[i] + 0.25 * waits[i] for i in range(4)]
        best = max(range(4), key=lambda i: scores[i])

        if current_action is not None and not 0 <= current_action < 4:
            raise ValueError("current_action must be from 0 to 3.")

        if self.last_action != best:
            self.last_action = best
            self.steps_on_action = 1
        else:
            self.steps_on_action += 1

        if (
            current_action is not None
            and best != current_action
            and self.steps_on_action < self.min_green_steps
        ):
            best = current_action

        return {
            "action": best,
            "target_phase": GREEN_PHASES[best],
            "scores": scores,
        }
