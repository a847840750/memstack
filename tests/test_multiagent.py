"""MultiAgentHub 测试：隔离、共享、授权、事件总线。"""

import unittest

from memstack import MemoryStack
from memstack.multiagent import AccessDenied, MultiAgentHub


def _hub() -> MultiAgentHub:
    return MultiAgentHub(MemoryStack())


class TestMultiAgentHub(unittest.TestCase):
    def setUp(self) -> None:
        self.hub = _hub()
        self.hub.register_agent("planner", role="planner")
        self.hub.register_agent("executor", role="executor")
        self.hub.register_agent("outsider")
        self.hub.create_team("trip", members=["planner", "executor"], policy="closed")

    def test_private_isolation(self) -> None:
        self.hub.remember_as("executor", [{"role": "user", "content": "The flight departs June 3rd."}])
        bundle = self.hub.recall_as("planner", "flight departure date")
        self.assertNotIn("June", bundle.text)

    def test_team_share_visible_to_members(self) -> None:
        self.hub.remember_as(
            "planner",
            [{"role": "user", "content": "The trip budget is 5000 dollars."}],
            team_id="trip", share=True,
        )
        bundle = self.hub.recall_as("executor", "what is the trip budget", teams=["trip"])
        self.assertIn("5000", bundle.text)
        self.assertIn("team:trip", bundle.trace["merge"]["parts"])

    def test_closed_team_blocks_outsider(self) -> None:
        with self.assertRaises(AccessDenied):
            self.hub.recall_as("outsider", "budget", teams=["trip"])
        with self.assertRaises(AccessDenied):
            self.hub.remember_as(
                "outsider", [{"role": "user", "content": "spam"}], team_id="trip",
            )

    def test_open_team_allows_anyone_to_read(self) -> None:
        self.hub.create_team("public", members=["planner"], policy="open")
        self.hub.remember_as(
            "planner", [{"role": "user", "content": "The meetup is Friday."}],
            team_id="public", share=True,
        )
        bundle = self.hub.recall_as("outsider", "when is the meetup", teams=["public"])
        self.assertIn("Friday", bundle.text)

    def test_grant_reads_private_memory(self) -> None:
        self.hub.remember_as("executor", [{"role": "user", "content": "Hotel is booked for two nights."}])
        bundle = self.hub.recall_as("planner", "hotel booking")
        self.assertNotIn("hotel", bundle.text)
        self.hub.grant("executor", "planner", "read")
        bundle = self.hub.recall_as("planner", "hotel booking")
        self.assertIn("hotel", bundle.text)
        self.hub.revoke("executor", "planner", "read")
        bundle = self.hub.recall_as("planner", "hotel booking")
        self.assertNotIn("hotel", bundle.text)

    def test_event_bus(self) -> None:
        event_id = self.hub.post_event("trip", "planner", "plan_ready", "Itinerary confirmed.")
        events = self.hub.pull_events("executor", "trip", since_id=0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["id"], event_id)
        self.assertEqual(events[0]["type"], "plan_ready")
        later = self.hub.pull_events("executor", "trip", since_id=event_id)
        self.assertEqual(later, [])

    def test_shared_supersede(self) -> None:
        self.hub.remember_as(
            "planner", [{"role": "user", "content": "The budget is 3000."}],
            team_id="trip", share=True,
        )
        self.hub.remember_as(
            "executor", [{"role": "user", "content": "The budget is now 4000."}],
            team_id="trip", share=True,
        )
        bundle = self.hub.recall_as("planner", "current trip budget", teams=["trip"])
        self.assertIn("4000", bundle.text)

    def test_events_as_memory(self) -> None:
        self.hub.post_event("trip", "planner", "decision", "We will fly instead of drive.",
                            as_memory=True)
        bundle = self.hub.recall_as("executor", "how will we travel", teams=["trip"])
        self.assertIn("fly", bundle.text)


if __name__ == "__main__":
    unittest.main()