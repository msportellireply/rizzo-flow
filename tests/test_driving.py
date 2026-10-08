"""Exercise the extracted driving UI and the real local question adapter."""

import copy
import json
import os

import pytest
from fastapi.testclient import TestClient
from test_service import FakeBackend

from rizzo_flow.api import create_app
from rizzo_flow.driving import State, driving_answers, question_states
from rizzo_flow.engine import Engine

# Initial observation emitted by the extracted Simulation.observe().
OBSERVATION = json.loads(
    r"""{"ego":{"speed_kmh":0,"lane":1,"changing_lane":false,"stopping_distance_m":0},"environment":{"weather":"clear","time_of_day":"day","visibility_m":220,"grip":1,"speed_limit_kmh":50},"signal":{"color":"green","stop_line_distance_m":83.7,"changes_in_s":19,"in_intersection":false,"approach_phase":"clear","approach_speed_kmh":50,"comfortable_stopping_distance_m":1.5},"road_users":[{"kind":"car","distance_m":45.2,"lane":0,"speed_kmh":22.367814553901553,"crossing":false},{"kind":"car","distance_m":79.2,"lane":1,"speed_kmh":23.34861212829128,"crossing":false},{"kind":"cross_traffic","distance_m":93.2,"lane":-1,"speed_kmh":26.68075907835737,"crossing":true},{"kind":"car","distance_m":113.2,"lane":0,"speed_kmh":30.08439862076193,"crossing":false},{"kind":"car","distance_m":147.2,"lane":1,"speed_kmh":28.05159485852346,"crossing":false},{"kind":"car","distance_m":181.2,"lane":0,"speed_kmh":21.96846003457904,"crossing":false},{"kind":"car","distance_m":215.2,"lane":1,"speed_kmh":25.890651900786906,"crossing":false}],"lanes":[{"lane":0,"gap_ahead_m":45,"gap_behind_m":220,"lead_speed_kmh":22.367814553901553,"rear_speed_kmh":0,"safe_to_enter":true},{"lane":1,"gap_ahead_m":79,"gap_behind_m":220,"lead_speed_kmh":23.34861212829128,"rear_speed_kmh":0,"safe_to_enter":true}],"navigation":{"current_street":"Linden Avenue","heading":"Northbound","planned_direction":"undecided","requested_direction":"auto","preferred_direction":"left","required_lane":-1,"turning":false,"turn_clear":true,"turn_conflict_distance_m":null,"options":[{"direction":"straight","street":"Linden Avenue","vehicles_ahead":4,"visits":0},{"direction":"left","street":"Market Street","vehicles_ahead":0,"visits":0},{"direction":"right","street":"Market Street","vehicles_ahead":1,"visits":0}]},"overtaking":{"active":false,"passed_clear":false,"lead_gap_m":79,"lead_speed_kmh":23.34861212829128,"beneficial":false,"right_lane_clear":true,"lane_change_allowed":true}}"""
)


def test_driving_assets_and_local_decisions():
    with TestClient(create_app(Engine(FakeBackend()))) as client:
        page = client.get("/drive")
        assert page.status_code == 200
        assert "/drive/assets/app.js" in page.text
        assert "TypeSafe API" not in page.text
        assert client.get("/drive-classic").status_code == 200
        for asset in (
            "app.js",
            "simulation.js",
            "network.js",
            "renderer.js",
            "style.css",
            "vendor/three.module.js",
            "vendor/three.core.js",
        ):
            assert client.get(f"/drive/assets/{asset}").status_code == 200
        status = client.get("/drive/api/status").json()
        assert status["configured"] is True
        response = client.post("/drive/api/decide", json={"sequence": 7, "state": OBSERVATION})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["sequence"] == 7
        assert result["model"] == status["model"]
        assert result["input_tokens"] > 0
        assert result["latency_ms"] >= 0
        assert set(result["answers"]) == {"pace", "lane", "route", "attention"}
        assert result["answers"]["pace"]["choice"] == "approach"
        assert result["answers"]["lane"]["choice"] == "left"
        assert result["answers"]["route"]["choice"] == "straight"
        assert result["answers"]["attention"]["choice"] == "signal"


def test_driving_rejects_invalid_observations():
    with TestClient(create_app(Engine(FakeBackend()))) as client:
        state = copy.deepcopy(OBSERVATION)
        state["ego"]["speed_kmh"] = -1
        assert (
            client.post("/drive/api/decide", json={"sequence": 0, "state": state}).status_code
            == 422
        )
        assert (
            client.post(
                "/drive/api/decide", json={"sequence": -1, "state": OBSERVATION}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/drive/api/decide", json={"sequence": 0, "state": OBSERVATION, "model": "external"}
            ).status_code
            == 422
        )


def test_driving_reports_context_overflow():
    with TestClient(create_app(Engine(FakeBackend(), ctx=10))) as client:
        response = client.post("/drive/api/decide", json={"sequence": 0, "state": OBSERVATION})
        assert response.status_code == 422
        assert "no truncation" in response.json()["detail"]


@pytest.mark.parametrize(
    ("navigation", "choice", "calls"),
    [
        ({}, "straight", 4),
        ({"requested_direction": "left"}, "left", 3),
        (
            {"destination": {"east": 160, "north": 100, "distance_m": 180, "reached": False}},
            "left",
            3,
        ),
        ({"planned_direction": "right"}, "keep", 3),
        ({"turning": True}, "keep", 3),
    ],
)
def test_navigator_routes_skip_model_scoring(navigation, choice, calls):
    class CountingBackend(FakeBackend):
        calls = 0

        def score(self, prefix, jobs, mode):
            self.calls += 1
            if calls == 3:
                assert all(job.id != "route" for job in jobs)
            return super().score(prefix, jobs, mode)

    backend = CountingBackend()
    state = copy.deepcopy(OBSERVATION)
    state["navigation"].update(navigation)
    with TestClient(create_app(Engine(backend))) as client:
        response = client.post("/drive/api/decide", json={"sequence": 1, "state": state})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["answers"]["route"]["choice"] == choice
        assert result["routing_source"] == ("navigator" if calls == 3 else "clm")
        assert sum(result["answers"]["route"]["probabilities"].values()) == pytest.approx(1)
        assert backend.calls == calls


def test_driving_api_key():
    with TestClient(create_app(Engine(FakeBackend()), api_key="test-key")) as client:
        assert client.get("/drive/api/status").status_code == 401
        payload = {"sequence": 0, "state": OBSERVATION}
        assert client.post("/drive/api/decide", json=payload).status_code == 401
        headers = {"Authorization": "Bearer test-key"}
        assert client.get("/drive/api/status", headers=headers).status_code == 200
        assert client.post("/drive/api/decide", json=payload, headers=headers).status_code == 200


def driving_scenario(name):
    """Controlled cases isolate decisions from unrelated background traffic."""
    state = copy.deepcopy(OBSERVATION)
    state["road_users"] = []
    for lane in state["lanes"]:
        lane.update(gap_ahead_m=220, gap_behind_m=220, lead_speed_kmh=0, rear_speed_kmh=0)
    state["overtaking"].update(lead_gap_m=220, lead_speed_kmh=0, beneficial=False)
    expected = {"pace": "cruise", "lane": "hold", "route": "left", "attention": "open_road"}
    if name == "red_at_line":
        state["signal"].update(
            color="red", stop_line_distance_m=1.5, approach_phase="at_line", approach_speed_kmh=0
        )
        expected.update(pace="stop", attention="signal")
    elif name == "pedestrian":
        state["ego"].update(speed_kmh=35, stopping_distance_m=20)
        state["road_users"] = [
            {"kind": "pedestrian", "distance_m": 12, "lane": 1, "speed_kmh": 4, "crossing": True}
        ]
        state["navigation"].update(planned_direction="left", required_lane=0)
        expected.update(pace="stop", lane="left", route="keep", attention="pedestrian")
    elif name == "clear_turn":
        state["navigation"].update(turning=True, planned_direction="left", required_lane=0)
        state["signal"].update(
            color="red", in_intersection=True, stop_line_distance_m=-5, approach_phase="clear"
        )
        state["overtaking"]["lane_change_allowed"] = False
        expected.update(pace="slow", route="keep")
    elif name != "clear":
        raise ValueError(name)
    return state, expected


# Opt-in smoke checks of model behavior; FakeBackend tests above verify only integration.
# Run: RIZZO_REAL=1 .venv/bin/pytest -q -s tests/test_driving.py -m integration
@pytest.fixture(scope="module")
def driving_backend():
    if os.environ.get("RIZZO_REAL") != "1":
        pytest.skip("set RIZZO_REAL=1 to load real weights")
    from rizzo_flow.loader import load_backend

    backend = load_backend(threads=4)
    yield backend
    backend.session.close()


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["clear", "red_at_line", "pedestrian", "clear_turn"])
def test_driving_model_scenarios(driving_backend, scenario):
    state, expected = driving_scenario(scenario)
    engine = Engine(driving_backend)
    try:
        result = driving_answers(engine, State.model_validate(state))
    finally:
        engine._worker.shutdown()
    choices = {key: answer["choice"] for key, answer in result["answers"].items()}
    print(scenario, choices, "input_tokens:", result["input_tokens"])
    assert choices == expected


def test_driving_observations_do_not_leak_future_routes_or_passed_signals():
    state, _ = driving_scenario("clear_turn")
    observations = question_states(State.model_validate(state))
    assert set(observations["lane"]["navigation"]) == {"turning", "required_lane"}
    assert "signal" not in observations["pace"]
    assert "signal" not in observations["attention"]
    assert observations["route"]["navigation"]["preferred_direction"] == "left"
    state, _ = driving_scenario("red_at_line")
    observations = question_states(State.model_validate(state))
    assert observations["pace"]["signal"]["color"] == "red"
    assert observations["attention"]["signal"]["approach_phase"] == "at_line"
