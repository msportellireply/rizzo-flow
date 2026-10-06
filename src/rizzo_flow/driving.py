"""City driving observations and prompts adapted from Jev Drive; local inference only."""

from pathlib import Path
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .compat import ChoiceQuestion, SystemOneRequest, from_native, model_name, to_native

ASSETS = Path(__file__).with_name("driving")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Ego(StrictModel):
    speed_kmh: float = Field(ge=0, le=150)
    lane: int = Field(ge=0, le=1)
    changing_lane: bool
    stopping_distance_m: float = Field(ge=0, le=1000)


class Environment(StrictModel):
    weather: Literal["clear", "rain", "fog", "snow"]
    time_of_day: Literal["day", "night"]
    visibility_m: float = Field(ge=0, le=1000)
    grip: float = Field(gt=0, le=1)
    speed_limit_kmh: float = Field(ge=0, le=100)


class Signal(StrictModel):
    color: Literal["red", "amber", "green"]
    stop_line_distance_m: float = Field(ge=-100, le=1000)
    changes_in_s: float = Field(ge=0, le=60)
    in_intersection: bool
    approach_phase: Literal["clear", "cruise", "braking", "at_line"]
    approach_speed_kmh: float = Field(ge=0, le=50)
    comfortable_stopping_distance_m: float = Field(ge=0, le=2000)


class RoadUser(StrictModel):
    kind: Literal["car", "ambulance", "pedestrian", "barrier", "cross_traffic"]
    distance_m: float = Field(ge=-1000, le=1000)
    lane: int = Field(ge=-1, le=1)
    speed_kmh: float = Field(ge=0, le=150)
    crossing: bool


class Lane(StrictModel):
    lane: int = Field(ge=0, le=1)
    gap_ahead_m: float = Field(ge=0, le=1000)
    gap_behind_m: float = Field(ge=0, le=1000)
    safe_to_enter: bool
    lead_speed_kmh: float = Field(ge=0, le=150)
    rear_speed_kmh: float = Field(ge=0, le=150)


class RouteOption(StrictModel):
    direction: Literal["straight", "left", "right"]
    street: str = Field(max_length=60)
    vehicles_ahead: int = Field(ge=0, le=500)
    visits: int = Field(ge=0)


class Navigation(StrictModel):
    current_street: str = Field(max_length=60)
    heading: Literal["Northbound", "Eastbound", "Southbound", "Westbound"]
    planned_direction: Literal["undecided", "straight", "left", "right"]
    requested_direction: Literal["auto", "straight", "left", "right"]
    preferred_direction: Literal["straight", "left", "right"]
    required_lane: int = Field(ge=-1, le=1)
    turning: bool
    turn_clear: bool
    turn_conflict_distance_m: float | None = Field(default=None, ge=0, le=1000)
    options: list[RouteOption] = Field(min_length=3, max_length=3)


class Overtaking(StrictModel):
    active: bool
    passed_clear: bool
    lead_gap_m: float = Field(ge=0, le=1000)
    lead_speed_kmh: float = Field(ge=0, le=150)
    beneficial: bool
    right_lane_clear: bool
    lane_change_allowed: bool


class State(StrictModel):
    ego: Ego
    environment: Environment
    signal: Signal
    road_users: list[RoadUser] = Field(max_length=40)
    lanes: list[Lane] = Field(min_length=2, max_length=2)
    navigation: Navigation
    overtaking: Overtaking


class DecisionRequest(StrictModel):
    sequence: int = Field(ge=0)
    state: State


def questions():
    """Independent decisions: each question contains only its own driving rules."""
    return {
        "pace": ChoiceQuestion(
            type="choice",
            instructions=(
                "Choose the target speed for the CURRENT path, without assuming a lane change. "
                "Distances are metres of clearance; negative means behind. "
                "Outside a turn, only same-lane vehicles/barriers and crossing road users ahead "
                "constrain speed. During a turn, road_users is already filtered to the turn path. "
                "Apply these priorities:\n"
                "1. Stop for a crossing pedestrian, crossing vehicle, barrier or stationary lead "
                "vehicle within ego.stopping_distance_m + 8. Exception: at near-zero speed, "
                "a barrier more than 8m away permits crawl if a lane change is allowed and "
                "the adjacent lane is safe_to_enter. Steering requires forward motion.\n"
                "2. For a blocked turn, stop if turn_conflict_distance_m is within "
                "ego.stopping_distance_m + 2; otherwise approach cautiously. "
                "Before a planned turn, yield at the entrance if turn_clear=false "
                "and the stop line is within stopping distance. "
                "A clear ongoing turn uses slow; the controller caps cornering speed. "
                "Ignore entrance signals once signal.in_intersection or navigation.turning.\n"
                "3. Before entry, a red/amber signal uses approach_phase: at_line = stop, "
                "braking = approach, cruise = no signal speed reduction. "
                "Approach follows approach_speed_kmh, not an immediate stop.\n"
                "4. Otherwise match a nearby moving lead vehicle, slow for distant path hazards, "
                "and reduce speed for poor visibility/grip. Select the most restrictive "
                "applicable speed, respecting the speed limit. Never brake for actors behind "
                "or in another lane."
            ),
            criteria={
                "stop": "0 km/h: imminent current-path hazard, blocked turn entrance, or red/amber at_line BEFORE entry. Not a clear ongoing turn.",
                "approach": "Signal approach_speed_kmh: red/amber braking phase ONLY BEFORE entry; navigation.turning=false and signal.in_intersection=false.",
                "crawl": "10 km/h: very tight clearance or low-speed escape around a barrier.",
                "slow": "22 km/h: navigation.turning=true AND turn_clear=true, even if the entrance signal is red; also poor visibility/grip or slow traffic.",
                "steady": "35 km/h: moderately reduced speed for rain or moving traffic.",
                "cruise": "Posted speed limit: clear current path and good conditions; distant red alone does not prevent cruising.",
            },
        ),
        "lane": ChoiceQuestion(
            type="choice",
            instructions=(
                "Choose a lane action. Lane 0 is left/passing; lane 1 is right/travel. "
                "Check permissions BEFORE turn preparation: navigation.turning=true means HOLD. "
                "ego.changing_lane=true means HOLD. overtaking.lane_change_allowed=false means HOLD. "
                "Only when all three allow a change, consider a safe destination lane. "
                "Use only required_lane for turn preparation; preferred_direction and "
                "requested_direction are future routes, NOT lane-change commands. "
                "Then apply priorities: prepare navigation.required_lane (0 left, 1 right, "
                "-1 no requirement); yield right to an ambulance behind; keep an active pass "
                "until passed_clear; start a beneficial pass from lane 1 into lane 0; "
                "otherwise return from lane 0 to lane 1 when right_lane_clear. "
                "Hold if already in the required lane or no permitted move applies. "
                "Specifically, lane=1, required_lane=-1 and beneficial=false means hold."
            ),
            criteria={
                "hold": "Keep current lane. Required whenever turning=true OR changing_lane=true OR lane_change_allowed=false; also lane=1 with required_lane=-1 and beneficial=false.",
                "left": "Change 1 to 0 ONLY when turning=false, changing_lane=false, lane_change_allowed=true, destination safe, AND either required_lane=0 or (required_lane=-1 and beneficial=true).",
                "right": "Change 0 to 1 ONLY when turning=false, changing_lane=false, lane_change_allowed=true and destination safe: prepare right turn, yield to ambulance, or return after passing.",
            },
        ),
        "route": ChoiceQuestion(
            type="choice",
            instructions=(
                "Select only the future route, independently of current speed and lane. "
                "If navigation.turning or planned_direction is not undecided, choose keep. "
                "Otherwise select exactly navigation.preferred_direction: the simulator "
                "already combines the user's request and exploration preference. "
                "A red light does not change the route; pace controls stopping."
            ),
            criteria={
                "keep": "An existing route is committed: turning=true or planned_direction is straight/left/right.",
                "straight": "No committed route and preferred_direction=straight.",
                "left": "No committed route and preferred_direction=left.",
                "right": "No committed route and preferred_direction=right.",
            },
        ),
        "attention": ChoiceQuestion(
            type="choice",
            instructions=(
                "Classify the most relevant condition requiring attention now, independently "
                "of the other answers. Prefer immediate current-path hazards over distant ones: "
                "crossing pedestrian, obstruction, conflicting/followed traffic, signal before "
                "entry, ambulance behind, poor weather/visibility. Negative distance is behind. "
                "Adjacent-lane vehicles and distant cross traffic alone are not urgent. "
                "Ignore an entrance signal during a turn or inside an intersection. "
                "Choose open_road when no relevant condition constrains the drive."
            ),
            criteria={
                "open_road": "Clear path and good conditions; includes turn_clear=true during a turn with no path hazard. A passed red signal is irrelevant.",
                "signal": "Red/amber requiring braking or stopping ONLY when navigation.turning=false AND signal.in_intersection=false. Not a signal behind a turning car.",
                "pedestrian": "A pedestrian ahead crossing the current path.",
                "traffic": "Nearby lead vehicle or crossing traffic conflicting with the current path.",
                "weather": "Reduced visibility or grip requiring slower driving.",
                "obstruction": "Barrier blocking the current path ahead.",
                "emergency": "Ambulance behind requiring room to pass.",
            },
        ),
    }


def question_states(state: State):
    """Expose only relevant observations, so independent choices do not imply each other."""
    data = state.model_dump()
    navigation = data["navigation"]
    before_entry = not (navigation["turning"] or data["signal"]["in_intersection"])
    signal = {"signal": data["signal"]} if before_entry else {}
    return {
        "pace": {
            "ego": data["ego"],
            "environment": data["environment"],
            "road_users": data["road_users"],
            "lanes": data["lanes"],
            "navigation": {
                key: navigation[key]
                for key in (
                    "turning",
                    "turn_clear",
                    "turn_conflict_distance_m",
                    "planned_direction",
                )
            },
            "overtaking": {"lane_change_allowed": data["overtaking"]["lane_change_allowed"]},
            **signal,
        },
        "lane": {
            "ego": {key: data["ego"][key] for key in ("lane", "changing_lane")},
            "lanes": data["lanes"],
            "navigation": {key: navigation[key] for key in ("turning", "required_lane")},
            "overtaking": data["overtaking"],
            "road_users": [actor for actor in data["road_users"] if actor["kind"] == "ambulance"],
        },
        "route": {
            "navigation": {
                key: navigation[key]
                for key in ("turning", "planned_direction", "preferred_direction")
            },
        },
        "attention": {
            "ego": data["ego"],
            "environment": data["environment"],
            "road_users": data["road_users"],
            "navigation": {key: navigation[key] for key in ("turning", "turn_clear")},
            **signal,
        },
    }


def driving_answers(engine, state: State):
    answers = {}
    input_tokens = 0
    observations = question_states(state)
    for key, question in questions().items():
        wire = SystemOneRequest(
            model="rizzo-latest", state=observations[key], questions={key: question}
        )
        native, options = to_native(wire)
        result = from_native(
            wire, engine.decide(native), options, model_name(engine.backend.metadata)
        )
        answers.update(result["answers"])
        input_tokens += result["usage"]["input_tokens"]
    return {
        "answers": answers,
        "input_tokens": input_tokens,
        "model": model_name(engine.backend.metadata),
    }


def register_driving(app, engine):
    router = APIRouter(prefix="/drive")

    @router.get("", include_in_schema=False)
    def index():
        return FileResponse(ASSETS / "index.html")

    @router.get("/api/status")
    def status():
        return {"configured": True, "model": model_name(engine.backend.metadata)}

    @router.post("/api/decide")
    def decide(body: DecisionRequest):
        started = perf_counter()
        try:
            result = driving_answers(engine, body.state)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "sequence": body.sequence,
            "answers": result["answers"],
            "model": result["model"],
            "latency_ms": round((perf_counter() - started) * 1000),
            "input_tokens": result["input_tokens"],
        }

    app.include_router(router)
    app.mount("/drive/assets", StaticFiles(directory=ASSETS), name="driving-assets")
