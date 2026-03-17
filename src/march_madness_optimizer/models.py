from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Any


def logistic(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


def logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class TeamProfile:
    name: str
    region: str
    seed: int
    adj_o: float
    adj_d: float
    tempo: float
    last10_adj_em: float
    non_garbage_adj_em: float
    elo: float
    network: float
    resume: float
    three_point_rate: float
    three_point_defense: float
    turnover_rate: float
    turnover_creation: float
    offensive_rebounding: float
    defensive_rebounding: float
    free_throw_rate: float
    foul_rate: float
    size: float
    experience: float
    continuity: float
    bench_depth: float
    coach_score: float
    travel_index: float = 0.0
    rest_edge: float = 0.0
    availability: float = 1.0
    market_power: float = 0.0
    public_pick_rate: float = 0.0

    @property
    def adj_em(self) -> float:
        return self.adj_o - self.adj_d


class ProbabilityModel:
    """Blended matchup model built from ratings, styles, context, and market signal."""

    def __init__(
        self,
        logistic_weights: dict[str, float] | None = None,
        stack_weights: dict[str, float] | None = None,
        isotonic_like_temperature: float = 0.92,
    ) -> None:
        self.logistic_weights = logistic_weights or {
            "intercept": 0.0,
            "adj_em_diff": 0.082,
            "last10_diff": 0.028,
            "non_garbage_diff": 0.024,
            "elo_diff": 0.0060,
            "network_diff": 0.050,
            "resume_diff": 0.032,
            "seed_diff": -0.115,
            "travel_diff": -0.140,
            "rest_diff": 0.160,
            "coach_diff": 0.190,
            "availability_diff": 1.050,
            "pace_mismatch": -0.010,
            "three_point_edge": 0.850,
            "turnover_edge": 0.700,
            "rebounding_edge": 0.680,
            "free_throw_edge": 0.550,
            "size_edge": 0.240,
            "experience_edge": 0.200,
            "continuity_edge": 0.180,
            "bench_edge": 0.160,
        }
        self.stack_weights = stack_weights or {
            "logistic": 0.34,
            "elo": 0.22,
            "market": 0.24,
            "style": 0.20,
        }
        total = sum(self.stack_weights.values())
        self.stack_weights = {key: value / total for key, value in self.stack_weights.items()}
        self.temperature = isotonic_like_temperature

    def probability(self, team_a: TeamProfile, team_b: TeamProfile, round_index: int = 1) -> float:
        components = self.component_probabilities(team_a, team_b, round_index=round_index)
        blended_logit = sum(
            self.stack_weights[name] * logit(probability) for name, probability in components.items()
        )
        calibrated = logistic(blended_logit / self.temperature)
        return min(max(calibrated, 0.01), 0.99)

    def component_probabilities(
        self, team_a: TeamProfile, team_b: TeamProfile, round_index: int = 1
    ) -> dict[str, float]:
        logistic_base = logistic(self._linear_features(team_a, team_b, round_index=round_index))
        elo_prob = logistic((team_a.elo - team_b.elo) / 120.0)
        market_prob = logistic(
            (
                (team_a.market_power - team_b.market_power)
                + 0.55 * (team_a.adj_em - team_b.adj_em)
                + 0.30 * (team_a.resume - team_b.resume)
            )
            / 10.0
        )
        style_prob = logistic(self._style_signal(team_a, team_b) + 0.025 * (team_a.adj_em - team_b.adj_em))
        return {
            "logistic": logistic_base,
            "elo": elo_prob,
            "market": market_prob,
            "style": style_prob,
        }

    def all_matchups(self, teams: list[TeamProfile]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, team_a in enumerate(teams):
            for team_b in teams[index + 1 :]:
                probability = self.probability(team_a, team_b)
                rows.append(
                    {
                        "team_a": team_a.name,
                        "team_b": team_b.name,
                        "team_a_win_probability": round(probability, 4),
                        "team_b_win_probability": round(1.0 - probability, 4),
                    }
                )
        return rows

    def _linear_features(self, team_a: TeamProfile, team_b: TeamProfile, round_index: int) -> float:
        weights = self.logistic_weights
        pace_mismatch = abs(team_a.tempo - team_b.tempo)
        features = {
            "intercept": weights["intercept"],
            "adj_em_diff": weights["adj_em_diff"] * (team_a.adj_em - team_b.adj_em),
            "last10_diff": weights["last10_diff"] * (team_a.last10_adj_em - team_b.last10_adj_em),
            "non_garbage_diff": weights["non_garbage_diff"]
            * (team_a.non_garbage_adj_em - team_b.non_garbage_adj_em),
            "elo_diff": weights["elo_diff"] * (team_a.elo - team_b.elo),
            "network_diff": weights["network_diff"] * (team_a.network - team_b.network),
            "resume_diff": weights["resume_diff"] * (team_a.resume - team_b.resume),
            "seed_diff": weights["seed_diff"] * (team_a.seed - team_b.seed),
            "travel_diff": weights["travel_diff"] * (team_a.travel_index - team_b.travel_index),
            "rest_diff": weights["rest_diff"] * (team_a.rest_edge - team_b.rest_edge),
            "coach_diff": weights["coach_diff"] * (team_a.coach_score - team_b.coach_score),
            "availability_diff": weights["availability_diff"] * (team_a.availability - team_b.availability),
            "pace_mismatch": weights["pace_mismatch"] * pace_mismatch,
            "three_point_edge": weights["three_point_edge"]
            * (
                (team_a.three_point_rate - team_b.three_point_defense)
                - (team_b.three_point_rate - team_a.three_point_defense)
            ),
            "turnover_edge": weights["turnover_edge"]
            * (
                (team_a.turnover_creation - team_b.turnover_rate)
                - (team_b.turnover_creation - team_a.turnover_rate)
            ),
            "rebounding_edge": weights["rebounding_edge"]
            * (
                (team_a.offensive_rebounding - team_b.defensive_rebounding)
                - (team_b.offensive_rebounding - team_a.defensive_rebounding)
            ),
            "free_throw_edge": weights["free_throw_edge"]
            * ((team_a.free_throw_rate - team_b.foul_rate) - (team_b.free_throw_rate - team_a.foul_rate)),
            "size_edge": weights["size_edge"] * (team_a.size - team_b.size),
            "experience_edge": weights["experience_edge"] * (team_a.experience - team_b.experience),
            "continuity_edge": weights["continuity_edge"] * (team_a.continuity - team_b.continuity),
            "bench_edge": weights["bench_edge"] * (team_a.bench_depth - team_b.bench_depth),
        }
        round_pressure = 1.0 + ((round_index - 1) * 0.025)
        return sum(features.values()) * round_pressure

    def _style_signal(self, team_a: TeamProfile, team_b: TeamProfile) -> float:
        return (
            1.10
            * (
                (team_a.three_point_rate - team_b.three_point_defense)
                - (team_b.three_point_rate - team_a.three_point_defense)
            )
            + 0.90
            * (
                (team_a.turnover_creation - team_b.turnover_rate)
                - (team_b.turnover_creation - team_a.turnover_rate)
            )
            + 0.80
            * (
                (team_a.offensive_rebounding - team_b.defensive_rebounding)
                - (team_b.offensive_rebounding - team_a.defensive_rebounding)
            )
            + 0.65
            * ((team_a.free_throw_rate - team_b.foul_rate) - (team_b.free_throw_rate - team_a.foul_rate))
            + 0.25 * (team_a.size - team_b.size)
            + 0.18 * (team_a.experience - team_b.experience)
            + 0.12 * (team_a.continuity - team_b.continuity)
            + 0.10 * (team_a.bench_depth - team_b.bench_depth)
        )
