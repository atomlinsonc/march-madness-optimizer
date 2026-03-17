from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import ProbabilityModel, TeamProfile


ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}


@dataclass(frozen=True)
class GameNode:
    game_id: str
    round_index: int
    slot_a: str
    slot_b: str
    region: str


class Tournament:
    def __init__(self, teams: dict[str, TeamProfile], games: list[GameNode], scoring: dict[int, int] | None = None) -> None:
        self.teams = teams
        self.games = games
        self.scoring = scoring or {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}

    @classmethod
    def from_file(cls, path: str | Path) -> "Tournament":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        teams = {
            item["name"]: TeamProfile(
                name=item["name"],
                region=item["region"],
                seed=item["seed"],
                adj_o=item["adj_o"],
                adj_d=item["adj_d"],
                tempo=item["tempo"],
                last10_adj_em=item["last10_adj_em"],
                non_garbage_adj_em=item["non_garbage_adj_em"],
                elo=item["elo"],
                network=item["network"],
                resume=item["resume"],
                three_point_rate=item["three_point_rate"],
                three_point_defense=item["three_point_defense"],
                turnover_rate=item["turnover_rate"],
                turnover_creation=item["turnover_creation"],
                offensive_rebounding=item["offensive_rebounding"],
                defensive_rebounding=item["defensive_rebounding"],
                free_throw_rate=item["free_throw_rate"],
                foul_rate=item["foul_rate"],
                size=item["size"],
                experience=item["experience"],
                continuity=item["continuity"],
                bench_depth=item["bench_depth"],
                coach_score=item["coach_score"],
                travel_index=item.get("travel_index", 0.0),
                rest_edge=item.get("rest_edge", 0.0),
                availability=item.get("availability", 1.0),
                market_power=item.get("market_power", 0.0),
                public_pick_rate=item.get("public_pick_rate", 0.0),
            )
            for item in payload["teams"]
        }
        games = [GameNode(**item) for item in payload["games"]]
        scoring = {int(key): value for key, value in payload.get("scoring", {}).items()}
        return cls(teams=teams, games=games, scoring=scoring or None)

    def ordered_teams(self) -> list[TeamProfile]:
        return sorted(self.teams.values(), key=lambda team: (team.region, team.seed, team.name))

    def matchup_matrix(self, model: ProbabilityModel) -> list[dict[str, object]]:
        return model.all_matchups(self.ordered_teams())

    def chalk_bracket(self, model: ProbabilityModel) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for game in self.games:
            team_a = self._resolve_slot(game.slot_a, resolved)
            team_b = self._resolve_slot(game.slot_b, resolved)
            probability = model.probability(team_a, team_b, round_index=game.round_index)
            resolved[game.game_id] = team_a.name if probability >= 0.5 else team_b.name
        return resolved

    def sample_bracket(
        self,
        model: ProbabilityModel,
        rng: random.Random,
        public: bool = False,
        leverage: float = 0.0,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for game in self.games:
            team_a = self._resolve_slot(game.slot_a, resolved)
            team_b = self._resolve_slot(game.slot_b, resolved)
            base = model.probability(team_a, team_b, round_index=game.round_index)
            adjusted = self._public_probability(team_a, team_b, base) if public else base
            if leverage:
                adjusted = self._apply_leverage(team_a, team_b, adjusted, leverage)
            resolved[game.game_id] = team_a.name if rng.random() < adjusted else team_b.name
        return resolved

    def simulate_outcome(self, model: ProbabilityModel, rng: random.Random) -> dict[str, str]:
        return self.sample_bracket(model=model, rng=rng, public=False, leverage=0.0)

    def score_bracket(self, picks: dict[str, str], actual: dict[str, str]) -> int:
        score = 0
        for game in self.games:
            if picks.get(game.game_id) == actual.get(game.game_id):
                score += self.scoring[game.round_index]
        return score

    def advancement_summary(self, simulations: Iterable[dict[str, str]]) -> dict[str, dict[str, float]]:
        counts: dict[str, dict[int, int]] = {name: {round_index: 0 for round_index in self.scoring} for name in self.teams}
        total = 0
        for bracket in simulations:
            total += 1
            for game in self.games:
                winner = bracket[game.game_id]
                counts[winner][game.round_index] += 1
        summary: dict[str, dict[str, float]] = {}
        for team_name, per_round in counts.items():
            summary[team_name] = {
                ROUND_NAMES[round_index]: round(per_round[round_index] / max(total, 1), 4)
                for round_index in sorted(per_round)
            }
        return summary

    def _resolve_slot(self, slot: str, resolved: dict[str, str]) -> TeamProfile:
        team_name = resolved.get(slot, slot)
        return self.teams[team_name]

    def _public_probability(self, team_a: TeamProfile, team_b: TeamProfile, base_probability: float) -> float:
        market_bias = 0.10 * (team_a.seed < team_b.seed) - 0.10 * (team_b.seed < team_a.seed)
        popularity_bias = 0.55 * (team_a.public_pick_rate - team_b.public_pick_rate)
        adjusted = base_probability + market_bias + popularity_bias
        return min(max(adjusted, 0.05), 0.95)

    def _apply_leverage(self, team_a: TeamProfile, team_b: TeamProfile, probability: float, leverage: float) -> float:
        edge_a = probability - team_a.public_pick_rate
        edge_b = (1.0 - probability) - team_b.public_pick_rate
        shifted = probability + leverage * (edge_a - edge_b) * 0.4
        return min(max(shifted, 0.03), 0.97)
