from __future__ import annotations

import json
import random
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Iterable

from .models import ProbabilityModel, TeamProfile


REGION_CODES = {
    "East": "E",
    "West": "W",
    "South": "S",
    "Midwest": "M",
}


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
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        scraped_metrics: dict[str, dict[str, float]] = {}
        metrics_path_value = payload.get("metrics_path")
        if isinstance(metrics_path_value, str):
            metrics_path = file_path.parent / metrics_path_value
            if metrics_path.exists():
                scraped_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if "regions" in payload:
            teams = cls._build_teams_from_regions(payload["regions"], scraped_metrics)
            games = cls._build_standard_games(payload["regions"])
        else:
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

    @staticmethod
    def _build_teams_from_regions(
        regions: dict[str, list[dict[str, object]]],
        scraped_metrics: dict[str, dict[str, float]],
    ) -> dict[str, TeamProfile]:
        teams: dict[str, TeamProfile] = {}
        for region_name, entries in regions.items():
            for entry in entries:
                team = Tournament._synthesize_team_profile(region_name, entry, scraped_metrics)
                teams[team.name] = team
        return teams

    @staticmethod
    def _synthesize_team_profile(
        region_name: str,
        entry: dict[str, object],
        scraped_metrics: dict[str, dict[str, float]],
    ) -> TeamProfile:
        name = str(entry["name"])
        seed = int(entry["seed"])
        overall_seed = float(entry.get("overall_seed", 68))
        wins = int(entry.get("wins", 20))
        losses = int(entry.get("losses", 12))
        record_games = max(wins + losses, 1)
        win_pct = wins / record_games
        quality = 69.0 - overall_seed
        profile_seed = sum((index + 1) * ord(character) for index, character in enumerate(name))
        style_a = ((profile_seed % 19) - 9) / 100.0
        style_b = (((profile_seed // 19) % 19) - 9) / 100.0
        metrics = scraped_metrics.get("teams", {}).get(name, {})
        net_rank = float(metrics.get("net_rank", 70 - quality))
        ap_rank = metrics.get("ap_rank")
        bpi_rank = metrics.get("bpi_rank")
        odds_values = metrics.get("title_odds_american")
        composite_ranks: list[tuple[float, float]] = [(net_rank, 0.45), (overall_seed, 0.20)]
        if isinstance(ap_rank, (int, float)):
            composite_ranks.append((float(ap_rank), 0.20))
        if isinstance(bpi_rank, (int, float)):
            composite_ranks.append((float(bpi_rank), 0.15))
        weight_total = sum(weight for _, weight in composite_ranks)
        composite_rank = sum(rank * weight for rank, weight in composite_ranks) / max(weight_total, 1e-6)
        strength = max(2.0, 40.0 - (composite_rank * 0.5))
        market_implied = Tournament._average_implied_probability(odds_values)
        market_boost = log((market_implied + 1e-6) / (1.0 - market_implied + 1e-6)) if market_implied else 0.0

        adj_em = max(1.0, 1.5 + (strength * 0.42) + ((win_pct - 0.5) * 6.0) + (market_boost * 1.6))
        adj_o = 100.0 + (strength * 0.20) + ((profile_seed % 11) * 0.24) + (market_boost * 0.55)
        adj_d = adj_o - adj_em
        public_base = max(0.01, min(0.92, (17.0 - seed) / 20.0))
        public_pick_rate = min(max(market_implied * 3.2 if market_implied else public_base, 0.01), 0.92)
        return TeamProfile(
            name=name,
            region=region_name,
            seed=seed,
            adj_o=round(adj_o, 2),
            adj_d=round(adj_d, 2),
            tempo=round(65.0 + ((profile_seed % 8) * 0.85), 2),
            last10_adj_em=round(adj_em * 0.96, 2),
            non_garbage_adj_em=round(adj_em * 1.03, 2),
            elo=round(1495.0 + (strength * 4.4) + ((win_pct - 0.5) * 85.0) + (market_boost * 12.0), 2),
            network=round((strength * 0.32) + ((win_pct - 0.5) * 10.0), 2),
            resume=round((strength * 0.28) + ((win_pct - 0.5) * 14.0), 2),
            three_point_rate=round(0.31 + style_a + (0.008 * (seed <= 4)), 3),
            three_point_defense=round(0.34 - (strength / 550.0) + style_b, 3),
            turnover_rate=round(0.18 - (strength / 850.0) + (style_b / 2.5), 3),
            turnover_creation=round(0.15 + (strength / 720.0) + (style_a / 2.2), 3),
            offensive_rebounding=round(0.27 + (strength / 430.0) - (style_a / 2.8), 3),
            defensive_rebounding=round(0.68 + (strength / 390.0) - (style_b / 3.0), 3),
            free_throw_rate=round(0.28 + (strength / 620.0) + (style_a / 3.0), 3),
            foul_rate=round(0.31 - (strength / 710.0) + (style_b / 3.0), 3),
            size=round(6.9 + (strength / 22.0) + (style_b * 3.5), 2),
            experience=round(5.4 + ((profile_seed % 7) * 0.35), 2),
            continuity=round(5.0 + (((profile_seed // 7) % 7) * 0.38), 2),
            bench_depth=round(5.6 + (((profile_seed // 13) % 7) * 0.33), 2),
            coach_score=round(5.8 + ((strength / 14.0) + ((profile_seed % 5) * 0.22)), 2),
            travel_index=round(((profile_seed % 9) * 0.03), 2),
            rest_edge=0.0,
            availability=float(entry.get("availability", 1.0)),
            market_power=round((adj_em * 0.65) + (market_implied * 50.0), 2),
            public_pick_rate=round(public_pick_rate, 3),
        )

    @staticmethod
    def _average_implied_probability(american_odds: object) -> float:
        if isinstance(american_odds, (int, float)):
            american_values = [float(american_odds)]
        elif isinstance(american_odds, list):
            american_values = [float(item) for item in american_odds if isinstance(item, (int, float))]
        else:
            american_values = []
        if not american_values:
            return 0.0
        implied_values: list[float] = []
        for value in american_values:
            if value > 0:
                implied_values.append(100.0 / (value + 100.0))
            elif value < 0:
                implied_values.append((-value) / ((-value) + 100.0))
        return sum(implied_values) / max(len(implied_values), 1)

    @staticmethod
    def _build_standard_games(regions: dict[str, list[dict[str, object]]]) -> list[GameNode]:
        seed_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
        region_finals: dict[str, str] = {}
        games: list[GameNode] = []

        for region_name in ("East", "West", "South", "Midwest"):
            entries = {int(item["seed"]): str(item["name"]) for item in regions[region_name]}
            code = REGION_CODES[region_name]
            round_one_ids: list[str] = []
            for offset, (high_seed, low_seed) in enumerate(seed_pairs, start=1):
                game_id = f"{code}{offset}"
                round_one_ids.append(game_id)
                games.append(
                    GameNode(
                        game_id=game_id,
                        round_index=1,
                        slot_a=entries[high_seed],
                        slot_b=entries[low_seed],
                        region=region_name,
                    )
                )
            round_two_ids: list[str] = []
            for index in range(0, len(round_one_ids), 2):
                game_id = f"{code}{9 + (index // 2)}"
                round_two_ids.append(game_id)
                games.append(
                    GameNode(
                        game_id=game_id,
                        round_index=2,
                        slot_a=round_one_ids[index],
                        slot_b=round_one_ids[index + 1],
                        region=region_name,
                    )
                )
            sweet_sixteen_ids: list[str] = []
            for index in range(0, len(round_two_ids), 2):
                game_id = f"{code}{13 + (index // 2)}"
                sweet_sixteen_ids.append(game_id)
                games.append(
                    GameNode(
                        game_id=game_id,
                        round_index=3,
                        slot_a=round_two_ids[index],
                        slot_b=round_two_ids[index + 1],
                        region=region_name,
                    )
                )
            regional_final_id = f"{code}15"
            games.append(
                GameNode(
                    game_id=regional_final_id,
                    round_index=4,
                    slot_a=sweet_sixteen_ids[0],
                    slot_b=sweet_sixteen_ids[1],
                    region=region_name,
                )
            )
            region_finals[region_name] = regional_final_id

        games.append(GameNode(game_id="F1", round_index=5, slot_a=region_finals["East"], slot_b=region_finals["West"], region="Final Four"))
        games.append(GameNode(game_id="F2", round_index=5, slot_a=region_finals["South"], slot_b=region_finals["Midwest"], region="Final Four"))
        games.append(GameNode(game_id="C1", round_index=6, slot_a="F1", slot_b="F2", region="Championship"))
        return games

    def ordered_teams(self) -> list[TeamProfile]:
        return sorted(self.teams.values(), key=lambda team: (team.region, team.seed, team.name))

    def matchup_matrix(self, model: ProbabilityModel) -> list[dict[str, object]]:
        return self.tournament_matchup_matrix(model)

    def tournament_matchup_matrix(self, model: ProbabilityModel) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        winner_cache: dict[str, dict[str, float]] = {}
        for game in self.games:
            left_distribution = self._slot_distribution(game.slot_a, model, winner_cache)
            right_distribution = self._slot_distribution(game.slot_b, model, winner_cache)
            for left_team, left_reach_probability in left_distribution.items():
                for right_team, right_reach_probability in right_distribution.items():
                    matchup_probability = left_reach_probability * right_reach_probability
                    if matchup_probability <= 0:
                        continue
                    team_a = self.teams[left_team]
                    team_b = self.teams[right_team]
                    team_a_win_probability = model.probability(team_a, team_b, round_index=game.round_index)
                    rows.append(
                        {
                            "game_id": game.game_id,
                            "round_index": game.round_index,
                            "round_name": ROUND_NAMES[game.round_index],
                            "region": game.region,
                            "team_a": left_team,
                            "team_b": right_team,
                            "team_a_reach_probability": round(left_reach_probability, 4),
                            "team_b_reach_probability": round(right_reach_probability, 4),
                            "matchup_probability": round(matchup_probability, 4),
                            "team_a_win_probability": round(team_a_win_probability, 4),
                            "team_b_win_probability": round(1.0 - team_a_win_probability, 4),
                        }
                    )
        rows.sort(
            key=lambda row: (
                int(row["round_index"]),
                -float(row["matchup_probability"]),
                str(row["game_id"]),
                str(row["team_a"]),
                str(row["team_b"]),
            )
        )
        return rows

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

    def exact_advancement_summary(self, model: ProbabilityModel) -> dict[str, dict[str, float]]:
        winner_cache: dict[str, dict[str, float]] = {}
        counts: dict[str, dict[int, float]] = {name: {round_index: 0.0 for round_index in self.scoring} for name in self.teams}
        for game in self.games:
            winner_distribution = self._winner_distribution(game, model, winner_cache)
            for team_name, probability in winner_distribution.items():
                counts[team_name][game.round_index] += probability
        summary: dict[str, dict[str, float]] = {}
        for team_name, per_round in counts.items():
            summary[team_name] = {
                ROUND_NAMES[round_index]: round(per_round[round_index], 4)
                for round_index in sorted(per_round)
            }
        return summary

    def _resolve_slot(self, slot: str, resolved: dict[str, str]) -> TeamProfile:
        team_name = resolved.get(slot, slot)
        return self.teams[team_name]

    def _slot_distribution(
        self,
        slot: str,
        model: ProbabilityModel,
        winner_cache: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        if slot in self.teams:
            return {slot: 1.0}
        game = next(game for game in self.games if game.game_id == slot)
        return self._winner_distribution(game, model, winner_cache)

    def _winner_distribution(
        self,
        game: GameNode,
        model: ProbabilityModel,
        winner_cache: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        if game.game_id in winner_cache:
            return winner_cache[game.game_id]
        left_distribution = self._slot_distribution(game.slot_a, model, winner_cache)
        right_distribution = self._slot_distribution(game.slot_b, model, winner_cache)
        winners: dict[str, float] = {}
        for left_team, left_reach_probability in left_distribution.items():
            for right_team, right_reach_probability in right_distribution.items():
                matchup_probability = left_reach_probability * right_reach_probability
                if matchup_probability <= 0:
                    continue
                team_a = self.teams[left_team]
                team_b = self.teams[right_team]
                team_a_win_probability = model.probability(team_a, team_b, round_index=game.round_index)
                winners[left_team] = winners.get(left_team, 0.0) + (matchup_probability * team_a_win_probability)
                winners[right_team] = winners.get(right_team, 0.0) + (matchup_probability * (1.0 - team_a_win_probability))
        winner_cache[game.game_id] = winners
        return winners

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
