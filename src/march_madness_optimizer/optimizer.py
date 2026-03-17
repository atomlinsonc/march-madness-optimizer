from __future__ import annotations

import random
from dataclasses import dataclass

from .bracket import Tournament
from .models import ProbabilityModel


@dataclass(frozen=True)
class OptimizationResult:
    bracket: dict[str, str]
    expected_score: float
    win_rate: float
    top_decile_rate: float
    strategy: str


class PoolOptimizer:
    def __init__(self, tournament: Tournament, model: ProbabilityModel, seed: int = 2026) -> None:
        self.tournament = tournament
        self.model = model
        self.seed = seed

    def optimize(
        self,
        pool_size: int,
        entries: int = 200,
        simulations: int = 1500,
    ) -> OptimizationResult:
        rng = random.Random(self.seed)
        if pool_size <= 25:
            leverage = 0.02
            strategy = "small-pool"
        elif pool_size <= 250:
            leverage = 0.10
            strategy = "medium-pool"
        else:
            leverage = 0.22
            strategy = "large-pool"

        candidates = [self.tournament.chalk_bracket(self.model)]
        for _ in range(entries - 1):
            candidates.append(
                self.tournament.sample_bracket(
                    model=self.model,
                    rng=rng,
                    public=False,
                    leverage=leverage,
                )
            )

        outcomes = [self.tournament.simulate_outcome(self.model, rng) for _ in range(simulations)]
        opponent_brackets = [
            [
                self.tournament.sample_bracket(
                    model=self.model,
                    rng=rng,
                    public=True,
                    leverage=0.0,
                )
                for _ in range(max(pool_size - 1, 1))
            ]
            for _ in range(simulations)
        ]

        best_result: OptimizationResult | None = None
        for candidate in candidates:
            total_score = 0
            wins = 0
            top_decile = 0
            for simulation_index, actual in enumerate(outcomes):
                candidate_score = self.tournament.score_bracket(candidate, actual)
                total_score += candidate_score
                opponent_scores = [
                    self.tournament.score_bracket(opponent, actual)
                    for opponent in opponent_brackets[simulation_index]
                ]
                if candidate_score > max(opponent_scores):
                    wins += 1
                combined = sorted(opponent_scores + [candidate_score], reverse=True)
                placement = combined.index(candidate_score) + 1
                if placement <= max(1, pool_size // 10):
                    top_decile += 1
            result = OptimizationResult(
                bracket=candidate,
                expected_score=total_score / simulations,
                win_rate=wins / simulations,
                top_decile_rate=top_decile / simulations,
                strategy=strategy,
            )
            if best_result is None or self._dominates(result, best_result):
                best_result = result
        assert best_result is not None
        return best_result

    @staticmethod
    def _dominates(left: OptimizationResult, right: OptimizationResult) -> bool:
        left_value = (left.win_rate * 0.6) + (left.top_decile_rate * 0.25) + (left.expected_score * 0.15 / 192.0)
        right_value = (right.win_rate * 0.6) + (right.top_decile_rate * 0.25) + (right.expected_score * 0.15 / 192.0)
        return left_value > right_value
