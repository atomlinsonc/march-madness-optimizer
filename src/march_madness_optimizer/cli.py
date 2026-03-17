from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bracket import ROUND_NAMES, Tournament
from .models import ProbabilityModel
from .optimizer import PoolOptimizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="March Madness matchup model and bracket optimizer")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[2] / "data" / "sample_2026_bracket.json"),
        help="Path to tournament JSON data",
    )
    parser.add_argument("--pool-size", type=int, default=25, help="Number of entries in the pool")
    parser.add_argument("--entries", type=int, default=200, help="Number of candidate brackets to search")
    parser.add_argument("--simulations", type=int, default=1500, help="Number of tournament simulations")
    parser.add_argument("--matchups-out", default="", help="Optional path for pairwise matchup probabilities JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tournament = Tournament.from_file(args.input)
    model = ProbabilityModel()

    if args.matchups_out:
        payload = tournament.matchup_matrix(model)
        Path(args.matchups_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    optimizer = PoolOptimizer(tournament=tournament, model=model)
    result = optimizer.optimize(
        pool_size=args.pool_size,
        entries=args.entries,
        simulations=args.simulations,
    )
    advancement = tournament.exact_advancement_summary(model)
    championship_label = ROUND_NAMES[max(tournament.scoring)]
    title_odds = sorted(
        ((team, rounds[championship_label]) for team, rounds in advancement.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    output = {
        "strategy": result.strategy,
        "expected_score": round(result.expected_score, 2),
        "win_rate": round(result.win_rate, 4),
        "top_decile_rate": round(result.top_decile_rate, 4),
        "title_odds_top_10": [{team: probability} for team, probability in title_odds],
        "advancement_odds_top_16": {
            team: advancement[team] for team, _ in sorted(
                ((team, rounds[championship_label]) for team, rounds in advancement.items()),
                key=lambda item: item[1],
                reverse=True,
            )[:16]
        },
        "recommended_bracket": result.bracket,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
