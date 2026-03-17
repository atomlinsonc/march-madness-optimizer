# March Madness Optimizer

This project implements the two linked systems you outlined:

1. A matchup probability engine that outputs `P(Team A beats Team B)` for every possible pairing.
2. A pool-aware bracket simulator and optimizer that converts those probabilities into an actual bracket strategy.

The current build is stdlib-only and uses the official 2026 bracket structure plus a scraped public metrics layer so it runs immediately without external dependencies.

## What It Includes

- blended matchup probabilities from efficiency, Elo, network, resume, market, style, and context features
- full pairwise matchup matrix generation
- bracket-tree simulation
- pool-size-aware candidate search
- public-pick leverage handling
- exact advancement and title odds from the bracket tree

## Usage

From the project root:

```powershell
$env:PYTHONPATH="C:\Users\atoml\Documents\march-madness-optimizer\src"
py -3 -m march_madness_optimizer.cli --pool-size 25 --entries 250 --simulations 2000
```

Optional matchup export:

```powershell
$env:PYTHONPATH="C:\Users\atoml\Documents\march-madness-optimizer\src"
py -3 -m march_madness_optimizer.cli --matchups-out matchup_probs.json
```

## Files

- `src/march_madness_optimizer/models.py`: matchup model
- `src/march_madness_optimizer/bracket.py`: bracket tree and simulation
- `src/march_madness_optimizer/optimizer.py`: pool strategy search
- `src/march_madness_optimizer/cli.py`: command-line entry point
- `data/sample_2026_bracket.json`: official 2026 bracket structure and field
- `data/scraped_metrics_2026.json`: scraped public ranking and market inputs

## Notes

- The bracket structure is real, but the matchup engine is still a public-data composite rather than a historically trained stack.
- The next serious step is frozen-time historical training and calibrated stacking from held-out seasons.
