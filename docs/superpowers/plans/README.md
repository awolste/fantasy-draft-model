# Plan Index

**Read this first if you are picking up this project.**

1. Read the design spec: [`../specs/2026-08-02-ff-draft-2026-design.md`](../specs/2026-08-02-ff-draft-2026-design.md). It contains the league configuration, the approach and why alternatives were rejected, and the validation gate. Do not re-litigate decisions recorded there without a reason.
2. Find the first stage below not marked **Complete**.
3. If its plan exists, execute it. If it does not exist yet, write it using the `superpowers:writing-plans` skill, informed by what the previous stage actually produced.

Each stage produces working, independently testable software. Stages are written one at a time, on purpose: writing Stage 3's plan before Stage 2's models exist would lock in decisions about model internals before the real data has been seen.

## The league-specific edge (Stage 3, confirmed by the owner)

**This league drafts quarterbacks later than consensus ADP** — fitted positional bias −0.100 log-reach, the strongest negative of any position. That is backwards from what the scoring implies: the league uses **6-point passing touchdowns** while essentially every public ranking assumes 4, so QBs are genuinely worth more here than the rankings leaguemates are reading.

The owner confirmed the cause: the league drafts off generic ESPN rankings and is not especially sophisticated about adjusting for scoring settings. This is therefore a real, persistent inefficiency rather than a one-season artifact.

It should surface on its own without hand-coding, because two independent parts of the model already encode it: Stage 2's player distributions anchor QB means to **actual league scoring computed from raw stat lines**, and Stage 3's opponent model says QBs survive longer than that value warrants. If Stage 3's recommender does *not* show QB value later than consensus, something is wrong with how those two pieces compose — treat that as a bug signal.

Also note the deep-pick censoring caveat: `adp_history` is a top-N export that does not reach this league's deepest bench stashes, so 7.85% of picks fail to match, concentrated in DST (19.2%), K (15.8%) and TE (15.4%). That censors the latest picks for those positions and inflates their fitted "earlier than ADP" reading. Directions are trustworthy; magnitudes for those three are not.

## Stage 2 findings that Stage 3 depends on

Measured, not assumed. Do not re-derive these; do check them if a Stage 3 result looks strange.

- **Realistic edge is small.** In the 2024 calibration, the best-drafted roster had ~16% championship probability against a 10% baseline, and the actual champion drew 10.3%. *(Corrected during Stage 3 Task 1: fallback/replacement players were having an availability model applied on top of a replacement level already conditioned on streaming a healthy player, double-counting injury risk. The champion's figure was 9.1% before the fix.)* A 60% playoff rate plus a 3-round bracket washes out most roster edge. If Stage 3's backtest claims a large edge over ADP-following, suspect overfitting rather than success.
- **Correlation is not material — measured, not assumed.** Real weekly correlations: QB1–WR1 same team **0.372**, QB1–TE1 0.281, opposing-team skill totals 0.187, QB1–RB1 0.070, and WR1–WR2 same team only **0.034** (shared offense is cancelled by target competition). Injecting a calibrated copula moves the stacked-vs-diversified championship gap by **−0.42pp** against a 0.22pp SE bound. The production simulator stays on independent sampling. Note the sign: correlation *reduces* a stacked roster's edge, because clustered bad weeks cost more than clustered good weeks when 60% of the league makes the playoffs. See `scripts/correlation_impact.py`.
- **Replacement level is an order statistic, not a mean.** What you get off waivers is the best available option each week, not the average deep player. Measured (points/week): QB 18.3, RB 9.6, WR 9.9, TE 8.1, K 7.9 — against starter medians of 24.3 / 15.9 / 16.4 / 12.4 / 9.0. A rostered RB starter is worth ~6.4/week over waivers and a WR ~6.5 — nearly identical, which bears directly on the 0RB question.
- **Availability must be persistent, not independent.** P(out next week | out this week) is **75.5%**, versus 7.1% if available. Modeled as a two-state Markov chain. Independent draws would treat a season-ending injury as 14 coin flips.
- **Lineups are set from realized scores**, which no real manager can do. Quantified hindsight bias on real 2024 rosters: max 2.5pp swing in championship probability. Modest, but those rosters were not deliberately variance-stacked, so this does not bound the effect on boom/bust builds.
- **Rank curves are only valid inside their fitted ADP range.** Deep players have no ADP, so beyond the boundary the projection comes from empirical deep-rank data, and `RankCurve.mean_for_rank` raises past `max_supported_rank` rather than extrapolating. An earlier unconstrained power law put QB50 at 16.5 ppg against ~3.3 observed.

## Stages

| # | Plan | Produces | Status |
|---|---|---|---|
| 1 | [`2026-08-02-data-foundation.md`](2026-08-02-data-foundation.md) | Every input pulled, ID-matched, scored under league rules, validated | **Complete** — 97 tests, 9 datasets |
| 2 | [`2026-08-02-player-and-season-model.md`](2026-08-02-player-and-season-model.md) | Weekly points distributions per player; season simulator taking 10 rosters to a champion | **Complete** — 210 tests, 1,000 seasons in 0.17s |
| 3 | [`2026-08-03-opponent-model-and-optimizer.md`](2026-08-03-opponent-model-and-optimizer.md) | Opponent draft model, draft rollout, recommender, backtest, and the 0RB vs. Hero RB answer | **Ready to start** |
| 4 | *not yet written* — `live-assistant.md` | Streamlit draft-day tool with type-ahead pick entry | Not started |

**Stage 4 should not begin before August 2026**, when ESPN's 2026 endpoints are live and live-draft behavior can be tested against reality.

## Context that is easy to lose

These are the non-obvious constraints that took a conversation to establish. They are recorded in the spec in full; this is the short list so nobody rediscovers them the hard way.

- **6-point passing TDs.** Nearly every public ranking and projection assumes 4. All fantasy points are computed from raw stat lines for this reason. Never consume a precomputed fantasy total.
- **The waiver wire is ordinary, not rich.** 10 teams × 18 roster spots = 180 rostered players, the same depth as a 12-team league with 15 spots. An early assumption that a 10-team league implies a shallow pool was wrong.
- **60% of teams make the playoffs**, so regular-season wins are cheap except for the top-2 bye. The objective is championship probability, not points and not wins. Variance has positive value here.
- **Managers are keyed by SWID**, never team ID or team name. Membership is mostly stable 2018–2025 but some managers left and joined; per-manager tendencies must be shrunk toward the league prior in proportion to seasons on record.
- **Defenses go earlier than ADP here** because benches are deep. Used as a check that the opponent model fitted correctly.
- **Draft slot is 8** in a 10-team snake, so picks come in back-to-back pairs near the turn (8 and 13) and are better planned as pairs.
- **The backtest is a gate, not a formality.** If the engine does not beat ADP-following on held-out 2025, it does not work. A large apparent edge means overfitting, not success.
- **K and DST scoring settings are now known** and recorded in the spec. Two bands are *neutral, not missing*: points allowed 18–27 and yards allowed 300–349 both score 0. Encode them explicitly so a lookup cannot silently fall through. Also note FG 50–59 and 60+ both score 5 — distance past 50 buys nothing in this league.
- **DST scoring cannot come from `weekly_stats`.** Points- and yards-allowed are team-game properties, not player rows, so Stage 2 needs a separate team-level ingest. This is the largest unknown in Stage 2.
- **Kickers are entirely absent from the model, not merely approximate.** They match the ID crosswalk at **0.0%** (nflverse's `load_ff_playerids` has essentially no kicker coverage by name) *and* score 0.0 fantasy points (no kicking terms in `ScoringRules`). Two independent gaps stacking on the same position. Any Stage 2 approach must treat K as a modeled constant, not as a player looked up by ID.
- **The collision-review rule is too noisy as built.** `validate.py` flags collision groups whose top-2 `draft_year` values are within 10 years; that catches 162 of 363 fantasy-relevant groups (45%). Many are not father/son pairs but duplicate nflverse entries where the discarded row carries a `draft_year=0` sentinel. Exclude sentinel rows before applying the rule to make the output actionable.
- **K and DST currently score zero.** Confirmed against real data: `ScoringRules` has no kicking or defensive terms and DST is not ingested, so every kicker scores 0.0 and defenses have no rows. The league starts both. This must be resolved before Stage 2's season simulator, and it needs the owner's exact ESPN K/DST scoring settings. See "Open questions" in the spec.
- **Never commit raw ESPN payloads.** ESPN responses identify real people: every league member's account GUID (which for the authenticated user *is* the `SWID` cookie), their real first and last names, ESPN usernames, and team names. A fixture was committed with all of that intact during Task 6 and had to be scrubbed from history. Run `scripts/sanitize_espn_fixture.py` on anything derived from an ESPN response before it goes near git. Raw payloads belong only in the gitignored `data/raw/`.
- **Roster size changed mid-history: drafts were 17 rounds (170 picks) in 2018–2022 and 18 rounds (180 picks) from 2023 on.** Any check assuming a fixed 180 picks per season will fail on the early years, and any model treating roster size as constant across the 8 seasons is wrong. All eight seasons have 10 teams throughout.
- **ADP comes from Fantasy Football Calculator, not FantasyPros.** FantasyPros fences ADP behind registration (returns 5 rows logged-out); its ECR rankings are still used and work fine. FFC's `teams` parameter does **not** filter — `teams=10` and `teams=12` return identical data, so there is no 10-team-specific ADP available anywhere. FFC covers 2018–2024 and 2026 but **has no 2025 data**, which is why the backtest holds out 2024 rather than 2025.
- **Silent-corruption failure modes are the main risk in this project.** Bad ID matching, an unresolved upstream column, or a miscalibrated opponent model do not crash — they produce confident, plausible, wrong recommendations. Prefer loud failures over defaults everywhere in the data layer. Two instances have already been caught and fixed this way.
