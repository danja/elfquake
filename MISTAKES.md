# Mistakes

Log of what went wrong, why, and what prevents a repeat. Newest first.

## 2026-08-21 — Used a null control that carried a hundred times the evidence

**What happened.** The timestamp-shuffle control (`permute-spatial-targets`) was
the standing temporal null from item 8 onward, and items 8 and 93 drew a
conclusion from it: all five shuffled controls beat real chronological order, so
destroying temporal structure makes the task easier, which is the opposite of a
predictive signal. Counting held-out label transitions showed the controls were
not solving the same problem. Real order carried **1** transition in `era_0` and
**19** in `era_3`; the shuffled controls carried **94-119** and **315-342**. The
control was scored on up to a hundred times the evidence of the run it was
supposed to null, so the comparison was meaningless in both directions. Under a
matched null the pattern disappears entirely.

**Root cause.** Shuffling is a valid null for independent rows. These rows are
not independent: anchors are 30 minutes apart against a 1-7 day horizon, so each
cell's labels are a few long runs, and shuffling the order converts them into
many short ones. Every new run boundary is a label transition. The control
destroyed the autocorrelation along with the signal, and the autocorrelation was
most of what the labels contained. Nothing reported the transition count, so the
asymmetry was invisible for four rounds of results.

**Prevention.** `evaluate-italy-spatial-shift-controls.sh` replaces it with a
circular shift that preserves each cell's sequence length and positive count
exactly and its run structure apart from the wrap seam. It prints
`transitions: N before, M after`, and a difference beyond two or three means the
control is unusable. `_stratified_metrics` reports `label_transitions` for every
run including controls, so a control and its run can always be compared on
evidence before their scores are compared. When choosing a null for
autocorrelated data, ask what the null does to the effective sample size, not
just to the signal.

## 2026-08-21 — Evaluated against a four-day-stale event catalog

**What happened.** The item-104 evaluation was run, reported, and written up
against an INGV catalog whose last event was 2026-08-17. Refreshing it added
**22 events between 2026-08-17 and 2026-08-21**, all inside the held-out
partition, every one of which had been labeled a non-event. The corrected run
moved `era_3` held-out transitions from 10 to 19, scoreable cells from 6 to 8,
and `seismic_only` stratified from `0.652961` to `0.552836`. The `0.652961` was
the largest apparent VLF-era result in the project and it was an artifact of
missing events.

**Root cause.** Two things, and either alone would have been caught.

The live `elfquake-prospective.timer` fires every 30 minutes and looks like a
data collector, but `elfquake-prospective.service` only extracts VLF image
features and updates the window table. It never fetches INGV. The event catalog
advances only when a human runs `refresh-prospective-labels.sh`. A timer that
had run four minutes earlier was taken as evidence the data was current.

`build-italy-spatial-vlf-targets.sh` then defaulted `CATALOG_END` to
`$(date -u)`, which asserts the catalog covers up to this moment. The maturity
guard was therefore certifying coverage from wall-clock time rather than from
the catalog, so anchors in the staleness gap were declared mature and labeled
negative — absence of events in a catalog that had stopped.

**Prevention.** `CATALOG_END` now defaults to the catalog's own
`max(ingested_at_utc)`, which is when the fetch actually happened, so the guard
cannot certify coverage that does not exist. The
`elfquake-fixed-cell-evaluation` skill opens with the refresh and a freshness
check, and pins `AS_OF`/`CATALOG_END` so a reported number can be regenerated.
Before any evaluation, print `max(ingested_at_utc)` and compare it with now; a
running timer is not evidence of a current catalog. The live collector now runs
`refresh-ingv-events.sh` as its first step, so the catalog advances every 30
minutes instead of only when someone remembers (item 105(a), installed and
verified 2026-08-21).

## 2026-08-21 — Quoted a whole-record diagnostic for a per-era evaluation

**What happened.** Item 103 recommended a 1-day / M≥2.0 target design on the
strength of `27` held-out label transitions, and item 104 was written to test
whether `27` was enough. Neither number was ever available to a real run: item
96 forbids pooling `era_0` and `era_3`, so every evaluation happens inside one
era, and the split leaves **1** held-out transition in `era_0` and **19** in
`era_3`. The design was recommended, built, and scheduled for evaluation
against an evidence count no run could see.

**Root cause.** `diagnose_spatial_target_design` measures a CSV as one record.
The evaluator splits by era first and then takes a held-out fraction of each.
Two tools computed "held-out transitions" under different partitions and only
one label was used for both. The whole-record figure is not wrong; it answers a
question no run asks.

**Prevention.** The transition count is now computed inside the evaluator —
`_stratified_metrics` emits `label_transitions` per stratum and in the summary,
and the CLI prints it beside every score — so the number quoted always comes
from the same partition as the score it qualifies. When a design diagnostic and
an evaluation disagree, the evaluator's count is operative. Run design
diagnostics per era, not per file, before recommending a design.

## 2026-08-20 — Reported held-out row counts as if they were sample sizes

**What happened.** Four rounds of fixed-cell Italy results (next-actions items
6, 8, 93, 96, 102) were reported as scores over 1,064 or 1,368 held-out rows.
The cell-stratified control shows the held-out partition contains **seven**
independent label transitions at the current design, and **zero** in `era_0` —
no cell's held-out label varies there at all. Every one of those scores was
computed against single-digit information. They were treated as weak evidence
when they were no evidence.

**Root cause.** Anchors are 30 minutes apart and the target horizon is 7 days,
so consecutive target windows overlap by `99.7%` and the rows within a cell are
near-copies. Nothing in the pipeline reported the quantity that actually varies,
so the row count was the only number available and it stood in for the sample
size by default. The repeated `0.5` results were read as "the features carry no
signal" when "the target carries almost no variation" fits equally well and was
never checked.

**Prevention.** `./scripts/diagnose-spatial-target-design.sh` counts label
transitions for a candidate design before any model is fitted, and
`./scripts/evaluate-italy-spatial-cell-stratified.sh` reports how many strata
could be scored at all. Report the held-out transition count next to every
fixed-cell score; a score without it is not reportable. More generally: when
target windows overlap, state the effective sample size, not the row count.

## 2026-08-17 — New systemd unit omitted the Numba cache guard its sibling carried

**What happened.** `elfquake-space-weather.service` failed on its first timer
run with `RuntimeError: cannot cache function '_measure_local_damage': no
locator available`. Importing `elfquake.cli` pulls in `elfquake.sim`, whose
`@njit(cache=True)` kernels try to write a cache directory that
`ProtectSystem=strict` makes read-only. The existing
`elfquake-prospective.service` and `elfquake.service` both set
`Environment=ELFQUAKE_NUMBA_CACHE=0` for exactly this; the new unit did not.

**Root cause.** The unit was written from the shape of its sibling rather than
from its contents, and the omission was invisible to every local check — the
script runs fine outside the sandbox, so testing it in a normal shell proved
nothing about the sandboxed path.

**Prevention.** When adding a systemd unit that runs the CLI, diff it against an
existing working unit and carry over every `Environment=` line unless there is a
reason not to. A unit that runs under `ProtectSystem=strict` has to be exercised
under those settings, not just from a shell — `systemd-run` with the same
directives, or `systemctl start` plus `journalctl`, before considering it done.

## 2026-08-17 — Kyoto Dst normalizer split a fixed-width table on whitespace

**What happened.** `normalize_kyoto_dst_text` split each line on whitespace and
took fields 3 onward as the 24 hourly values. Kyoto's monthly pages are
fixed-width: a missing hour is the sentinel `9999`, and consecutive missing
hours run together with no separating space, so a line ending
`-10 -19999999999999 9999...` yields three whitespace tokens where the table has
twelve values. Any month with a data gap — which is every current month, since
the tail of the month is not yet observed — would have produced hours shifted
against their values, with no error raised.

**Root cause.** The function was written against an assumed plain-text WDC
export and never run against a real capture. It had test cover, but the test
constructed its own whitespace-delimited input, so the test and the code shared
the same wrong assumption about the format.

**Prevention.** Parse by column position and derive the month from the page
itself. Test fixtures for a fixed-width format must reproduce the real
layout including its sentinel runs, not a convenient whitespace variant. More
generally: a normalizer is not validated by a test whose fixture the same author
invented — check it against a stored raw capture before marking the source
usable, per the source-validation rule in AGENTS.md.

## 2026-08-17 — Test fixture built with the wrong column offset made a correct parser look broken

**What happened.** After writing the fixed-width Dst parser and verifying it
against a real capture, the unit test failed with 13 rows instead of 26. The
parser was right; the synthetic fixture used a 4-character day prefix where the
real format uses 3, shifting every value field by one.

**Root cause.** The fixture was written from a visual reading of the sample
output rather than from the offsets the parser was verified against.

**Prevention.** When a test for freshly verified code fails, check the fixture
against the real artifact before touching the code.

## 2026-08-17 — Two long pipeline steps in one foreground command hit the tool timeout mid-loop

**What happened.** A `for` loop rebuilding the `all_italy` and `central_italy`
prospective tables was run in the foreground. It timed out after two minutes
with the first scope written and the second not, leaving the derived layer in a
half-updated state.

**Root cause.** No runtime estimate before running a loop over commands that
each scan 700+ capture metadata files.

**Prevention.** Run multi-step data rebuilds in the background, or raise the
timeout explicitly. Prefer one command per invocation when each is slow, so a
timeout cannot leave a partial rebuild.

## 2026-08-22 — The transformer trained on dead channels for 13 days because sequences and fixture are separate artifacts

**What happened.** Item 100 aligned 19 real astronomy channels into
`common_transformer_fixture.csv` on 2026-08-17, and item 99's audit was
treated as answered. But the transformer does not read the fixture; it reads
the sequence tensors materialized from it, and nothing rebuilt those. Until
2026-08-22 the astronomy sequence still carried `astro_capture_count` and
`astro_noaa_solar_cycle_f107_value` — the two channels item 99 had condemned
as a collector-activity indicator and a monthly constant.

**Root cause.** A two-hop derivation where only the first hop was on anyone's
refresh list. `materialize-common-transformer-sequences.sh` is a separate
manual step with no dependency check, so "the fixture has the channels" was
read as "the model gets the channels".

**Prevention.** `run-cross-region-generative-smoke.sh` now checks both hops
with `require_fresh_inputs` (see `docs/input-freshness.md`), and the
reproduction steps in `docs/astronomy-alignment.md` name the materialize step
explicitly. More generally: when checking whether a fix reached a model, look
at the artifact the model opens, not the artifact the fix wrote.

## 2026-08-22 — A frozen literal in an argument list is staleness no timestamp can show

**What happened.** Both weekly forecast scripts defaulted
`AS_OF_UTC="${AS_OF_UTC:-2026-07-08T00:00:00Z}"`. Every run since then
forecast the same July week regardless of how far the event catalog had
advanced, on inputs that were themselves current.

**Root cause.** A date pinned during development to make a run reproducible,
left as the default. The item-89 audit looked for stale *files*; this was a
stale *question* asked of fresh files, and no modification-time check would
ever have found it.

**Prevention.** Both scripts now derive the default from
`catalog_coverage_end`, so the as-of date follows the catalog and an explicit
`AS_OF_UTC` is needed to reproduce an old run. When pinning a date, magnitude,
or window during development, make the pinned value the override and the
derived value the default, not the other way round.
