# Analysis Report

Date: 2026-07-11

## Progress So Far

The ideal outcome for ELFQuake would be to identify strong, consequential earthquakes in Italy before they happen, giving a useful estimate of when and where they will occur and how large they may be, while keeping false alarms low enough for the result to be actionable. That is the bottom-line standard against which the project should be judged.

Progress so far is mainly in building the machinery needed to test that goal. ELFQuake now has a working live-data pipeline for Italy, regularly collecting INGV earthquake events, Cumiana VLF spectrogram images, and available astronomy and space-weather measurements. It retains the original source files, aligns observations by time, trains baseline and Transformer models, tests the contribution of each modality, and can emit a demonstration seven-day event list and map. Historical INGV data provides seismic context, while the avalanche simulation supplies controlled synthetic seismic-like and piezo/VLF-like signals for model development. Self-supervised learning also lets the Transformer learn from live VLF observations before enough real earthquake labels have accumulated.

Compared with the ideal of predicting strong events, the present system is still at an early feasibility stage. It has not predicted a strong earthquake on held-out real data, and no multimodal model has yet beaten seismic-only and historical-rate baselines over a substantial real period. The overlapping live VLF and seismic record is short, the prospective target tables still lack usable class balance, and strong earthquakes are rare enough that a credible test will require much longer coverage. Current weekly forecasts demonstrate the intended software and output format only; they are not actionable predictions. The hypothesis remains open and testable, but useful strong-event prediction is not currently demonstrated or operationally viable.

The prospective-label pipeline now only marks a target mature when the fetched INGV catalog covers its entire target horizon. Current image-aligned tables contain 278 rows each: 149 mature labels and 129 future pending rows. The all-Italy mature rows are all positive and the central-Italy rows all negative because the closely spaced captures share the same few seven-day horizons, not because labels were reversed. This remains unsuitable for supervised real training. On the synthetic side, localized target refilling now uses persistent loading sites. Its potential-like piezo sensor shows a short lead only in an oracle diagnostic that selects the sensor closest to the future avalanche; the causal spatial average does not reproduce the effect across nine fresh episodes. It is therefore not used as a predictive model feature.

The first reduced-source (`SOURCE_COUNT=64`) screening run briefly met a causal potential-lead rule across three episodes, but this did not survive a nine-episode confirmation across 40 events. It is not a precursor feature. Its initial empty chronological test tail was traced to a fixed 84-hour modeling window after a 50-hour simulation; episode batches now derive the end timestamp from the number of simulated steps. With duration-aligned windows, the nine-episode reduced-source profile (`SOURCE_COUNT=64`, refill `470`, removal interval `20`, `q=0.998/window=120`) has 387 labeled rows, a `0.470284` positive rate, temporal drift `0.181728`, and a weak simple temporal score of `0.518750` balanced accuracy. It is a valid synthetic target baseline, not evidence of a piezo precursor.

Pre-relaxation spatial contact, coherence, and stress-weighted criticality were then added as observable grid diagnostics. They also failed a nine-episode causal confirmation across 39 events. The conclusion is now stronger: within the present instantaneous-relaxation sandpile, available state readouts are associated with the avalanche at the same step but do not demonstrate a stable earlier analogue of a piezo precursor.

An opt-in delayed-failure extension then changed the simulation dynamics rather than merely its sensors. Near-critical cells accumulate bounded damage before relaxation, local damage lowers the failure threshold, and toppling resets damage. On nine damage-enabled episodes, pre-relaxation `damage_total` passes the causal 5--15 step lead rule: AUC `0.652315`, standardized change difference `0.484616`, and 6 of 9 episode effects positive. The associated target table has 387 labeled rows (`197` positive, `190` negative) and temporal drift `0.245581`, just inside the current gate. The simple temporal event-list head is still weak at `0.415810` balanced accuracy, so this validates the synthetic precursor mechanism, not useful forecasting or a model improvement.

## Scope

This report summarizes the current statistical comparison between real Italy-scoped seismic/VLF data and signals derived from the avalanche simulation, plus the current model-interface and PyTorch smoke results.

The comparison is diagnostic only. It does not demonstrate earthquake prediction capability.

## Inputs

Real seismic:

* `data/derived/ingv/events_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 176 normalized all-Italy INGV event rows
* `data/derived/ingv/events_central_italy_2026-06-01_2026-07-08.combined.normalized.csv`
* 22 normalized central-Italy INGV event rows
* `data/derived/ingv/events_italy_all_available.combined.normalized.csv`
* 4836 normalized all-Italy rows, from `2024-01-01T21:38:30.320000Z` to `2026-07-07T08:51:55.110000Z`
* `data/derived/ingv/events_central_italy_all_available.combined.normalized.csv`
* 594 normalized central-Italy rows, from `2024-01-03T07:43:40.720000Z` to `2026-07-07T08:51:55.110000Z`

Real VLF:

* 247 Cumiana `last_E_VLF` spectrogram image-feature rows
* `data/derived/multimodal/cumiana_last_E_VLF.image_features.csv`
* image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`

Synthetic:

* current model training rows: `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.csv`
* 501 combined relabeled hourly synthetic rows from seeds `40`, `41`, and `42`
* source simulation outputs include `*.piezo.csv`, `*.avalanche_signal.csv`, and `*.avalanche_events.csv`

Real prospective model rows:

* `data/derived/models/all_italy.real_vlf_aligned_windows.csv`
* `data/derived/models/central_italy.real_vlf_aligned_windows.csv`
* each table has 278 rows, 149 labeled rows, and 129 future pending rows
* all-Italy labels are currently 149 positive / 0 negative
* central-Italy labels are currently 0 positive / 149 negative
* both are `insufficient_class_variation`

## Method

Seismic event lists were converted into hourly binned energy traces using magnitude-derived energy proxies.

VLF spectrograms were reduced to column-intensity traces after cropping the plotted signal region.

Synthetic piezo and direct avalanche signals were reduced to per-step summed sensor traces.

For each trace, the report computes time-domain shape metrics and frequency-domain PSD metrics. Pairwise `normalized_distance` uses dimensionless shape and PSD-ratio metrics; amplitude, sample count, and duration are reported as deltas but are not included in that distance.

In model reports, `ablation` means removing one feature group or modality, such as VLF or astronomy, and comparing performance with and without it.

Output files:

* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.central_italy_signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.central_italy_signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.all_italy_signal_shape_series.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.all_italy_signal_shape_pairs.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_vlf_comparison.csv`
* `data/derived/sim/mountain_256x256_seed42_10000.piezo_sensor_scan.csv`
* `data/derived/models/model_family_comparison.json`
* `data/derived/models/sequence_full_regime/sequence_full_model_run_summary.json`
* `data/derived/models/sequence_full_regime/post_burn_in_temporal_split_diagnostics.json`
* `data/derived/models/sequence_full_balanced/sequence_full_balanced_model_run_summary.json`
* `data/derived/models/sequence_full_balanced/regime_balanced_split.json`
* `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`
* `data/derived/models/real_synthetic_compact_comparison.json`
* `data/derived/models/deep_patch_transformer/deep_patch_transformer_synthetic.json`
* `data/derived/models/deep_patch_transformer/all_italy.real_finetune.json`
* `data/derived/models/deep_patch_transformer/central_italy.real_finetune.json`
* `data/derived/models/self_supervised/real_vlf_image_autoencoder.json`
* `data/derived/models/self_supervised/real_vlf_image_embeddings.csv`
* `data/derived/models/self_supervised/real_vlf_vs_synthetic_piezo_embedding_domain.json`
* `data/derived/models/self_supervised/real_vlf_vs_synthetic_piezo_embeddings.csv`
* `data/derived/models/trial_forecast/mag_gt2_weekly_trial_forecast.json`
* `data/derived/models/trial_forecast/mag_gt2_weekly_trial_events.csv`
* `data/derived/models/learned_forecast/mag_gt2_weekly_learned_forecast.json`
* `data/derived/models/learned_forecast/mag_gt2_weekly_learned_events.csv`
* `data/derived/models/forecast_comparison/trial_vs_learned_weekly_forecast.json`
* `data/derived/models/forecast_comparison/trial_vs_learned_weekly_forecast.csv`
* `data/derived/models/synthetic_event_list_patch_transformer/h6_patch_transformer.json`
* `data/derived/models/synthetic_event_list_patch_transformer_sweep/summary.json`

## Current Transformer Tuning

The current supervised Transformer work is still synthetic-only because real VLF-aligned labels remain one-class. A new target adapter maps the richer synthetic event-list labels into the standard Transformer input contract, preserving an 80/20 time-ordered split within each synthetic episode. On the warmed nine-episode h6 event-list target table, a short four-configuration CPU sweep found the best row at lookback `12`, patch `3`, dropout `0.1`: the piezo/VLF-only ablation reached calibrated balanced accuracy `0.608629`, while the full direct-avalanche plus piezo/VLF plus summary ablation reached `0.463892`. This suggests the VLF-like precursor channel is the most useful synthetic signal in this setup, but the result is not yet stable enough for forecast promotion.
* `data/derived/models/missing_modality/missing_modality_seed42_summary.json`
* `data/derived/models/sequence_modality_diagnostic.json`
* `data/derived/models/all_italy.ingv_backfill_seismic_windows.temporal_holdout.json`
* `data/derived/models/central_italy.ingv_backfill_seismic_windows.temporal_holdout.json`

## Current Status

Real-data status:

* INGV prospective refresh now rebuilds stable current catalogs and labels only horizons fully covered by the fetched catalog; historical backfill currently runs through `2026-07-07T00:00:00Z`.
* Cumiana VLF image capture and image-feature extraction are working.
* Real VLF-aligned model tables are scaffolded for all-Italy and central Italy.
* Real PyTorch training should not start yet, because each real table has only one target class. The real deep patch Transformer wrapper records this blocker instead of training.
* Current real VLF-aligned label counts are 149 positive / 0 negative for all-Italy and 0 positive / 149 negative for central Italy; each scope has 129 future pending rows.
* Self-supervised real VLF pretraining is available and is now the default model-development path until supervised labels have both classes.
* Historical seismic-only backfill for `2024-01-01` to `2026-07-07` produced 130 weekly training windows per scope.
* Backfilled all-Italy seismic windows are ready but heavily positive-skewed: train 95 positive / 9 negative, test 25 positive / 1 negative.
* Backfilled central-Italy seismic windows are more balanced: train 13 positive / 91 negative, test 6 positive / 20 negative.
* Current seismic-only temporal smoke scores are weak: all-Italy calibrated balanced accuracy `0.740000` on a one-negative test set; central Italy calibrated balanced accuracy `0.441667`.

Synthetic-model status:

* The current relabeled synthetic model table has 501 rows, with 67 positives and 434 negatives.
* CPU PyTorch tabular and GRU sequence models are implemented and compared.
* Current corrected-label temporal smoke scores remain weak: tabular PyTorch calibrated balanced accuracy `0.507576`; sequence GRU calibrated balanced accuracy `0.500000`.
* Current corrected-label seed holdouts are stronger but synthetic-only: tabular PyTorch ranges from `0.732602` to `0.784169`, while sequence GRU ranges from `0.720690` to `0.821105`.
* Older sequence sweeps, matched comparisons, repeated training-seed runs, and tiny patch Transformer checks were produced before the target relabeling and should be rerun before model selection.
* Corrected-label temporal split diagnostics still show a label/regime shift: `gt0` train positive rate `0.080000`, test positive rate `0.346535`; largest drift features are direct-avalanche active-topple and summary-topple aggregates.
* A post-burn-in regime-balanced explicit split has matched train/test class rates and gives `sequence_full` calibrated balanced accuracy `0.650000`; use this as an engineering diagnostic, not as forecasting evidence.
* The selected deeper patch Transformer pretrain now writes `deep_patch_transformer_synthetic.pt`; its latest synthetic calibrated scores are `0.737879` for piezo/VLF-only and `0.583333` for full sequence.
* The first self-supervised real VLF autoencoder smoke used 247 Cumiana VLF rows and 224 windows. Test masked reconstruction MSE was `0.835488`, better than the zero baseline `1.074356`.
* A shared patch-Transformer self-supervision evaluation now compares five initialization regimes over seeds `7`, `42`, and `99`. Synthetic pretraining raises mean held-out full-model balanced accuracy from random initialization `0.451754` to `0.488678`; balanced joint real/synthetic pretraining reaches `0.487352`, and real-only pretraining reaches `0.470726`. All remain below the `0.60` synthetic gate.
* Synthetic masked reconstruction beats zero and last-patch baselines for all pretrained seeds. Real VLF reconstruction beats zero but not last-patch persistence, reflecting the small 186-window real training set and strong short-term persistence.
* Sequential synthetic-then-real pretraining has the strongest mean frozen probe (`0.519176`) but weak fine-tuning (`0.452264`), so the next transfer check should use joint rehearsal or freeze the shared encoder during the real-only stage.
* Positive/negative recall remains seed-sensitive; most runs fail to keep both above `0.40`. The initialization gains are therefore useful engineering evidence but not a promotion result.
* Transformer components now use stable name-derived initialization. Adding an unused modality adapter cannot alter shared parameters or advance the caller's PyTorch random stream, making architecture comparisons genuinely matched.
* In the controlled transfer rerun, random-init piezo/VLF-only is strongest at mean balanced accuracy `0.619033` over three seeds, with a `0.577723` to `0.663709` range and both recalls above `0.40` in every seed. Synthetic pretraining falls to `0.534272`; current self-supervision does not improve this synthetic downstream target.
* Controlled late gated fusion also does not improve the model. Random anchored full and direct-only variants average `0.609241` and `0.606385`, below the `0.619033` anchor. Disabling direct avalanche raises the direct-only mean to `0.611281`, so that branch has not demonstrated incremental value.
* A stricter leave-one-episode-out run holds out each of nine simulation episodes across three model seeds. Mean balanced accuracy drops to `0.578712`, ranges from `0.275641` to `0.758730`, and only 14 of 27 folds keep both recalls above `0.40`. This fails the synthetic stability gate and shows that the promising within-episode score does not yet generalize reliably across simulation episodes.
* Averaging the three predeclared model seeds raises unseen-episode mean balanced accuracy to `0.632634`, but only 6 of 9 episodes retain both recall floors. Training-only threshold calibration outperforms a fixed `0.5` threshold (`0.601361`) without fixing the three unstable episodes, so the remaining failure is mainly episode generalization rather than threshold choice.
* Episode diagnostics show that piezo feature effects change sign across trajectories. Shorter h3 labels (`0.580991` ensemble mean), longer 60-minute h6 context (`0.512103`), and simple spatial sensor aggregates (`0.544096`) all regress. The h6/12-minute mean-only representation remains best, but zero of four variants passes the combined mean and recall-stability gate.
* A second label-free real VLF anomaly layer now scores descriptor reconstruction and embedding novelty by window. The current 7-day smoke forecast artifact covers `2026-07-06T06:50:50Z` to `2026-07-13T06:50:50Z`, with demo probability `0.952514` and demo predicted event `1`; this is not trained on earthquake labels and is not a validated forecast.
* The tuned shape-profile synthetic-to-real embedding-domain diagnostic encoded 59,931 synthetic piezo/VLF windows through a descriptor autoencoder trained on real VLF windows. Synthetic centroid distance was `1.291640` and synthetic-to-real nearest mean distance was `1.846295`.
* The same diagnostic is still only a baseline, but it is stronger than the previous full-descriptor version: held-out real masked reconstruction MSE is `0.895188` versus a zero baseline of `0.960585`; synthetic masked reconstruction is `4.642818` versus a zero baseline of `4.721987`.
* A real-like synthetic inlier subset now marks the closest 25% of synthetic descriptor windows. It keeps 14,983 synthetic windows and reduces synthetic-to-real nearest mean distance to `1.162097`, with scale mean absolute delta `0.057490`.
* A synthetic-inlier transfer diagnostic now trains the masked descriptor autoencoder only on those 14,983 synthetic windows and evaluates on held-out real VLF descriptors. Held-out real masked reconstruction MSE is `0.688280` versus a zero baseline of `0.759011`, but the transfer embedding centroid distance remains high at `4.281796`.
* A mixed-domain alignment diagnostic now trains on real VLF plus 14,983 locally selected synthetic piezo/VLF windows with a CORAL embedding-alignment penalty. Held-out real masked reconstruction improves to `0.294475` versus a zero baseline of `0.588513`, and held-out embedding centroid distance improves to `1.033580`.
* Mixed-domain controls remain important: centroid-inlier selection scored `1.011474`, random synthetic selection `1.142438`, and capped full-synthetic selection `1.617345` on held-out centroid distance. This means alignment training is useful, but the local inlier criterion is not yet clearly superior to centroid selection.
* A first end-to-end trial weekly event-list forecast now combines current INGV history, VLF context, astronomy captures, and synthetic avalanche event artifacts. The `2026-07-08T00:00:00Z` run for the following week produced 25 capped `>M2` coordinate rows, with an uncapped expected count proxy of `33.411669`; this is a contract smoke test and not a validated prediction.
* A first synthetic-trained learned weekly event-list forecast now preserves the same CSV contract. It trained a small logistic scorer on 501 aligned synthetic rows with 67 positives, produced a latest synthetic score of `0.899322`, and emitted 25 capped coordinate rows with an uncapped expected count proxy of `32.709463`.
* The learned scorer is only a scaffold: its held-out synthetic balanced accuracy is `0.507576`, with 35/101 positives in the test split and very poor negative recall (`0.015152`).
* A forecast comparison report now makes success criteria explicit. Stage 1 event-contract output passes; Stage 2 synthetic utility fails because balanced accuracy is below `0.60` and negative recall is below `0.40`.
* The learned forecast currently changes count/probability metadata but not event placement; candidate-to-baseline nearest distance is `0.0 km`, so a model-informed spatial adapter remains a necessary next step.
* Synthetic event-list target generation now derives forecast-shaped targets directly from avalanche event CSVs: future event count, occurrence, max/mean magnitude, centroid, first-event time, and time-to-first-event. The current h6 target table has 483 labeled rows with 213 positives and a `0.440994` positive rate, making it the healthiest current horizon.
* A dependency-light synthetic event-list model now trains separate heads for occurrence, count, max magnitude, and centroid. On the h6 temporal split it still fails occurrence utility (`0.500000` balanced accuracy, `0.000000` negative recall), but on the h6 balanced engineering split it reaches `0.887566` balanced accuracy, count MAE `0.506783`, max-magnitude MAE `0.009317`, and centroid median error `145.585806 km`.
* Current event-list model artifacts are `data/derived/models/synthetic_event_list_model/h6_event_list_model.json`, `h6_event_list_predictions.csv`, `h6_balanced_event_list_model.json`, and `h6_balanced_event_list_predictions.csv`. Treat the balanced result as evidence that the synthetic target shape is learnable, not as forecasting validation.
* Drift-aware synthetic event-list validation now writes `data/derived/models/synthetic_event_list_drift/h6_drift.json` and `h6_drift_buckets.csv`. The temporal split confirms the failure mode: train positive rate `0.318653`, test positive rate `0.927835`, positive-rate delta `0.609182`, warning `large_positive_rate_shift`.
* The largest feature shifts in the failing temporal split are direct-avalanche active-topple counts, summary topple counts, summary max height, and piezo near-critical cell counts. This points to a simulation lifecycle/regime issue rather than a purely model-capacity issue.
* Episode annotations now write `mountain_256x256_seeds40-42_20000.synthetic_event_list_targets_h6.episodes.csv` with 21 episode blocks. Episode-balanced validation remains learnable with balanced accuracy `0.878079`, positive recall `0.928571`, negative recall `0.827586`, count MAE `0.497667`, and centroid median error `162.751648 km`.
* The first stationarity episode batch, `mountain_256x256_seeds4000-4201_5000`, generated six 5000-step episodes. It reduced overall h6 positive rate to `0.188312`, but made chronological drift worse: train positive rate `0.056911`, test positive rate `0.709677`, positive-rate delta `0.652766`. The run still accumulated mass strongly, ending near mean height `111`, so the loading/removal balance remained wrong.
* `sim.sh` now exposes `DEPOSITION_PROBABILITY` and `WARMUP_STEPS`. `run-synthetic-episode-batch.sh` now defaults to a warmed aggressive stationarity profile: target fill limit `WIDTH * HEIGHT / 128`, bottom-layer removal every `25` steps, deposition probability `0.45`, and `3000` unrecorded warm-up steps.
* The aggressive stationarity probe, `mountain_256x256_seeds4000-4200_3000`, generated three 3000-step episodes. It kept final mean height near `3.2` and fixed the h6 drift warning: overall positive rate `0.318182`, train positive rate `0.304762`, test positive rate `0.370370`, positive-rate delta `0.065608`, warning `ok`.
* The aggressive probe is too small for model conclusions. Its temporal occurrence balanced accuracy is `0.505882`; episode-balanced split reaches `0.628571`, but count and centroid heads remain weak. Use it as evidence that the simulation drift can be controlled, then scale the profile to more episodes and tune event extraction density.
* Structured initial fill is now implemented via `INITIAL_FILL_MODE=structured`, `INITIAL_FILL_MEAN_HEIGHT`, `INITIAL_FILL_VARIATION`, and `INITIAL_FILL_SMOOTH_PASSES`. The first structured-fill probe, `mountain_256x256_seeds6000-6200_3000`, started near the loaded regime and ended near mean height `3.2`, but did not improve drift versus the no-prefill aggressive probe: overall positive rate `0.310606`, train positive rate `0.247619`, test positive rate `0.555556`, positive-rate delta `0.307937`, warning `large_positive_rate_shift`.
* Unrecorded warm-up is more promising than initial fill. The warm-up probe, `mountain_256x256_seeds7000-7200_3000`, used `1000` unrecorded steps before each recorded episode. It produced overall positive rate `0.356061`, train positive rate `0.352381`, test positive rate `0.370370`, positive-rate delta `0.017989`, warning `ok`. Treat this as the current best small stationarity profile, but still too small for model-selection claims.
* Scaling the `1000`-step warm-up profile to nine episodes, `mountain_256x256_seeds8000-8202_3000`, did not preserve the small-probe drift result. It produced 396 labeled rows, overall positive rate `0.277778`, train positive rate `0.218354`, test positive rate `0.512500`, positive-rate delta `0.294146`, warning `large_positive_rate_shift`. Mean height stayed bounded, so the remaining failure is early-vs-late target drift rather than overflow.
* A `3000`-step warm-up probe, `mountain_256x256_seeds9000-9200_3000`, started closer to the late-run regime and restored h6 drift control: 132 labeled rows, overall positive rate `0.257576`, train positive rate `0.247619`, test positive rate `0.296296`, positive-rate delta `0.048677`, warning `ok`. The temporal event-list smoke model reached balanced accuracy `0.674342`, positive recall `0.875000`, negative recall `0.473684`, and count MAE `0.330187`; still treat this as a small engineering check.
* Scaling the `3000`-step warm-up profile to nine episodes, `mountain_256x256_seeds10000-10202_3000`, preserved acceptable h6 drift: 396 labeled rows, overall positive rate `0.325758`, train positive rate `0.287975`, test positive rate `0.475000`, positive-rate delta `0.187025`, warning `ok`. The temporal event-list model remained weak (`0.468045` balanced accuracy), but balanced and episode-balanced checks reached `0.609579` and `0.590008`, respectively.
* Direct avalanche extraction density was tested on the same nine warmed episodes without rerunning simulation. The dense `_q099_w60_m10` profile produced 10 events per episode and stayed drift-ok, but saturated h6 targets: positive rate `0.828283`, temporal balanced accuracy `0.530333`, and poor negative recall. The midpoint `_q0995_w120_m5` profile produced 5 events per episode and a healthier positive rate `0.512626`, but model checks were weaker than the sparse default: temporal `0.462475`, balanced `0.493902`, episode-balanced `0.554605`.
* Interpretation: keep sparse event extraction as the default for now. It is not dense enough for final synthetic training, but the denser profiles did not improve model learnability; the next improvement should come from better event-list heads, loss weighting, or richer target construction rather than simply lowering the extraction threshold.
* The dependency-light event-list occurrence head now supports deterministic feature-bag ensembles. `train-synthetic-event-list-model.sh` defaults to 8 ensemble members with 50% feature bags. On the scaled sparse nine-episode table this improved the temporal split from `0.468045` to `0.498120`, the balanced split from `0.609579` to `0.616473`, and the episode-balanced split from `0.590008` to `0.607551`. The gain is modest but consistent; temporal utility remains below the `0.60` synthetic gate.
* Synthetic event-list targets are now richer: they include event-rate, log magnitude energy, early/middle/final horizon event counts, peak timing, event duration, and spatial spread in addition to occurrence, count, magnitude, centroid, and first-event timing. Matching regression heads now write `predicted_*` fields and event-shape metrics into the model reports.
* On the scaled sparse nine-episode balanced split, the richer shape heads give event-rate MAE `0.068936`, early/middle/final count MAE `0.180167`/`0.246829`/`0.194170`, spatial-spread MAE `20.317859 km`, and time-to-peak MAE `6769.366174 s`. These are useful shape checks, but they do not by themselves solve chronological generalization.
* An optional boosted-stump occurrence head was added for nonlinear dependency-light checks. It improved the balanced engineering split to `0.629173` balanced accuracy but failed the chronological split at `0.347744`, so the feature-bag logistic ensemble remains the default.
* A synthetic event-list probe harness now compares horizon, burn-in, feature-cap, ensemble-size, boosted-stump, and balanced-control variants, then writes compact JSON/CSV summaries. A constrained h3 smoke run passed end to end; at 20 epochs, the 16-member feature-bag temporal variant reached `0.557692` balanced accuracy, while boosted stumps remained weaker at `0.371154`. Treat this as a harness check, not a model-selection result.
* The full event-list probe wrote `data/derived/models/synthetic_event_list_probes/summary.csv` with 54 model/drift reports. Among drift-ok chronological runs, the best result was h6 with a 32-feature cap: balanced accuracy `0.511278`, positive recall `0.736842`, negative recall `0.285714`, count MAE `0.520321`, and centroid median error `640.573894 km`. The nominal best chronological score was h12 with 20% burn-in and 64 features at `0.579268`, but that target had `large_positive_rate_shift` (`0.296459`) and should not be promoted.
* Balanced controls remain much stronger than chronological checks. The best drift-ok balanced control was h12 boosted stumps with balanced accuracy `0.651543`; h12 episode-balanced reached `0.626879`, h6 boosted balanced reached `0.629173`, and h6 episode-balanced reached `0.607551`. This confirms target learnability under controlled splits but not chronological utility.
* Interpretation: simple horizon changes, burn-in trimming, tabular feature caps, larger feature-bag ensembles, and boosted stumps did not solve temporal generalization. The next modeling improvement should add explicit temporal context or sequence/event-process heads rather than more one-row tabular occurrence heads.
* Lagged-context h6 probes add previous-row synthetic feature history while excluding target and diagnostic fields. This is the first explicit temporal-context check. The 128-feature cap improved drift-ok chronological balanced accuracy to `0.545739`; the 256-feature cap improved it further to `0.587093`, with positive recall `0.578947` and negative recall `0.595238`. The 64-feature cap regressed to `0.439223`, adding lag 12 scored `0.535088`, and all features regressed to `0.530702`.
* Interpretation: lagged context is the first change that materially narrows the temporal gap, but it is still just below the `0.60` gate. The next implementation should use an explicit sequence/event-process head with regularization rather than simply widening tabular lag features.
* A regularized CPU PyTorch event-list sequence head now reads synthetic event-list target tables directly, builds grouped lookback sequences, excludes target/diagnostic fields, and uses class weighting, dropout, AdamW weight decay, and gradient clipping. On h6 drift-ok chronological validation, the default lookback-12 seed-42 run reaches calibrated balanced accuracy `0.609649`, positive recall `0.552632`, and negative recall `0.666667`, passing the `0.60` synthetic gate for the first time on this target family.
* Sequence-head controls show the result is promising but not yet stable. Lookback 8 seed-42 scored `0.603383`; lookback 4 scored `0.467419`; hidden size 32 scored `0.578321`; lookback 12 with seed 7 scored `0.528822`; lookback 12 with seed 99 scored `0.584586`; lookback 12 with dropout `0.30` scored `0.569549`. Treat the gate pass as a directionally useful engineering result, not a promoted forecast adapter.
* A first sequence-head stability sweep wrote `data/derived/models/synthetic_event_list_sequence_sweep/summary.csv` across three seeds, lookbacks `8` and `12`, and dropout `0.1`/`0.2`/`0.3`. The best mean configuration is now lookback `12`, dropout `0.1`: mean balanced accuracy `0.600459`, min `0.576441`, max `0.645363`, stddev `0.031777`, with one of three seeds passing the `0.60` gate. The default sequence-head script now uses this setting.
* Interpretation: the sequence head is the first event-list model family with mean performance at the synthetic gate, but it is still seed-sensitive. The next improvement should reduce variance, for example by model ensembling, longer training controls, or stronger regularization/early stopping.
* Sequence-head probability ensembling is now implemented. The three-seed lookback-12/dropout-0.1 ensemble scored `0.591479`, below the gate. Pairwise ensembles show selection sensitivity: seed `42+99` scored `0.644110`, seed `7+42` scored `0.604637`, and seed `7+99` scored `0.555764`. This means ensembling can improve a chosen pair, but naive all-seed averaging does not yet solve stability.
* Validation-selected sequence heads were tested with a 20% chronological validation slice inside the training period. This underperformed the train-calibrated baseline: lookback-12/dropout-0.1 mean balanced accuracy fell to `0.539265` with no seed passing the gate, and dropout `0.2` fell to `0.519632`.
* Early-stopped sequence heads were tested on the same validation slice with patience `10` and train-calibrated thresholds. This also underperformed and increased variance: dropout `0.1` mean balanced accuracy was `0.494570`, min `0.335213`, max `0.615288`; dropout `0.2` mean was `0.471178`. Current validation slices appear too small or distribution-shifted to guide early stopping safely.
* Interpretation: do not use validation-selected thresholding or early stopping as the next default. Keep the full-epoch lookback-12/dropout-0.1 sequence head as the current occurrence candidate, and pursue variance reduction through larger episode batches, more stable validation design, or representation sharing with count/location heads.
* Interpretation: initial fill can skip cold-start height growth, but with immediate bottom-layer removal and the current event-extraction settings it does not by itself produce a more stationary event process. Prefer warmed episodes over structured initial fill unless a later delayed-removal experiment beats the warm-up profile.
* A short piezo/VLF transform sweep added deterministic high-pass, burst, near-threshold, release-mix, and sensor-gain variants. The best transformed variant, `gain_burst`, improved short-run held-out embedding centroid distance to `1.757251` versus `1.841903` for the current signal, but worsened held-out masked reconstruction to `0.318687` versus `0.281600`.
* Refreshed missing-modality seed-42 checks give `0.632445` calibrated balanced accuracy for piezo/VLF-only and `0.722257` for direct-avalanche-only.
* Refreshed sequence modality diagnostics still rank direct-avalanche-only highest on grouped synthetic checks (`0.8359` calibrated balanced accuracy), so direct seismic-like and piezo/VLF-like channels should remain separate.
* A short diversity smoke run generated extra 128x128, 1000-step seeds `43` and `44` and refreshed aligned tensors with evaluations disabled; use larger runs before drawing model conclusions.
* Aligned synthetic targets now use true future look-ahead semantics: `target_horizon_rows=N` labels an input row from the sum of direct avalanche events in the next `N` complete rows, not from the current row or a single offset row.

## Shape Diagnostics

Real seismic vs synthetic seismic event traces:

* central-Italy seed `42` sparse-event distance: `1.4709`
* all-Italy seed `42` sparse-event distance: `1.3992`
* central-Italy real seismic is very sparse: nonzero ratio `0.0245`, PSD slope `-0.0380`
* all-Italy real seismic is less sparse: nonzero ratio `0.1756`, PSD slope `-0.0272`
* current synthetic seismic events remain too dense: nonzero ratio `0.4275`, PSD slope `0.0689`
* raw synthetic avalanche signal is effectively continuous: nonzero ratio `0.9991`

Real VLF image columns vs synthetic piezo/VLF signal:

* shape distance: `1.4620`
* real VLF PSD slope: `-0.3506`
* synthetic piezo PSD slope: `-0.2705`
* real VLF lag-1 autocorrelation: `0.5381`
* synthetic piezo lag-1 autocorrelation: `0.9658`

VLF image-level comparison:

* nearest Cumiana image: `data/raw/vlf/cumiana/captures/2026-06-30/last_E_VLF_2026-06-30T23-15-00Z.jpg`
* nearest normalized distance after render tuning: `13.6492`
* simulated intensity mean: `0.5987`
* real mean intensity: `0.4687`
* simulated high-intensity ratio: `0.1695`
* real high-intensity ratio: `0.1712`
* simulated vertical streak count: `115`
* real mean vertical streak count: `117.6957`

Piezo sensor scan:

* all 16 piezo sensors were compared individually against the current Cumiana VLF image-column trace
* default pre-tuning best sensor: `9`
* tuned best sensor: `5`
* default pre-tuning best sensor lag-1 autocorrelation: `0.8081`
* tuned best sensor lag-1 autocorrelation: `0.6002`
* current VLF reference lag-1 autocorrelation in the scan: about `0.5930`
* summed piezo lag-1 autocorrelation remains much smoother at about `0.9658`
* corrected default pre-tuning shape score: `0.9701`
* tuned threshold/locality shape score: `0.6217`
* tuned burst-run rate: `0.0299`; current VLF reference rate differs by about `0.0073`
* experimental threshold-40 release alone improved lag, but adding receiver locality improved the overall corrected score

Piezo multi-seed validation:

* output CSV: `data/derived/sim/piezo_seed_validation_summary.csv`
* seed `40`: best sensor `10`, shape score `0.6504`, lag-1 `0.5816`
* seed `41`: best sensor `0`, shape score `0.6206`, lag-1 `0.6267`
* seed `42`: best sensor `5`, shape score `0.6165`, lag-1 `0.6002`
* mean shape score over seeds `40`-`42`: `0.6292`
* mean lag-1 autocorrelation over seeds `40`-`42`: `0.6028`

Full-size multi-seed check:

* `compare-simulation-grid.sh` ran for full-size `256 x 256`, `10000` step runs on seeds `40`, `41`, and `42`
* metrics-only runs used `RUN_HEATMAPS=0`, `RUN_VIDEO=0`, and `RUN_AUDIO=0`
* summary CSV: `data/derived/sim/full_size_seed_comparison.csv`

| seed | events | seismic distance | event nonzero | event PSD slope | piezo/VLF distance | piezo lag-1 | nearest VLF image distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 189 | 1.5775 | 0.8030 | 0.2524 | 1.4041 | 0.9581 | 13.2128 |
| 41 | 191 | 1.6198 | 0.8106 | -0.4042 | 1.4663 | 0.9666 | 13.4896 |
| 42 | 172 | 1.5258 | 0.7652 | -0.1678 | 1.4620 | 0.9658 | 13.6492 |

Direct avalanche event extraction tuning:

* tuning grid: quantiles `0.90`, `0.95`, `0.975`, `0.99`; local-max windows `5`, `15`, `30`, `60`
* output CSVs: `data/derived/sim/mountain_256x256_seed*_10000.avalanche_event_tuning.csv`
* best quantile for seeds `40`, `41`, and `42`: `0.99`
* best local-max window is seed-dependent: seed `40` prefers `5`, seed `41` prefers `60`, seed `42` prefers `30`
* current default `0.95/15` is denser than the tuned `0.99` candidates
* longer `20000` step runs for seeds `40`, `41`, and `42` also favour quantile `0.99`
* in the longer runs, window `30` is best for seeds `40` and `41`, and second-best for seed `42`
* refined sparse tuning adds `max_events` and a stable `shape_score` independent of candidate-grid size
* refined 10000-step seed `42` central-Italy sparse profile: `q=0.99`, window `480`, max events `3`
* refined 20000-step seeds `40`-`42` consistently prefer: `q=0.999`, window `240`, max events `5`
* refined 20000-step sparse profile event shape scores: seed `40` `0.193665`, seed `41` `0.119737`, seed `42` `0.144586`

## Interpretation

Sparse local-peak extraction improved the direct synthetic seismic event trace. The extended 2024-2026 INGV comparison showed the previous synthetic seismic event list was still too dense, especially against central Italy. The refined sparse profile improves 10000-step seed `42` central-Italy distance from `1.4709` to `1.0218`, with nonzero ratio `0.0259` versus real `0.0245`. Keep it as a sparse seismic profile until downstream target balance is re-evaluated.

Across the full-size seed check, seed `42` is still the best current default on direct seismic event-shape distance and sparsity, but all tested seeds remain much denser than real seismic events. The next tuning target is therefore direct avalanche event extraction thresholds and burst spacing, not map projection.

The old extraction tuning pass supported quantile `0.99` and local-max window `30` as a usable model-training default. The refined sparse profile uses far fewer events (`q=0.999`, window `240`, max events `5` on 20000-step runs) and better matches central-Italy sparsity, but it may make synthetic model targets too sparse if promoted without redesigning the target windows.

The target-window redesign is now implemented. On the refined sparse seed `40`-`42`, `20000`-step profile, positive labels scale with the look-ahead horizon as expected: horizon `1` has `3/501` positives, horizon `3` has `9/495`, horizon `6` has `18/486`, horizon `12` has `36/468`, and horizon `24` has `72/432`. Temporal splits still have zero positive test rows because the sparse synthetic events occur too early in each run, so the sparse profile should not become the default until event timing is less clustered or substantially longer synthetic runs are available.

The piezo/VLF image rendering is now much closer to Cumiana image statistics for brightness, high-intensity coverage, and vertical streaks. The underlying piezo time series is still too smooth in time, with very high autocorrelation, so later tuning should focus on signal dynamics rather than adding display artifacts.

The per-sensor scan shows that summing all piezo sensors over-smooths the VLF-like channel. Single sensors are more plausible. Raw burst-run counts were misleading because the image-derived VLF trace has many more samples than the simulation trace, so comparisons now use burst-run rate.

The current tuned default uses thresholded accumulated-charge release plus a local receiver footprint. It improves lag-1 autocorrelation and the corrected overall shape score without artificial spikes. The seed `40`-`42` validation is consistent enough to keep these as current simulation defaults, but the best receiver is seed-dependent, so model data preparation should preserve sensor identity and support sensor selection or pooling.

Piezo receiver locality is now separated from the direct avalanche signal range. Tuning the VLF-like piezo channel should not silently change the seismic-like avalanche channel.

After that separation and the target relabeling fix, seed `40`-`42` simulation CSVs, aligned synthetic model rows, tensors, temporal diagnostics, tabular PyTorch reports, sequence GRU reports, and the tabular-vs-sequence comparison were refreshed. The chronological holdout remains weak, while leave-one-seed-out checks remain stronger and should be treated as synthetic transfer diagnostics only.

Longer synthetic run check:

* regenerated seed `40`, `41`, and `42` at `20000` steps under current defaults
* sparse event counts: seed `40` has `130`, seed `41` has `129`, seed `42` has `135`
* combined `gt0` aligned table after future look-ahead relabeling: `501` rows, `67` positives, `434` negatives
* combined `gt1` aligned table after future look-ahead relabeling: `501` rows, `2` positives, `499` negatives
* `gt0` chronological best calibrated balanced accuracy: `0.507576`
* `gt0` leave-one-seed-out best calibrated balanced accuracy range: `0.591536` to `0.826389`
* `gt1` remains mostly a sparsity check

Chronological split diagnostics:

* diagnostic outputs:
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows.temporal_diagnostics.csv`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.json`
  * `data/derived/models/mountain_256x256_seeds40-42_20000.aligned_hourly_synthetic_windows_gt1.temporal_diagnostics.csv`
* `gt0` train positive rate is `0.080000`; test positive rate is `0.346535`
* `gt1` train positive rate is `0.000000`; test positive rate is `0.019802`
* later test windows have much higher avalanche/topple intensity: `synthetic_direct_avalanche_active_topple_cell_count_*` and `synthetic_summary_topple_count_*` drift upward by more than one training standard deviation
* several direct avalanche and summary features still change target-correlation strength between train and test in the `gt0` split

The longer run increases target support but does not improve chronological generalization. It should be used to stress-test the model interface and synthetic-transfer workflow, not as evidence of predictive value.

The weak `20000` chronological holdout appears to be mostly a non-stationary split problem. The simulation trajectory changes regime over time: later windows are taller and more avalanche-active, and the target rate changes substantially. Time-split evaluation is still the right conservative check, but the current synthetic generator needs burn-in handling, regime-balanced splitting, or longer/more varied runs before model metrics are meaningful.

The direct avalanche signal should remain separate from the piezo/VLF channel. Cross-modality distances can be useful sanity checks, but tuning should compare real seismic primarily with direct avalanche outputs, and real VLF primarily with piezo-derived outputs.

## Model Scaffold Update

Current CPU PyTorch tabular and sequence reports can now be compared directly with `compare-model-runs.sh`. The sequence path also has a bounded sweep script, a missing-modality smoke script, and a real Cumiana VLF image sequence materializer so synthetic piezo/VLF and real VLF inputs keep the same model-facing shape.

Corrected-label smoke outputs:

* tabular-vs-sequence comparison: `data/derived/models/mountain_256x256_seeds40-42_20000.tabular_vs_sequence_model_comparison.json`
* best calibrated row in that comparison: `0.826389`, `seismic_vlf_unified`, held-out `seed41`
* tabular PyTorch temporal row: calibrated balanced accuracy `0.507576`
* tabular PyTorch seed holdouts: calibrated balanced accuracy `0.784169`, `0.762832`, and `0.732602`
* sequence GRU temporal row: calibrated balanced accuracy `0.500000`
* sequence GRU seed holdouts: calibrated balanced accuracy `0.720690`, `0.821105`, and `0.768339`
* real VLF image sequence manifest: `data/derived/models/cumiana_vlf_image_sequence/manifest.json`, with `247` time steps and `25` channels
* prospective VLF-aligned labels currently have `23` all-Italy positives and `0` negatives, while central Italy has `0` positives and `23` negatives, so real VLF-aligned training still has insufficient class variation

Pre-relabel sweep outputs that need rerunning before model selection:

* tiny sequence sweep smoke: `data/derived/models/sequence_sweep_smoke/sequence_sweep_comparison.json`; best calibrated row `0.709624`, `sequence_full`, held-out `seed41`
* missing-modality smoke: `data/derived/models/missing_modality/missing_modality_seed42_summary.json`; no-piezo direct avalanche scored higher than piezo-only in this short run
* full sequence sweep: `data/derived/models/sequence_sweep/sequence_sweep_comparison.json`, `24` reports; best calibrated row `0.766942`, `sequence_direct_avalanche_only`, `lookback=60`, `hidden=24`, held-out `seed42`
* combined family comparison: `data/derived/models/model_family_comparison.json`, `37` rows; pre-relabel best calibrated row `0.772558`, `sequence_piezo_vlf_only`, held-out `seed42`
* sequence modality diagnostic: `data/derived/models/sequence_modality_diagnostic.json`, `112` evaluation rows; best default sequence row uses `20` epochs and piezo/VLF-only, while best sweep row uses `10` epochs and direct avalanche-only
* matched 20-epoch sequence comparison: `data/derived/models/sequence_sweep_20epoch/default_vs_matched_sequence_diagnostic.json`, `64` evaluation rows; pre-relabel best row `sequence_piezo_vlf_only`, `lookback=60`, `hidden=24`, held-out `seed42`, calibrated balanced accuracy `0.772558`
* repeated training-seed comparison: `data/derived/models/sequence_training_seed_repeat/sequence_training_seed_selection.json`; pre-relabel best single row `sequence_piezo_vlf_only` at `0.772558`, with `sequence_full` best on mean group score (`0.741342`) and worst-held-out-seed score (`0.712754`)
* tiny patch Transformer scaffold: `data/derived/models/tiny_patch_transformer/tiny_patch_transformer_model_run_summary.json`; pre-relabel best calibrated row `0.637500`, `sequence_piezo_vlf_only`, explicit regime-balanced split
* real model-input scaffold: `data/derived/models/all_italy.real_vlf_aligned_windows.csv` and `data/derived/models/central_italy.real_vlf_aligned_windows.csv`; both have `247` rows and `23` labeled rows but still lack class variation

Sequence diagnostic interpretation: do not change the default GRU lookback from `60` on current evidence. The corrected-label temporal sequence row remains at balanced accuracy `0.5`; older sweep and missing-modality reports need rerunning before choosing between direct avalanche, piezo/VLF, and full sequence inputs.

Post-burn-in regime interpretation: `sequence_full` does not yet show robust performance once the first 20 percent of each synthetic seed is removed and holdouts are made by seed/regime block. This supports the earlier concern that the current synthetic generator and split design still contain regime effects that should be understood before larger model runs.

Compact model comparison:

* artifact: `data/derived/models/real_synthetic_compact_comparison.json`
* CSV view: `data/derived/models/real_synthetic_compact_comparison.csv`
* central-Italy historical seismic-only temporal baseline: calibrated balanced accuracy `0.441667`
* corrected-label synthetic temporal sequence run: calibrated balanced accuracy `0.500000`
* corrected-label best synthetic seed-holdout row: `seismic_vlf_unified`, held-out `seed41`, calibrated balanced accuracy `0.826389`
* post-burn-in `sequence_full` regime holdouts: mean calibrated balanced accuracy `0.508413`
* post-burn-in `sequence_full` regime-balanced split: calibrated balanced accuracy `0.650000`
* pre-relabel tiny patch Transformer regime-balanced split: `sequence_piezo_vlf_only`, calibrated balanced accuracy `0.637500`

## Limitations

The real VLF data is image-derived, not raw waveform data.

The current historical seismic sample covers January 1, 2024 through July 7, 2026. It is large enough for smoke baselines and shape diagnostics, but still not enough for robust predictive claims.

Simulation step time is an assumed mapping. Frequency-domain comparisons are therefore shape diagnostics, not physical frequency validation.

The current direct avalanche signal input uses `*.avalanche_signal.csv`; model smoke runs currently use the 20000-step seed `40`-`42` synthetic tables.
