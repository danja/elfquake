# Japan Earthquake VLF Window

## Scope

This exploratory check covers the 14 days before the 28 July 2026 Kyushu earthquake. The local Japan seismic refresh found a main event at `2026-07-28T07:27:15Z`, magnitude `6.8`, depth `10 km`, at `32.6817, 130.722`, followed by a `M5.6` event and smaller aftershocks. NIED Hi-net lists the corresponding local-time event at `2026/07/28 16:27:15 JST`; contemporary reporting described it as approximately M7.1. See [Hi-net](https://www.hinet.bosai.go.jp/backnumber/?LANG=en) and [AP News](https://apnews.com/article/japan-earthquake-tsunami-09e6f40acbcc96053946c9c104e7a242).

The official archive now lists hourly Moshiri CDFs through 29 July, including the event day. The local analysis has processed all 24 hourly files for 28 July and all 24 for 29 July, each with 8,646 spectral rows and both channels. One 30 July file is also available; the remaining archive coverage should still be treated as opportunistic rather than continuous station monitoring.

## Method

For each file, the analysis used median channel active fraction and median log-power in three non-floor spectral bands. Baselines were the available 15–24 July captures. A robust deviation score used the baseline median absolute deviation, with small scale floors to prevent near-zero variation from dominating. Power-floor values were excluded. The extended diagnostic is in `data/derived/reports/japan_2026-07-28_vlf_anomaly_check_extended.csv`.

## Results

| Date | Composite deviation | Interpretation |
|---|---:|---|
| 15 Jul | 0.66 | ordinary baseline variation |
| 18 Jul | 2.67 | elevated |
| 20 Jul | 1.69 | moderately elevated |
| 23 Jul | 1.47 | moderate |
| 24 Jul | 2.15 | elevated |
| 25 Jul | 2.26 | elevated |
| 26 Jul | 3.77 | highest in this sample |
| 27 Jul | 1.52 | moderate, not continuing upward |

July 26 is worth retaining as a candidate anomaly day, but it is not conclusive. Similar deviations occurred earlier, and the score fell on July 27 rather than forming a clear sustained pre-event progression. With one station, sparse daily sampling, no 28 July file, and no matched non-earthquake control period, this result does not establish a VLF precursor.

### New 27 July sample

Using the same robust baseline idea with power-floor values excluded, the 27 July 23:00 sample has a supplementary composite deviation score of `2.322`. It is elevated, and falls about 8 hours before the main event, but it is not the largest score: 15 July 12:00 scored `2.657` and 16 July 12:00 scored `2.345`. The timing is therefore compatible with a candidate anomaly but does not distinguish the earthquake from ordinary or instrumental variation. The detailed values are in `data/derived/reports/japan_2026-07-28_vlf_anomaly_check.csv`.

### Complete event-day samples

Across the complete 28 July series, the extended hourly score ranges from `1.529` to `5.418`; the maximum occurs at 13:00 UTC, after the `07:27:15Z` main event. The 00:00--06:00 pre-event values range from `1.950` to `3.735`. The highest 27 July value in the expanded report is `6.305` at 18:00 UTC, before the event.

The event-day observations are therefore elevated but not event-specific. The complete hourly coverage also shows that the event-day maximum is post-event, while a higher pre-event value occurred on 27 July. This does not support a precursor claim.

## Next Check

Repeat the analysis with a longer seasonal baseline, control days matched for receiver operation and local-time distribution, and an explicit pre-/post-event comparison. Keep the Japan CDF and derived products restricted to the permitted scientific research use.
