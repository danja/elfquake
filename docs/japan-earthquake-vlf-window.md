# Japan Earthquake VLF Window

## Scope

This exploratory check covers the 14 days before the 28 July 2026 Kyushu earthquake. The local Japan seismic refresh found a main event at `2026-07-28T07:27:15Z`, magnitude `6.8`, depth `10 km`, at `32.6817, 130.722`, followed by a `M5.6` event and smaller aftershocks. NIED Hi-net lists the corresponding local-time event at `2026/07/28 16:27:15 JST`; contemporary reporting described it as approximately M7.1. See [Hi-net](https://www.hinet.bosai.go.jp/backnumber/?LANG=en) and [AP News](https://apnews.com/article/japan-earthquake-tsunami-09e6f40acbcc96053946c9c104e7a242).

The official archive now lists hourly Moshiri CDFs through 29 July, including the event day. The local analysis has processed the new 28 July 00:00--06:00 UTC files, each with 8,646 spectral rows and both channels. The 07:00 UTC file and later event-day/post-event files remain listed but are not yet processed.

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

### New event-day samples

| UTC hour | Extended composite score |
|---|---:|
| 28 Jul 00:00 | 3.091 |
| 28 Jul 01:00 | 4.587 |
| 28 Jul 02:00 | 2.891 |
| 28 Jul 03:00 | 3.159 |
| 28 Jul 04:00 | 4.584 |
| 28 Jul 05:00 | 5.538 |
| 28 Jul 06:00 | 4.214 |

These pre-event observations are elevated, but 27 July reached higher scores, including `8.772` at 18:00 UTC. The event-day values therefore do not isolate a clear precursor, and the missing 07:00--post-event interval prevents a complete event-aligned comparison.

## Next Check

Process the remaining 28--29 July files, then repeat the analysis with complete hourly coverage, a longer seasonal baseline, and control days matched for receiver operation and local-time distribution. Keep the Japan CDF and derived products restricted to the permitted scientific research use.
