# Japan Earthquake VLF Window

## Scope

This exploratory check covers the 14 days before the 28 July 2026 Kyushu earthquake. The local Japan seismic refresh found a main event at `2026-07-28T07:27:15Z`, magnitude `6.8`, depth `10 km`, at `32.6817, 130.722`, followed by a `M5.6` event and smaller aftershocks. NIED Hi-net lists the corresponding local-time event at `2026/07/28 16:27:15 JST`; contemporary reporting described it as approximately M7.1. See [Hi-net](https://www.hinet.bosai.go.jp/backnumber/?LANG=en) and [AP News](https://apnews.com/article/japan-earthquake-tsunami-09e6f40acbcc96053946c9c104e7a242).

The VLF sample contains 13 daily Moshiri CDF captures at 00:00 UTC from 15–27 July, each with 8,646 spectral rows and both channels. The ISEE archive had not yet exposed 28 July files when this analysis ran, so there is no direct VLF observation of the mainshock or its immediate aftermath.

## Method

For each daily file, the analysis used median channel active fraction and median log-power in three non-floor spectral bands. Baselines were the daily medians from 15–24 July. A robust deviation score used the median absolute deviation of those baseline days. This avoids treating the CDF power-floor bands as meaningful amplitude changes.

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

## Next Check

When the archive exposes 28 July, acquire the full event-day and post-event files. Then repeat the analysis with hourly coverage, a longer seasonal baseline, and control days matched for receiver operation and local-time distribution. Keep the Japan CDF and derived products restricted to the permitted scientific research use.
