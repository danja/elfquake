# Smoke Baseline Run

Naive historical-rate baseline smoke output.

## Input

`data/derived/baseline/central_italy_7d_m3_smoke.input.csv`

## Output

`data/derived/baseline/central_italy_7d_m3_smoke.baseline.csv`

## Method

The intended baseline probability is:

```text
historical_positive_count / historical_window_count
```

For this smoke run, there are no prior training windows. The output therefore records `status=insufficient_history` and leaves `predicted_probability` empty.

## Result

| Field | Value |
| --- | --- |
| region | `central_italy` |
| target | magnitude `>= 3.0` in 7 days |
| historical windows | `0` |
| historical positive windows | `0` |
| predicted probability | unavailable |
| predicted label | `0` |
| observed target | `0` |
| status | `insufficient_history` |

## Interpretation

This run validates the baseline output shape only. A real historical-rate baseline needs multiple earlier windows from the same region before the validation window.
