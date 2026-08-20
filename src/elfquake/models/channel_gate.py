"""Fail loudly on model channels that carry no usable information.

Two defects have already reached fitted models in this project and neither was
visible from row counts or from the fixture report:

* A **constant channel**. `astro_noaa_solar_cycle_f107_value` sat at `125.69`
  for every row of the table. The standardizer's zero-variance guard did not
  catch it, because floating-point accumulation left a scale of `1.4e-14`
  rather than exactly zero, so the column standardized to a constant `1.0` and
  acted as a second intercept. Had the held-out partition carried a *different*
  constant, every test prediction would have saturated.
* An **unmasked missing channel**. A channel that is blank on some rows while
  no mask flag fires on those rows is imputed silently downstream. The model
  cannot distinguish "measured zero" from "no measurement", and the missing-
  aware fixture's whole purpose is defeated.

The gate runs over the assembled fixture rows and raises rather than warns.
A channel that is genuinely constant by design has to be named explicitly, so
the decision is recorded instead of inferred.
"""

from __future__ import annotations

from dataclasses import dataclass


# A channel needs at least this many rows before "every value is identical" is
# evidence of anything rather than an accident of a tiny fixture.
MIN_ROWS_FOR_CONSTANT_CHECK = 8

# Values that mark a mask as firing.
MASK_SET_VALUES = {"1", "1.0", "true", "True"}


class ChannelGateError(ValueError):
    """Raised when a fixture contains channels that cannot inform a model."""


@dataclass(frozen=True)
class ChannelDefect:
    channel: str
    defect: str
    detail: str

    def describe(self) -> str:
        return f"{self.channel}: {self.defect} ({self.detail})"


def audit_channels(
    rows: list[dict[str, str]],
    *,
    channels: list[str],
    mask_fields: list[str],
    allow_constant: frozenset[str] = frozenset(),
) -> list[ChannelDefect]:
    """Report channels that are empty, constant, or missing without a mask."""
    defects = []
    for channel in channels:
        values = [row.get(channel, "") for row in rows]
        present = [value for value in values if value != ""]
        if not present:
            defects.append(
                ChannelDefect(
                    channel=channel,
                    defect="empty_channel",
                    detail=f"no value in any of {len(rows)} rows",
                )
            )
            continue
        if (
            channel not in allow_constant
            and len(present) >= MIN_ROWS_FOR_CONSTANT_CHECK
            and len(set(present)) == 1
        ):
            defects.append(
                ChannelDefect(
                    channel=channel,
                    defect="constant_channel",
                    detail=f"{len(present)} rows all equal {present[0]!r}",
                )
            )
        missing_indexes = [index for index, value in enumerate(values) if value == ""]
        if missing_indexes and not _covered_by_mask(rows, missing_indexes, mask_fields):
            defects.append(
                ChannelDefect(
                    channel=channel,
                    defect="unmasked_missing_channel",
                    detail=(
                        f"blank on {len(missing_indexes)} of {len(rows)} rows "
                        "with no mask field set on all of them"
                    ),
                )
            )
    return defects


def numeric_channels(rows: list[dict[str, str]], *, fieldnames: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """Fields under `prefixes` whose values are numeric, excluding mask fields.

    Non-numeric fields (source identifiers, timestamps, provenance strings) are
    not model channels and are deliberately out of scope.
    """
    channels = []
    for field in fieldnames:
        if not field.startswith(prefixes) or _is_mask_field(field):
            continue
        values = [row.get(field, "") for row in rows]
        present = [value for value in values if value != ""]
        if present and all(_is_numeric(value) for value in present):
            channels.append(field)
    return channels


def mask_fields(fieldnames: list[str]) -> list[str]:
    """Fields that assert a value is *missing*.

    Presence flags are excluded: a `..._present` column reads `0` where the
    channel is blank, so it can never be the thing that is set on the missing
    rows.
    """
    return [field for field in fieldnames if _is_missing_mask(field)]


def raise_for_defects(defects: list[ChannelDefect]) -> None:
    if not defects:
        return
    lines = "\n".join(f"  - {defect.describe()}" for defect in defects)
    raise ChannelGateError(
        f"{len(defects)} channel defect(s) would reach the model:\n{lines}\n"
        "Fix the source, drop the channel, or name it as allowed-constant."
    )


def _covered_by_mask(
    rows: list[dict[str, str]], missing_indexes: list[int], mask_fields_: list[str]
) -> bool:
    """True when one mask field is set on every row where the channel is blank."""
    for field in mask_fields_:
        if all(rows[index].get(field, "") in MASK_SET_VALUES for index in missing_indexes):
            return True
    return False


def _is_mask_field(field: str) -> bool:
    return field.startswith("quality_")


def _is_missing_mask(field: str) -> bool:
    return field.startswith("quality_missing_") or (
        field.startswith("quality_") and field.endswith("_missing")
    )


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
