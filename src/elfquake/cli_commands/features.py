"""Feature construction CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.cli_commands.common import resolve_image_paths
from elfquake.features.astronomy import build_astronomy_features
from elfquake.features.design_matrix import build_design_matrix
from elfquake.features.multimodal_design import join_vlf_design_matrix
from elfquake.features.multimodal_smoke import build_multimodal_smoke_row
from elfquake.features.prospective import build_prospective_vlf_windows, update_prospective_vlf_table
from elfquake.features.prospective_report import summarize_prospective_table
from elfquake.features.table import build_multimodal_table_from_manifest, write_multimodal_manifest_template
from elfquake.features.targets import label_multimodal_targets
from elfquake.features.training_windows import build_seismic_training_windows
from elfquake.features.vlf import build_vlf_features
from elfquake.features.vlf_audio import build_vlf_audio_features
from elfquake.features.vlf_image import build_vlf_image_features
from elfquake.features.vlf_image_compare import compare_vlf_image_features
from elfquake.features.vlf_image_windows import join_vlf_image_features_to_windows
from elfquake.features.vlf_windows import build_vlf_window_features
from elfquake.features.italy_coverage import build_italy_coverage_report
from elfquake.features.vlf_event_association import build_vlf_event_association_report
from elfquake.features.spatial_targets import label_spatial_multimodal_targets


def register_feature_commands(subparsers: _SubParsersAction) -> None:
    coverage = subparsers.add_parser("build-italy-coverage-report")
    coverage.add_argument("--events", type=Path, required=True)
    coverage.add_argument("--vlf-metadata-root", type=Path)
    coverage.add_argument("--anomaly-scores", type=Path)
    coverage.add_argument("--out", type=Path, required=True)
    coverage.add_argument("--weekly-out", type=Path)
    coverage.set_defaults(func=_build_italy_coverage_report)
    association = subparsers.add_parser("build-vlf-event-association-report")
    association.add_argument("--events", type=Path, required=True)
    association.add_argument("--anomaly-scores", type=Path, required=True)
    association.add_argument("--out", type=Path, required=True)
    association.add_argument("--weekly-out", type=Path)
    association.add_argument("--permutations", type=int, default=2000)
    association.add_argument("--seed", type=int, default=42)
    association.set_defaults(func=_build_vlf_event_association_report)
    spatial = subparsers.add_parser("label-spatial-multimodal-targets")
    spatial.add_argument("--input", type=Path, required=True)
    spatial.add_argument("--events", type=Path, required=True)
    spatial.add_argument("--out", type=Path, required=True)
    spatial.add_argument("--as-of", required=True)
    spatial.add_argument("--catalog-end")
    spatial.add_argument("--cell-degrees", type=float, default=1.5)
    spatial.add_argument("--target-magnitude-min", type=float, default=2.5)
    spatial.set_defaults(func=_label_spatial_multimodal_targets)
    multimodal = subparsers.add_parser("build-multimodal-smoke")
    multimodal.add_argument("--events", type=Path, required=True)
    multimodal.add_argument("--vlf-metadata", type=Path, action="append", default=[])
    multimodal.add_argument("--astronomy-metadata", type=Path, action="append", default=[])
    multimodal.add_argument("--region-id", required=True)
    multimodal.add_argument("--window-start", required=True)
    multimodal.add_argument("--window-end", required=True)
    multimodal.add_argument("--target-end", required=True)
    multimodal.add_argument("--target-magnitude-min", default="3.0")
    multimodal.add_argument("--out", type=Path, required=True)
    multimodal.set_defaults(func=_build_multimodal_smoke)

    vlf_features = subparsers.add_parser("build-vlf-features")
    vlf_features.add_argument("--metadata", type=Path, action="append", default=[])
    vlf_features.add_argument("--window-start", required=True)
    vlf_features.add_argument("--window-end", required=True)
    vlf_features.add_argument("--out", type=Path, required=True)
    vlf_features.set_defaults(func=_build_vlf_features)

    vlf_window_features = subparsers.add_parser("build-vlf-window-features")
    vlf_window_features.add_argument("--training-windows", type=Path, required=True)
    vlf_window_features.add_argument("--metadata-root", type=Path, required=True)
    vlf_window_features.add_argument("--out", type=Path, required=True)
    vlf_window_features.set_defaults(func=_build_vlf_window_features)

    vlf_image_features = subparsers.add_parser("extract-vlf-image-features")
    vlf_image_features.add_argument("--image", type=Path, action="append", default=[])
    vlf_image_features.add_argument("--image-root", type=Path, action="append", default=[])
    vlf_image_features.add_argument("--filename-prefix", action="append", default=[])
    vlf_image_features.add_argument("--out", type=Path, required=True)
    vlf_image_features.add_argument("--crop-left", type=float, default=0.0)
    vlf_image_features.add_argument("--crop-top", type=float, default=0.13)
    vlf_image_features.add_argument("--crop-right", type=float, default=0.83)
    vlf_image_features.add_argument("--crop-bottom", type=float, default=0.95)
    vlf_image_features.set_defaults(func=_extract_vlf_image_features)

    vlf_image_compare = subparsers.add_parser("compare-vlf-image-features")
    vlf_image_compare.add_argument("--sim-image", type=Path, required=True)
    vlf_image_compare.add_argument("--real-image", type=Path, action="append", default=[])
    vlf_image_compare.add_argument("--real-image-root", type=Path, action="append", default=[])
    vlf_image_compare.add_argument("--filename-prefix", action="append", default=["last_E_VLF"])
    vlf_image_compare.add_argument("--out", type=Path, required=True)
    vlf_image_compare.add_argument("--sim-crop-left", type=float, default=0.0)
    vlf_image_compare.add_argument("--sim-crop-top", type=float, default=0.13)
    vlf_image_compare.add_argument("--sim-crop-right", type=float, default=1.0)
    vlf_image_compare.add_argument("--sim-crop-bottom", type=float, default=1.0)
    vlf_image_compare.add_argument("--real-crop-left", type=float, default=0.0)
    vlf_image_compare.add_argument("--real-crop-top", type=float, default=0.13)
    vlf_image_compare.add_argument("--real-crop-right", type=float, default=0.83)
    vlf_image_compare.add_argument("--real-crop-bottom", type=float, default=0.95)
    vlf_image_compare.set_defaults(func=_compare_vlf_image_features)

    vlf_image_join = subparsers.add_parser("join-vlf-image-features")
    vlf_image_join.add_argument("--windows", type=Path, required=True)
    vlf_image_join.add_argument("--image-features", type=Path, action="append", required=True)
    vlf_image_join.add_argument("--out", type=Path, required=True)
    vlf_image_join.add_argument("--exclude-window-end", action="store_true")
    vlf_image_join.set_defaults(func=_join_vlf_image_features)

    vlf_audio_features = subparsers.add_parser("extract-vlf-audio-features")
    vlf_audio_features.add_argument("--audio", type=Path, action="append", default=[])
    vlf_audio_features.add_argument("--audio-root", type=Path, action="append", default=[])
    vlf_audio_features.add_argument("--out", type=Path, required=True)
    vlf_audio_features.add_argument("--no-ffprobe", action="store_true")
    vlf_audio_features.set_defaults(func=_extract_vlf_audio_features)

    astro_features = subparsers.add_parser("build-astronomy-features")
    astro_features.add_argument("--metadata", type=Path, action="append", default=[])
    astro_features.add_argument("--window-start", required=True)
    astro_features.add_argument("--window-end", required=True)
    astro_features.add_argument("--out", type=Path, required=True)
    astro_features.set_defaults(func=_build_astronomy_features)

    label_targets = subparsers.add_parser("label-multimodal-targets")
    label_targets.add_argument("--input", type=Path, required=True)
    label_targets.add_argument("--events", type=Path, required=True)
    label_targets.add_argument("--as-of", required=True)
    label_targets.add_argument("--catalog-end")
    label_targets.add_argument("--out", type=Path, required=True)
    label_targets.set_defaults(func=_label_multimodal_targets)

    table = subparsers.add_parser("build-multimodal-table")
    table.add_argument("--manifest", type=Path, required=True)
    table.add_argument("--out", type=Path, required=True)
    table.set_defaults(func=_build_multimodal_table)

    table_template = subparsers.add_parser("write-multimodal-manifest-template")
    table_template.add_argument("--out", type=Path, required=True)
    table_template.set_defaults(func=_write_multimodal_manifest_template)

    prospective = subparsers.add_parser("build-prospective-vlf-windows")
    prospective.add_argument("--events", type=Path, required=True)
    prospective.add_argument("--vlf-metadata-root", type=Path, required=True)
    prospective.add_argument("--astronomy-metadata-root", type=Path, required=True)
    prospective.add_argument("--region-id", required=True)
    prospective.add_argument("--lookback-hours", type=int, default=24)
    prospective.add_argument("--horizon-days", type=int, default=7)
    prospective.add_argument("--min-anchor-gap-seconds", type=int, default=60)
    prospective.add_argument("--target-magnitude-min", default="3.0")
    prospective.add_argument("--out", type=Path, required=True)
    prospective.set_defaults(func=_build_prospective_vlf_windows)

    prospective_update = subparsers.add_parser("update-prospective-vlf-table")
    prospective_update.add_argument("--table", type=Path, required=True)
    prospective_update.add_argument("--events", type=Path, required=True)
    prospective_update.add_argument("--vlf-metadata-root", type=Path, required=True)
    prospective_update.add_argument("--astronomy-metadata-root", type=Path, required=True)
    prospective_update.add_argument("--region-id", required=True)
    prospective_update.add_argument("--lookback-hours", type=int, default=24)
    prospective_update.add_argument("--horizon-days", type=int, default=7)
    prospective_update.add_argument("--min-anchor-gap-seconds", type=int, default=60)
    prospective_update.add_argument("--target-magnitude-min", default="3.0")
    prospective_update.add_argument("--out", type=Path, required=True)
    prospective_update.set_defaults(func=_update_prospective_vlf_table)

    prospective_summary = subparsers.add_parser("summarize-prospective-table")
    prospective_summary.add_argument("--input", type=Path, required=True)
    prospective_summary.add_argument("--as-of")
    prospective_summary.add_argument("--out", type=Path, required=True)
    prospective_summary.set_defaults(func=_summarize_prospective_table)

    training = subparsers.add_parser("build-seismic-training-windows")
    training.add_argument("--events", type=Path, required=True)
    training.add_argument("--region-id", required=True)
    training.add_argument("--start", required=True)
    training.add_argument("--end", required=True)
    training.add_argument("--window-days", type=int, default=7)
    training.add_argument("--horizon-days", type=int, default=7)
    training.add_argument("--target-magnitude-min", default="3.0")
    training.add_argument("--out", type=Path, required=True)
    training.set_defaults(func=_build_seismic_training_windows)

    design = subparsers.add_parser("build-design-matrix")
    design.add_argument("--training-windows", type=Path, required=True)
    design.add_argument("--kp-ap", type=Path, required=True)
    design.add_argument("--f107", type=Path, required=True)
    design.add_argument("--out", type=Path, required=True)
    design.set_defaults(func=_build_design_matrix)

    vlf_design = subparsers.add_parser("join-vlf-design-matrix")
    vlf_design.add_argument("--design-matrix", type=Path, required=True)
    vlf_design.add_argument("--vlf-windows", type=Path, required=True)
    vlf_design.add_argument("--out", type=Path, required=True)
    vlf_design.set_defaults(func=_join_vlf_design_matrix)


def _build_multimodal_smoke(args: Namespace) -> int:
    row = build_multimodal_smoke_row(
        events_csv=args.events,
        vlf_metadata_paths=args.vlf_metadata,
        astronomy_metadata_paths=args.astronomy_metadata,
        region_id=args.region_id,
        window_start_utc=args.window_start,
        window_end_utc=args.window_end,
        target_end_utc=args.target_end,
        target_magnitude_min=args.target_magnitude_min,
        out_path=args.out,
    )
    print(f"wrote: {args.out}")
    print(f"window_id: {row['window_id']}")
    return 0


def _build_vlf_features(args: Namespace) -> int:
    row = build_vlf_features(
        metadata_paths=args.metadata,
        window_start_utc=args.window_start,
        window_end_utc=args.window_end,
        out_path=args.out,
    )
    print(f"wrote: {args.out}")
    print(f"vlf_capture_count: {row['vlf_capture_count']}")
    return 0


def _build_italy_coverage_report(args: Namespace) -> int:
    report = build_italy_coverage_report(
        events_csv=args.events,
        vlf_metadata_root=args.vlf_metadata_root,
        anomaly_scores_csv=args.anomaly_scores,
        out_path=args.out,
        weekly_out=args.weekly_out,
    )
    print(f"events: {report['events']['row_count']}")
    print(f"vlf metadata: {report['vlf_captures']['metadata_count']}")
    print(f"overlap weeks: {report['overlap']['weeks_with_both']}")
    print(f"output: {args.out}")
    return 0


def _build_vlf_event_association_report(args: Namespace) -> int:
    report = build_vlf_event_association_report(
        events_csv=args.events, anomaly_scores_csv=args.anomaly_scores,
        out_path=args.out, weekly_out=args.weekly_out,
        permutations=args.permutations, seed=args.seed,
    )
    print(f"status: {report['status']}")
    print(f"weekly VLF rows: {report['weekly_vlf_rows']}")
    print(f"output: {args.out}")
    return 0


def _label_spatial_multimodal_targets(args: Namespace) -> int:
    rows = label_spatial_multimodal_targets(
        input_csv=args.input, events_csv=args.events, out_path=args.out,
        as_of_utc=args.as_of, catalog_end_utc=args.catalog_end,
        cell_degrees=args.cell_degrees, target_magnitude_min=args.target_magnitude_min,
    )
    print(f"spatial target rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _build_vlf_window_features(args: Namespace) -> int:
    rows = build_vlf_window_features(
        training_windows_csv=args.training_windows,
        metadata_root=args.metadata_root,
        out_path=args.out,
    )
    print(f"vlf window rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _extract_vlf_image_features(args: Namespace) -> int:
    rows = build_vlf_image_features(
        image_paths=resolve_image_paths(
            image_paths=args.image,
            image_roots=args.image_root,
            filename_prefixes=args.filename_prefix,
        ),
        out_path=args.out,
        crop_left=args.crop_left,
        crop_top=args.crop_top,
        crop_right=args.crop_right,
        crop_bottom=args.crop_bottom,
    )
    print(f"image rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _compare_vlf_image_features(args: Namespace) -> int:
    row = compare_vlf_image_features(
        sim_image=args.sim_image,
        real_images=resolve_image_paths(
            image_paths=args.real_image,
            image_roots=args.real_image_root,
            filename_prefixes=args.filename_prefix,
        ),
        out_path=args.out,
        sim_crop_left=args.sim_crop_left,
        sim_crop_top=args.sim_crop_top,
        sim_crop_right=args.sim_crop_right,
        sim_crop_bottom=args.sim_crop_bottom,
        real_crop_left=args.real_crop_left,
        real_crop_top=args.real_crop_top,
        real_crop_right=args.real_crop_right,
        real_crop_bottom=args.real_crop_bottom,
    )
    print(f"real images: {row['real_image_count']}")
    print(f"nearest real: {row['nearest_real_image_file']}")
    print(f"nearest distance: {row['nearest_real_distance']}")
    print(f"output: {args.out}")
    return 0


def _join_vlf_image_features(args: Namespace) -> int:
    rows = join_vlf_image_features_to_windows(
        windows_csv=args.windows,
        image_features_csvs=args.image_features,
        out_path=args.out,
        include_window_end=not args.exclude_window_end,
    )
    print(f"joined rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _extract_vlf_audio_features(args: Namespace) -> int:
    audio_paths = _resolve_audio_paths(audio_paths=args.audio, audio_roots=args.audio_root)
    rows = build_vlf_audio_features(
        audio_paths=audio_paths,
        out_path=args.out,
        use_ffprobe=not args.no_ffprobe,
    )
    print(f"audio rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _build_astronomy_features(args: Namespace) -> int:
    row = build_astronomy_features(
        metadata_paths=args.metadata,
        window_start_utc=args.window_start,
        window_end_utc=args.window_end,
        out_path=args.out,
    )
    print(f"wrote: {args.out}")
    print(f"astro_capture_count: {row['astro_capture_count']}")
    return 0


def _label_multimodal_targets(args: Namespace) -> int:
    rows = label_multimodal_targets(
        input_csv=args.input,
        events_csv=args.events,
        as_of_utc=args.as_of,
        catalog_end_utc=args.catalog_end,
        out_path=args.out,
    )
    print(f"labeled rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _resolve_audio_paths(*, audio_paths: list[Path], audio_roots: list[Path]) -> list[Path]:
    resolved = list(audio_paths)
    for root in audio_roots:
        resolved.extend(sorted(root.glob("**/*.ogg")))
        resolved.extend(sorted(root.glob("**/*.oga")))
        resolved.extend(sorted(root.glob("**/*.wav")))
    unique = sorted(dict.fromkeys(resolved))
    if not unique:
        raise ValueError("at least one --audio or matching --audio-root file is required")
    return unique


def _build_multimodal_table(args: Namespace) -> int:
    rows = build_multimodal_table_from_manifest(manifest_path=args.manifest, out_path=args.out)
    print(f"built rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _write_multimodal_manifest_template(args: Namespace) -> int:
    write_multimodal_manifest_template(args.out)
    print(f"output: {args.out}")
    return 0


def _build_prospective_vlf_windows(args: Namespace) -> int:
    rows = build_prospective_vlf_windows(
        events_csv=args.events,
        vlf_metadata_root=args.vlf_metadata_root,
        astronomy_metadata_root=args.astronomy_metadata_root,
        region_id=args.region_id,
        lookback_hours=args.lookback_hours,
        horizon_days=args.horizon_days,
        min_anchor_gap_seconds=args.min_anchor_gap_seconds,
        target_magnitude_min=args.target_magnitude_min,
        out_path=args.out,
    )
    print(f"prospective rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _update_prospective_vlf_table(args: Namespace) -> int:
    report = update_prospective_vlf_table(
        table_path=args.table,
        events_csv=args.events,
        vlf_metadata_root=args.vlf_metadata_root,
        astronomy_metadata_root=args.astronomy_metadata_root,
        region_id=args.region_id,
        lookback_hours=args.lookback_hours,
        horizon_days=args.horizon_days,
        min_anchor_gap_seconds=args.min_anchor_gap_seconds,
        target_magnitude_min=args.target_magnitude_min,
        out_path=args.out,
    )
    print(f"existing rows: {report['existing_rows']}")
    print(f"candidate rows: {report['candidate_rows']}")
    print(f"new rows: {report['new_rows']}")
    print(f"refreshed rows: {report['refreshed_rows']}")
    print(f"retained rows: {report['retained_rows']}")
    print(f"total rows: {report['total_rows']}")
    print(f"output: {args.out}")
    return 0


def _summarize_prospective_table(args: Namespace) -> int:
    report = summarize_prospective_table(input_csv=args.input, as_of_utc=args.as_of, out_path=args.out)
    print(f"rows: {report['row_count']}")
    print(f"ready to label: {report['ready_to_label_count']}")
    print(f"missing vlf image features: {report['missing_vlf_image_features_count']}")
    print(f"output: {args.out}")
    return 0


def _build_seismic_training_windows(args: Namespace) -> int:
    rows = build_seismic_training_windows(
        events_csv=args.events,
        region_id=args.region_id,
        start_utc=args.start,
        end_utc=args.end,
        window_days=args.window_days,
        horizon_days=args.horizon_days,
        target_magnitude_min=args.target_magnitude_min,
        out_path=args.out,
    )
    print(f"training rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _build_design_matrix(args: Namespace) -> int:
    rows = build_design_matrix(
        training_windows_csv=args.training_windows,
        kp_ap_csv=args.kp_ap,
        f107_csv=args.f107,
        out_path=args.out,
    )
    print(f"design rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0


def _join_vlf_design_matrix(args: Namespace) -> int:
    rows = join_vlf_design_matrix(
        design_matrix_csv=args.design_matrix,
        vlf_windows_csv=args.vlf_windows,
        out_path=args.out,
    )
    print(f"design rows: {len(rows)}")
    print(f"output: {args.out}")
    return 0
