"""Shared model feature groups, roles, and ablation definitions."""

from __future__ import annotations


TARGET_FIELDS = {
    "target_event_count",
    "target_magnitude_min",
    "target_occurred",
    "target_start_utc",
    "target_status",
    "target_end_utc",
}

ID_FIELDS = {
    "dataset_id",
    "window_id",
    "region_id",
    "window_start_utc",
    "window_end_utc",
    "source_file",
    "seismic_source_file",
    "vlf_image_source_file",
    "vlf_image_latest_source_file",
}

VLF_METADATA_PREFIXES = ("vlf_capture_", "vlf_latest_", "vlf_total_", "vlf_jpeg_")
VLF_IMAGE_PREFIXES = (
    "vlf_image_",
    "vlf_intensity_",
    "vlf_high_",
    "vlf_hot_",
    "vlf_column_",
    "vlf_vertical_",
    "vlf_band_",
    "vlf_pixel_",
    "vlf_crop_",
)
SYNTHETIC_PIEZO_VLF_PREFIXES = ("synthetic_piezo_vlf_",)

FEATURE_GROUP_PREFIXES = {
    "seismic": ("seismic_",),
    "astronomy": ("astro_",),
    "vlf": VLF_METADATA_PREFIXES + VLF_IMAGE_PREFIXES + SYNTHETIC_PIEZO_VLF_PREFIXES,
    "vlf_metadata": VLF_METADATA_PREFIXES,
    "vlf_image": VLF_IMAGE_PREFIXES,
    "synthetic_seismic": ("synthetic_seismic_",),
    "synthetic_piezo_vlf": SYNTHETIC_PIEZO_VLF_PREFIXES,
    "synthetic_direct_avalanche": ("synthetic_direct_avalanche_",),
    "synthetic_summary": ("synthetic_summary_",),
    "quality": ("quality_",),
}

FEATURE_ROLE_GROUPS = {
    "seismic": ("seismic", "synthetic_seismic", "synthetic_direct_avalanche"),
    "vlf": ("vlf_metadata", "vlf_image", "synthetic_piezo_vlf"),
    "astronomy": ("astronomy",),
    "quality": ("quality",),
}

ABLATIONS = {
    "seismic_only": ("seismic",),
    "vlf_only": ("vlf",),
    "seismic_vlf": ("seismic", "vlf_metadata", "vlf_image"),
    "seismic_vlf_unified": ("seismic", "vlf"),
    "seismic_astronomy": ("seismic", "astronomy"),
    "full_multimodal": ("seismic", "astronomy", "vlf_metadata", "vlf_image"),
    "synthetic_vlf_only": ("synthetic_piezo_vlf",),
    "synthetic_seismic_only": ("synthetic_seismic",),
    "synthetic_seismic_piezo_vlf": ("synthetic_seismic", "synthetic_piezo_vlf"),
    "synthetic_seismic_vlf_unified": ("synthetic_seismic", "vlf"),
    "synthetic_seismic_direct_avalanche": ("synthetic_seismic", "synthetic_direct_avalanche"),
    "synthetic_full": (
        "synthetic_seismic",
        "synthetic_piezo_vlf",
        "synthetic_direct_avalanche",
        "synthetic_summary",
    ),
}
