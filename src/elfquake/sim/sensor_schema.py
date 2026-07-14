"""Stable CSV field definitions for simulated sensor modalities."""

PIEZO_SENSOR_FIELDS = [
    "step", "sensor_id", "x", "y", "piezo_signal", "piezo_potential_signal",
    "piezo_total_source", "piezo_potential_total_source", "near_critical_cell_count",
    "near_critical_contact_count", "near_critical_coherence", "near_critical_weighted_stress",
    "critical_cell_count", "nearest_critical_distance", "max_stress_ratio", "piezo_charge_total",
    "piezo_charge_max", "piezo_release_total", "damage_total", "damage_max", "damage_active_cell_count",
    "damage_local_mean", "damage_local_max", "damage_local_active_fraction", "damage_local_std",
    "mature_weakness_total", "mature_weakness_max", "mature_weakness_active_cell_count",
]

AVALANCHE_SIGNAL_SENSOR_FIELDS = [
    "step", "sensor_id", "x", "y", "avalanche_signal", "avalanche_total_source",
    "active_topple_cell_count", "max_local_topple", "nearest_topple_distance", "stress_drop_total",
    "stress_drop_max", "avalanche_release_total",
]
