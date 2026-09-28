"""Phase 3a acceptance tests: Synthetic sensor forward model."""
import numpy as np
import pytest

from src.forward import (
    add_60hz_pickup,
    add_dropout,
    add_white_noise,
    generate_cute_sensors,
)

oft_available = True
try:
    from src.forward import flux_loop, full_diagnostic_set, mirnov_probe
except ImportError:
    oft_available = False


# --- Sensor config tests ---


def test_sensor_config_counts():
    """[3a.1] 39 flux loops and 25 probes: the 32 probe entries minus 7 exact repeats."""
    config = generate_cute_sensors()
    assert config.n_flux_loops == 39
    assert config.n_mirnov_probes == 25
    ids = [s["id"] for s in config.flux_loops + config.mirnov_probes]
    assert len(set(ids)) == len(ids), "sensor ids must be unique"


def test_probe_angle_follows_table_direction():
    """A probe's angle points along its (N_R, N_Z) from the diagnostics table."""
    config = generate_cute_sensors()
    probes = {s["id"]: s for s in config.mirnov_probes}
    # S07 is (-0.9907, -0.1359); S07@180 has N_R flipped to +0.9907.
    assert np.cos(probes["S07"]["angle"]) == pytest.approx(-0.9907, abs=1e-4)
    assert np.sin(probes["S07"]["angle"]) == pytest.approx(-0.1359, abs=1e-4)
    assert np.cos(probes["S07_180"]["angle"]) == pytest.approx(0.9907, abs=1e-4)
    assert np.cos(probes["S13"]["angle"]) == pytest.approx(1.0)


def _polygon_contains(pt, poly) -> bool:
    x, y = pt
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _distance_to_polygon(pt, poly) -> float:
    p = np.asarray(pt)
    best = np.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):
        a, b = np.asarray(a), np.asarray(b)
        t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * (b - a)))))
    return best


def test_sensors_sit_on_the_vessel():
    """No sensor is inside the vacuum chamber, and all are within 1.5 cm of the vessel.

    Sensors mounted on the vessel land about 0.5 cm outside its outer surface,
    so a mistyped digit or a meters-for-centimeters slip in the diagnostics
    table would show up here. FS22 and FS25 (R = 0.162 m) sit 2 to 3 mm inside
    the 5 mm wall itself, next to neighbours at 0.165 m; that is either a
    groove in the wall or rounding in the table, so the test only requires
    sensors to be outside the chamber's inner surface.
    """
    import json
    from pathlib import Path

    geom = json.loads((Path(__file__).parents[1] / "config" / "CUTE_geom.json").read_text())
    inner = [tuple(p) for p in geom["vv"]["inner_contour"]]
    outer = [tuple(p) for p in geom["vv"]["outer_contour"]]
    config = generate_cute_sensors()
    for s in config.flux_loops + config.mirnov_probes:
        pt = (s["R"], s["Z"])
        assert not _polygon_contains(pt, inner), f"{s['id']} is inside the vacuum chamber"
        gap = _distance_to_polygon(pt, outer)
        assert gap < 0.015, f"{s['id']} is {100 * gap:.1f} cm from the vessel"


# --- Noise model tests ---


def test_white_noise_statistics():
    """[3a.5] White noise has correct mean and std."""
    rng = np.random.default_rng(42)
    clean = np.zeros(10000)
    noisy = add_white_noise(clean, sigma=0.1, rng=rng)
    assert abs(np.mean(noisy)) < 0.01
    assert abs(np.std(noisy) - 0.1) < 0.01


def test_60hz_pickup_frequency():
    """[3a.6] 60 Hz pickup produces correct frequency peak."""
    fs = 10000.0
    n = 100000
    t = np.linspace(0, n / fs, n, endpoint=False)
    clean = np.zeros(n)
    noisy = add_60hz_pickup(clean, amplitude=1.0, timestamps=t)

    fft = np.fft.rfft(noisy)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    peak_freq = freqs[np.argmax(np.abs(fft))]
    assert abs(peak_freq - 60.0) <= 1.0


def test_dropout_rate():
    """[3a.7] Dropout produces NaN at expected rate."""
    rng = np.random.default_rng(42)
    clean = np.ones(10000)
    noisy = add_dropout(clean, probability=0.05, rng=rng)
    nan_fraction = np.isnan(noisy).mean()
    assert abs(nan_fraction - 0.05) < 0.02


# --- OFT-dependent tests ---


@pytest.mark.skipif(not oft_available, reason="OFT not installed")
def test_flux_loop_values(tokamaker_session):
    """[3a.2] Flux loop forward model returns physically reasonable psi values."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)
    psi_eval = mygs.get_field_eval("psi")

    config = generate_cute_sensors()
    for sensor in config.flux_loops[:10]:
        val = flux_loop(psi_eval, (sensor["R"], sensor["Z"]))
        assert not np.isnan(val), f"NaN psi at {sensor['id']}"
        assert abs(val) < 1.0, f"psi={val} unreasonably large at {sensor['id']}"


@pytest.mark.skipif(not oft_available, reason="OFT not installed")
def test_mirnov_probe_values(tokamaker_session):
    """[3a.3] Mirnov probe forward model returns physically reasonable B values."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)
    B_eval = mygs.get_field_eval("B")

    config = generate_cute_sensors()
    for sensor in config.mirnov_probes[:10]:
        val = mirnov_probe(B_eval, (sensor["R"], sensor["Z"]), sensor["angle"])
        assert not np.isnan(val), f"NaN B at {sensor['id']}"
        assert abs(val) < 5.0, f"B={val} T unreasonably large at {sensor['id']}"


@pytest.mark.skipif(not oft_available, reason="OFT not installed")
def test_full_diagnostic_set_shape(tokamaker_session):
    """[3a.4] Full diagnostic set returns correct number of columns."""
    from tests.conftest import _solve_reference

    mygs = tokamaker_session
    _solve_reference(mygs)
    config = generate_cute_sensors()
    df = full_diagnostic_set(mygs, config, time_index=0.0)

    assert len(df) == 1
    n_expected = config.n_total + 1  # +1 for time column
    assert len(df.columns) == n_expected, f"Got {len(df.columns)}, expected {n_expected}"


def test_ip_estimate_from_probes():
    """The filament fit recovers a centered plasma's current from the probes alone."""
    from src.ml.dataset import SensorLayout, forward_signals
    from src.reconstruct.constraints import estimate_ip_from_mirnov, sensor_ids_ordered

    config = generate_cute_sensors()
    layout = SensorLayout.from_config(config)
    signals = forward_signals(1.5e5, 0.32, 0.0, 0.12, layout)
    measurements = dict(zip(sensor_ids_ordered(config), signals))
    assert estimate_ip_from_mirnov(measurements, config) == pytest.approx(1.5e5, rel=0.1)
