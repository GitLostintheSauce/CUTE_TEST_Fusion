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
    """[3a.1] Sensor config has >= 56 flux loops and >= 74 Mirnov probes."""
    config = generate_cute_sensors()
    assert config.n_flux_loops >= 56, f"Got {config.n_flux_loops} flux loops, need >= 56"
    assert config.n_mirnov_probes >= 74, f"Got {config.n_mirnov_probes} Mirnov probes, need >= 74"


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
