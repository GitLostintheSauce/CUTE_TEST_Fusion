"""Phase 3b acceptance tests: Signal processing library."""
import numpy as np

from src.signal import ProcessingConfig, RawSignal, process_signal
from src.signal.calibrate import calibrate
from src.signal.filters import bandpass, notch
from src.signal.integrate import integrate_with_drift_correction
from src.signal.outliers import fix_dropouts

FS = 100000.0  # 100 kHz sampling rate
DURATION = 1.0  # 1 second (needed for notch filter resolution)
N = int(FS * DURATION)
T = np.linspace(0, DURATION, N, endpoint=False)


def _make_raw(values: np.ndarray, sensor_type: str = "flux_loop", ch: str = "FL01") -> RawSignal:
    return RawSignal(
        timestamps=T,
        values=values,
        channel_id=ch,
        sensor_type=sensor_type,
    )


# --- Calibration ---


def test_calibration():
    """[3b.1] Calibration applies gain and offset correctly."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = calibrate(values, gain=2.0, offset=0.5)
    expected = (values - 0.5) / 2.0
    np.testing.assert_array_almost_equal(result, expected)


# --- Bandpass ---


def test_bandpass():
    """[3b.2] Bandpass filter removes out-of-band content."""
    # In-band: 100 Hz, out-of-band: 5000 Hz
    sig_100 = np.sin(2 * np.pi * 100 * T)
    sig_5000 = np.sin(2 * np.pi * 5000 * T)
    combined = sig_100 + sig_5000

    filtered = bandpass(combined, FS, low_cutoff=50, high_cutoff=500)

    # Check power at 5000 Hz is suppressed
    fft_input = np.fft.rfft(combined)
    fft_output = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(N, 1.0 / FS)

    idx_5000 = np.argmin(np.abs(freqs - 5000))
    idx_100 = np.argmin(np.abs(freqs - 100))

    power_5000_in = np.abs(fft_input[idx_5000]) ** 2
    power_5000_out = np.abs(fft_output[idx_5000]) ** 2
    power_100_in = np.abs(fft_input[idx_100]) ** 2
    power_100_out = np.abs(fft_output[idx_100]) ** 2

    assert power_5000_out < 0.01 * power_5000_in, "5000 Hz not sufficiently attenuated"
    assert power_100_out > 0.95 * power_100_in, "100 Hz too attenuated"


# --- Notch ---


def test_notch_filter():
    """[3b.3] Notch filter removes 60 Hz and harmonics."""
    # Broadband + 60/120/180 Hz tones
    np.random.seed(42)
    broadband = np.random.randn(N) * 0.1
    tone_60 = np.sin(2 * np.pi * 60 * T)
    tone_120 = np.sin(2 * np.pi * 120 * T)
    tone_180 = np.sin(2 * np.pi * 180 * T)
    combined = broadband + tone_60 + tone_120 + tone_180

    filtered = notch(combined, FS, freqs=[60.0, 120.0, 180.0])

    fft_in = np.fft.rfft(combined)
    fft_out = np.fft.rfft(filtered)
    freqs_arr = np.fft.rfftfreq(N, 1.0 / FS)

    for freq in [60, 120, 180]:
        idx = np.argmin(np.abs(freqs_arr - freq))
        power_in = np.abs(fft_in[idx]) ** 2
        power_out = np.abs(fft_out[idx]) ** 2
        ratio_db = 10 * np.log10(power_out / max(power_in, 1e-30))
        assert ratio_db < -20, f"Power at {freq} Hz reduced by only {ratio_db:.1f} dB"


# --- Phase preservation ---


def test_phase_preservation():
    """[3b.4] Bandpass filter preserves phase (cross-correlation > 0.99)."""
    sig = np.sin(2 * np.pi * 200 * T)  # 200 Hz, in-band
    filtered = bandpass(sig, FS, low_cutoff=50, high_cutoff=500)

    # Normalize and cross-correlate
    sig_norm = sig / np.linalg.norm(sig)
    filt_norm = filtered / np.linalg.norm(filtered)
    cross_corr = np.abs(np.dot(sig_norm, filt_norm))
    assert cross_corr > 0.99, f"Cross-correlation = {cross_corr:.4f}, expected > 0.99"


# --- Dropout interpolation ---


def test_dropout_interpolation():
    """[3b.5] Dropout detection fills NaN gaps within 10% of original."""
    clean = np.sin(2 * np.pi * 50 * T)
    noisy = clean.copy()
    # Set 5% to NaN
    rng = np.random.default_rng(42)
    dropout_idx = rng.choice(N, size=int(0.05 * N), replace=False)
    noisy[dropout_idx] = np.nan

    fixed = fix_dropouts(noisy, T)

    assert not np.any(np.isnan(fixed)), "NaN values remain after dropout fix"
    error = np.abs(fixed[dropout_idx] - clean[dropout_idx])
    max_error = np.max(error) / np.ptp(clean)
    assert max_error < 0.10, f"Max interpolation error {max_error:.3f} > 10% of signal range"


# --- Integration drift correction ---


def test_integration_drift():
    """[3b.6] Integration with drift correction recovers B from dB/dt."""
    # Use shorter segment for integration test
    t_int = np.linspace(0, 0.1, 10000, endpoint=False)
    # True B(t): a sine wave
    B_true = np.sin(2 * np.pi * 50 * t_int)
    # dB/dt: derivative
    dbdt = 2 * np.pi * 50 * np.cos(2 * np.pi * 50 * t_int)
    # Add linear drift
    dbdt_drifted = dbdt + 0.5  # constant drift in dB/dt -> linear drift in B

    B_recovered = integrate_with_drift_correction(dbdt_drifted, t_int, baseline_window=500)

    # Compare recovered B to true B (normalize both to zero mean for fair comparison)
    # Use middle 60% to avoid edge effects from integration
    n = len(t_int)
    sl = slice(n // 5, 4 * n // 5)
    B_true_zm = B_true[sl] - np.mean(B_true[sl])
    B_rec_zm = B_recovered[sl] - np.mean(B_recovered[sl])

    rms_error = np.sqrt(np.mean((B_rec_zm - B_true_zm) ** 2))
    rms_signal = np.sqrt(np.mean(B_true_zm**2))
    relative_error = rms_error / rms_signal
    assert relative_error < 0.15, f"Relative RMS error {relative_error:.3f} > 15%"


# --- End-to-end ---


def test_end_to_end_recovery():
    """[3b.7] Noisy synthetic -> processed recovers within 5% RMS for 90% of channels."""
    config = ProcessingConfig(
        gain=1.0,
        offset=0.0,
        bandpass_low=10.0,
        bandpass_high=5000.0,
        notch_freqs=[60.0, 120.0, 180.0],
        sample_rate=FS,
    )

    rng = np.random.default_rng(42)
    n_channels = 20
    pass_count = 0

    for i in range(n_channels):
        # Signal frequencies well within passband and away from notch freqs
        freq = 200 + i * 100  # 200 to 2100 Hz (well within 10-5000 Hz band)
        clean = np.sin(2 * np.pi * freq * T)

        # Add moderate noise
        noisy = clean.copy()
        noisy += rng.normal(0, 0.02, N)  # small white noise
        noisy += 0.2 * np.sin(2 * np.pi * 60 * T)  # 60 Hz pickup
        noisy += 0.1 * np.sin(2 * np.pi * 120 * T)  # 120 Hz harmonic

        raw = _make_raw(noisy, ch=f"FL{i:02d}")
        processed = process_signal(raw, config)

        # Use middle 80% to avoid edge transients from filtering
        n = len(T)
        sl = slice(n // 10, 9 * n // 10)
        rms_error = np.sqrt(np.mean((processed.values[sl] - clean[sl]) ** 2))
        rms_signal = np.sqrt(np.mean(clean[sl] ** 2))
        if rms_signal > 0 and rms_error / rms_signal < 0.05:
            pass_count += 1

    fraction = pass_count / n_channels
    assert fraction >= 0.90, f"Only {fraction*100:.0f}% channels within 5%, need 90%"


# --- Config file ---


def test_processing_config_from_toml():
    """[3b.8] Config file exists and has required keys."""
    import tomllib

    with open("config/processing.toml", "rb") as f:
        d = tomllib.load(f)

    assert "bandpass" in d
    assert "low_cutoff" in d["bandpass"]
    assert "high_cutoff" in d["bandpass"]
    assert "notch" in d


# --- Processing log ---


def test_processing_log():
    """[3b.9] Processing log records what was applied."""
    config = ProcessingConfig(sample_rate=FS)
    clean = np.sin(2 * np.pi * 100 * T)
    raw = _make_raw(clean)
    processed = process_signal(raw, config)

    assert len(processed.processing_log) > 0
    assert "calibrate" in processed.processing_log
    assert "bandpass" in processed.processing_log
    assert "notch" in processed.processing_log
