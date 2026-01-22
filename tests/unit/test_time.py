import pandas as pd
import pytest

from revoletion import time

TESTFREQS = [
    ("90s", 0.025),
    ("15min", 0.25),
    ("12h", 12.0),
    ("2D", 48.0),
]


class TestTimestep:
    @pytest.mark.parametrize(
        "timestep_str,expected_hours",
        TESTFREQS,
    )
    def test_from_str_valid_inputs(self, timestep_str: str, expected_hours: float):
        """Test from_str with various valid pandas timedelta strings."""
        result = time.Timestep.from_str(timestep_str)

        assert result.hours == pytest.approx(expected_hours)
        assert isinstance(result.td, pd.Timedelta)
        assert result.td == pd.Timedelta(timestep_str)

    def test_from_str_invalid_input(self):
        """Test from_str with an invalid timedelta string."""
        with pytest.raises(ValueError):
            _ = time.Timestep.from_str("invalid")

    @pytest.mark.parametrize(
        "freq,expected_hours",
        TESTFREQS,
    )
    def test_from_dti_with_freqstr(self, freq: str, expected_hours: float):
        """Test from_dti when DatetimeIndex has freqstr populated."""
        dti = pd.date_range("2024-01-01", periods=10, freq=freq)

        result = time.Timestep.from_dti(dti)

        assert result.hours == pytest.approx(expected_hours)
        assert isinstance(result.td, pd.Timedelta)

    @pytest.mark.parametrize(
        "freq,expected_hours",
        TESTFREQS,
    )
    def test_from_dti_without_freq_regular(self, freq: str, expected_hours: float):
        """Test from_dti when DatetimeIndex lacks freqstr but is regular."""
        dti = pd.date_range("2024-01-01", periods=10, freq=freq)
        # Remove freq by converting to Series and back
        dti_no_freq = pd.DatetimeIndex(dti.to_series().values)
        assert dti_no_freq.freqstr is None

        result = time.Timestep.from_dti(dti_no_freq)

        assert result.hours == pytest.approx(expected_hours)
        assert isinstance(result.td, pd.Timedelta)
