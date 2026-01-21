import pandas as pd
import pytest

from revoletion import time


class TestTimestep:
    @pytest.mark.parametrize(
        "timestep_str,expected_hours",
        [
            ("1H", 1.0),
            ("2H", 2.0),
            ("30T", 0.5),
            ("30min", 0.5),
            ("15T", 0.25),
            ("1D", 24.0),
            ("12H", 12.0),
            ("45min", 0.75),
            ("90S", 0.025),
        ],
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
        [
            ("1H", 1.0),
            ("2H", 2.0),
            ("30T", 0.5),
            ("15T", 0.25),
            ("1D", 24.0),
            ("12H", 12.0),
        ],
    )
    def test_from_dti_with_freqstr(self, freq: str, expected_hours: float):
        """Test from_dti when DatetimeIndex has freqstr populated."""
        dti = pd.date_range("2024-01-01", periods=10, freq=freq)

        result = time.Timestep.from_dti(dti)

        assert result.hours == pytest.approx(expected_hours)
        assert isinstance(result.td, pd.Timedelta)

    @pytest.mark.parametrize(
        "freq,expected_hours",
        [
            ("1H", 1.0),
            ("2H", 2.0),
            ("30T", 0.5),
            ("6H", 6.0),
            ("1D", 24.0),
        ],
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
