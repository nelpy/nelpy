import numpy as np
import pytest

from nelpy import formatters


class TestBaseFormatter:
    """Test the BaseFormatter class."""

    def test_base_formatter_initialization(self):
        """Test BaseFormatter initialization raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            formatters.BaseFormatter(42)


class TestArbitraryFormatter:
    """Test the ArbitraryFormatter class."""

    def test_arbitrary_formatter_initialization(self):
        """Test ArbitraryFormatter initialization"""
        formatter = formatters.ArbitraryFormatter(42.5)
        assert formatter.val == 42.5
        assert formatter.base_unit == "a.u."

    def test_arbitrary_formatter_str(self):
        """Test ArbitraryFormatter string representation"""
        formatter = formatters.ArbitraryFormatter(42.5)
        assert str(formatter) == "42.5"

    def test_arbitrary_formatter_repr(self):
        """Test ArbitraryFormatter repr"""
        formatter = formatters.ArbitraryFormatter(42.5)
        assert repr(formatter) == "42.5"

    def test_arbitrary_formatter_with_zero(self):
        """Test ArbitraryFormatter with zero value"""
        formatter = formatters.ArbitraryFormatter(0)
        assert str(formatter) == "0"

    def test_arbitrary_formatter_with_negative(self):
        """Test ArbitraryFormatter with negative value"""
        formatter = formatters.ArbitraryFormatter(-42.5)
        assert str(formatter) == "-42.5"


class TestPrettyBytes:
    """Test the PrettyBytes class."""

    def test_pretty_bytes_initialization(self):
        """Test PrettyBytes initialization"""
        formatter = formatters.PrettyBytes(1024)
        assert formatter.val == 1024
        assert formatter.base_unit == "bytes"

    def test_pretty_bytes_bytes(self):
        """Test PrettyBytes for bytes (< 1024)"""
        formatter = formatters.PrettyBytes(512)
        assert str(formatter) == "512 bytes"

    def test_pretty_bytes_kilobytes(self):
        """Test PrettyBytes for kilobytes"""
        formatter = formatters.PrettyBytes(2048)
        assert str(formatter) == "2.000 kilobytes"

    def test_pretty_bytes_megabytes(self):
        """Test PrettyBytes for megabytes"""
        formatter = formatters.PrettyBytes(2 * 1024**2)
        assert str(formatter) == "2.000 megabytes"

    def test_pretty_bytes_gigabytes(self):
        """Test PrettyBytes for gigabytes"""
        formatter = formatters.PrettyBytes(2 * 1024**3)
        assert str(formatter) == "2.000 gigabytes"

    def test_pretty_bytes_repr(self):
        """Test PrettyBytes repr"""
        formatter = formatters.PrettyBytes(1024)
        assert repr(formatter) == "1.000 kilobytes"

    def test_pretty_bytes_zero(self):
        """Test PrettyBytes with zero"""
        formatter = formatters.PrettyBytes(0)
        assert str(formatter) == "0 bytes"


class TestPrettyInt:
    """Test the PrettyInt class."""

    def test_pretty_int_initialization(self):
        """Test PrettyInt initialization"""
        formatter = formatters.PrettyInt(1000)
        assert formatter.val == 1000
        assert formatter.base_unit == "int"

    def test_pretty_int_str(self):
        """Test PrettyInt string representation"""
        formatter = formatters.PrettyInt(1000)
        assert str(formatter) == "1,000"

    def test_pretty_int_repr(self):
        """Test PrettyInt repr"""
        formatter = formatters.PrettyInt(1000)
        assert repr(formatter) == "1,000"

    def test_pretty_int_small_number(self):
        """Test PrettyInt with small number"""
        formatter = formatters.PrettyInt(42)
        assert str(formatter) == "42"

    def test_pretty_int_large_number(self):
        """Test PrettyInt with large number"""
        formatter = formatters.PrettyInt(1234567)
        assert str(formatter) == "1,234,567"

    def test_pretty_int_zero(self):
        """Test PrettyInt with zero"""
        formatter = formatters.PrettyInt(0)
        assert str(formatter) == "0"


class TestPrettyDuration:
    """Test the PrettyDuration class."""

    def test_pretty_duration_initialization(self):
        """Test PrettyDuration initialization"""
        duration = formatters.PrettyDuration(60.0)
        assert duration == 60.0
        assert duration.base_unit == "s"

    def test_pretty_duration_seconds(self):
        """Test PrettyDuration for seconds"""
        duration = formatters.PrettyDuration(30.5)
        assert str(duration) == "30.5 seconds"

    def test_pretty_duration_minutes(self):
        """Test PrettyDuration for minutes"""
        duration = formatters.PrettyDuration(180)
        assert str(duration) == "3:00 minutes"

    def test_pretty_duration_hours(self):
        """Test PrettyDuration for hours"""
        duration = formatters.PrettyDuration(3661)
        assert str(duration) == "1:01:01 hours"

    def test_pretty_duration_days(self):
        """Test PrettyDuration for days"""
        duration = formatters.PrettyDuration(90000)  # ~25 hours
        assert "days" in str(duration)

    def test_pretty_duration_milliseconds(self):
        """Test PrettyDuration for milliseconds"""
        duration = formatters.PrettyDuration(0.1)
        assert "milliseconds" in str(duration)

    def test_pretty_duration_repr(self):
        """Test PrettyDuration repr"""
        duration = formatters.PrettyDuration(60.0)
        assert repr(duration) == str(duration)

    def test_pretty_duration_to_dhms(self):
        """Test PrettyDuration.to_dhms static method"""
        result = formatters.PrettyDuration.to_dhms(3661.5)
        assert result.dd == 0
        assert result.hh == 1
        assert result.mm == 1
        assert result.ss == 1
        assert result.ms == 500.0
        assert result.pos is True

    def test_pretty_duration_to_dhms_negative(self):
        """Test PrettyDuration.to_dhms with negative value"""
        result = formatters.PrettyDuration.to_dhms(-3661.5)
        assert result.pos is False
        assert result.hh == 1
        assert result.mm == 1
        assert result.ss == 1

    def test_pretty_duration_time_string(self):
        """Test PrettyDuration.time_string static method"""
        result = formatters.PrettyDuration.time_string(3661.5)
        assert "1:01:01:500 hours" in result

    def test_pretty_duration_arithmetic(self):
        """Test PrettyDuration arithmetic operations"""
        d1 = formatters.PrettyDuration(60.0)
        d2 = formatters.PrettyDuration(30.0)

        # Addition
        result = d1 + d2
        assert result == 90.0
        assert isinstance(result, formatters.PrettyDuration)

        # Subtraction
        result = d1 - d2
        assert result == 30.0
        assert isinstance(result, formatters.PrettyDuration)

        # Multiplication
        result = d1 * 2
        assert result == 120.0
        assert isinstance(result, formatters.PrettyDuration)

        # Division
        result = d1 / 2
        assert result == 30.0
        assert isinstance(result, formatters.PrettyDuration)

    def test_pretty_duration_with_float(self):
        """Test PrettyDuration arithmetic with float"""
        duration = formatters.PrettyDuration(60.0)
        result = duration + 30.0
        assert result == 90.0
        assert isinstance(result, formatters.PrettyDuration)

    def test_pretty_duration_infinity(self):
        """Test PrettyDuration with infinity"""
        duration = formatters.PrettyDuration(np.inf)
        assert str(duration) == "inf"


class TestPrettySpace:
    """Test the PrettySpace class."""

    def test_pretty_space_initialization(self):
        """Test PrettySpace initialization"""
        space = formatters.PrettySpace(100.0)
        assert space == 100.0
        assert space.base_unit == "cm"

    def test_pretty_space_centimeters(self):
        """Test PrettySpace for centimeters"""
        space = formatters.PrettySpace(50.0)
        assert str(space) == "50 cm"

    def test_pretty_space_meters(self):
        """Test PrettySpace for meters"""
        space = formatters.PrettySpace(150.0)
        assert "1.5 m" in str(space)

    def test_pretty_space_kilometers(self):
        """Test PrettySpace for kilometers"""
        space = formatters.PrettySpace(100000.0)
        assert "1000 m" in str(space)

    def test_pretty_space_repr(self):
        """Test PrettySpace repr"""
        space = formatters.PrettySpace(100.0)
        assert repr(space) == str(space)

    def test_pretty_space_decompose_cm(self):
        """Test PrettySpace.decompose_cm static method"""
        result = formatters.PrettySpace.decompose_cm(150.0)
        assert result.m == 1
        assert result.cm == 50.0

    def test_pretty_space_decompose_cm2(self):
        """Test PrettySpace.decompose_cm2 static method"""
        result = formatters.PrettySpace.decompose_cm2(1000000.0)  # 10 km
        assert result.km == 10
        assert result.m == 0

    def test_pretty_space_space_string(self):
        """Test PrettySpace.space_string static method"""
        result = formatters.PrettySpace.space_string(150.0)
        assert "1.5 m" in result

    def test_pretty_space_arithmetic(self):
        """Test PrettySpace arithmetic operations"""
        s1 = formatters.PrettySpace(100.0)
        s2 = formatters.PrettySpace(50.0)

        # Addition
        result = s1 + s2
        assert result == 150.0
        assert isinstance(result, formatters.PrettySpace)

        # Subtraction
        result = s1 - s2
        assert result == 50.0
        assert isinstance(result, formatters.PrettySpace)

        # Multiplication
        result = s1 * 2
        assert result == 200.0
        assert isinstance(result, formatters.PrettySpace)

        # Division
        result = s1 / 2
        assert result == 50.0
        assert isinstance(result, formatters.PrettySpace)

    def test_pretty_space_with_float(self):
        """Test PrettySpace arithmetic with float"""
        space = formatters.PrettySpace(100.0)
        result = space + 50.0
        assert result == 150.0
        assert isinstance(result, formatters.PrettySpace)

    def test_pretty_space_zero(self):
        """Test PrettySpace with zero"""
        space = formatters.PrettySpace(0.0)
        assert str(space) == "0 um"

    def test_pretty_space_negative(self):
        """Test PrettySpace with negative value"""
        space = formatters.PrettySpace(-50.0)
        assert str(space) == "-50 cm"


class TestFormatterEdgeCases:
    """Test edge cases for formatters."""

    def test_pretty_bytes_edge_cases(self):
        """Test PrettyBytes edge cases"""
        # Test boundary values
        assert str(formatters.PrettyBytes(1023)) == "1023 bytes"
        assert str(formatters.PrettyBytes(1024)) == "1.000 kilobytes"
        assert str(formatters.PrettyBytes(1024**2 - 1)) == "1023.999 kilobytes"
        assert str(formatters.PrettyBytes(1024**2)) == "1.000 megabytes"

    def test_pretty_duration_edge_cases(self):
        """Test PrettyDuration edge cases"""
        # Test very small values
        assert "milliseconds" in str(formatters.PrettyDuration(0.001))

        # Test very large values
        large_duration = formatters.PrettyDuration(1000000)
        assert "days" in str(large_duration)

        # Test negative values
        neg_duration = formatters.PrettyDuration(-60)
        assert str(neg_duration) == "-1:00 minutes"

    def test_pretty_space_edge_cases(self):
        """Test PrettySpace edge cases"""
        # Test very small values
        assert str(formatters.PrettySpace(0.1)) == "100 mm"

        # Test very large values
        large_space = formatters.PrettySpace(1000000)
        assert "km" in str(large_space)

        # Test negative values
        neg_space = formatters.PrettySpace(-50)
        assert str(neg_space) == "-50 cm"

    def test_formatter_type_inheritance(self):
        """Test that formatters inherit from appropriate base types"""
        # Test that PrettyDuration inherits from float
        duration = formatters.PrettyDuration(60.0)
        assert isinstance(duration, float)

        # Test that PrettySpace inherits from float
        space = formatters.PrettySpace(100.0)
        assert isinstance(space, float)

        # Test that PrettyBytes inherits from int
        bytes_formatter = formatters.PrettyBytes(1024)
        assert isinstance(bytes_formatter, int)

        # Test that PrettyInt inherits from int
        int_formatter = formatters.PrettyInt(1000)
        assert isinstance(int_formatter, int)
