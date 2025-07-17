from nelpy.utils import PrettyDuration


class TestPrettyDuration:
    def test_1(self):
        t = PrettyDuration(180)
        assert str(t) == "3:00 minutes"

    def test_2(self):
        t = PrettyDuration(179.999999)
        assert str(t) == "3:00 minutes"

    def test_3(self):
        t = PrettyDuration(5.99999)
        assert str(t) == "6 seconds"

    def test_4(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.3)) == "1:02:300 minutes"

    def test_5(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.03)) == "1:02:030 minutes"

    def test_6(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(4393.5)) == "1:13:13:500 hours"

    def test_7(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(4393)) == "1:13:13 hours"

    def test_8(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.5)) == "3.5 seconds"

    def test_9(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.05)) == "3.05 seconds"

    def test_10(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.0)) == "3 seconds"

    def test_11(self):
        assert str(PrettyDuration(2586.3)) == "43:06:300 minutes"

    def test_12(self):
        assert str(PrettyDuration(2580)) == "43:00 minutes"

    def test_13(self):
        assert str(PrettyDuration(18.4)) == "18.4 seconds"

    def test_14(self):
        assert str(PrettyDuration(0.340)) == "340.0 milliseconds"

    def test_15(self):
        assert str(PrettyDuration(0.340)) == "340.0 milliseconds"

    def test_16(self):
        assert str(PrettyDuration(0.027)) == "27.0 milliseconds"
