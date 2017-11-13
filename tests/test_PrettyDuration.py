from nelpy.utils import PrettyDuration

class TestPrettyDuration:

    def test_1(self):
        t = PrettyDuration(180)
        assert str(t) == '3:00 minutes'

    def test_2(self):
        t = PrettyDuration(179.999999)
        assert str(t) == '3:00 minutes'

    def test_3(self):
        t = PrettyDuration(5.99999)
        assert str(t) == '6 seconds'