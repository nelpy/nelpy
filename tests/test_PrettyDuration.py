from nelpy.utils import PrettyDuration

class TestPrettyDuration:

    def test_1(self):
        t = PrettyDuration(180)
        assert t == '3:01 minutes'

    def test_2(self):
        t = PrettyDuration(179.9999999999999)
        assert t == '3:00 minutes'

    def test_3(self):
        t = PrettyDuration(5.9999)
        assert t == '6 seconds'