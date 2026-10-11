"""Opt-in real-server regression: TQSERVE_TEST_URL must name a persistent no-drafter server."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


@unittest.skipUnless(os.environ.get("TQSERVE_TEST_URL"), "requires a real no-drafter server")
class RequestCarryover(unittest.TestCase):
    def test_three_orders_on_one_server(self):
        source = Path(__file__).resolve().parents[1] / "tools/capability_probe.py"
        spec = importlib.util.spec_from_file_location("carryover_probe", source)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        args = SimpleNamespace(full=True, long_tokens=3000, long_chars=6000, concurrency=4)
        orders = [("tools", "sampling", "determinism", "long"),
                  ("sampling", "tools", "long", "determinism"),
                  ("long", "determinism", "sampling", "tools")]
        for run, order in enumerate(orders):
            for name in order:
                with self.subTest(run=run, check=name):
                    ok, detail = getattr(probe, "check_" + name)(os.environ["TQSERVE_TEST_URL"], args)
                    self.assertTrue(ok, detail)


if __name__ == "__main__":
    unittest.main()
