# tools/rc_gate.py verdict: a long-prompt decode collapse fails the arm even when the chats gain (67.1 -> 4.9 once passed).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from rc_gate import verdict

failures = 0
def check(ok, what):
    global failures
    if not ok: print("FAIL", what); failures += 1

check(not verdict(True, True, -6.0, 100 * (1 - 4.9 / 67.1), 3.0, 8.0), "long decode 67.1 -> 4.9 fails while chats gain 6 %")
check(verdict(True, True, -2.2, -1.7, 3.0, 8.0), "faster chats and long decode pass")
check(not verdict(True, True, 4.0, 0.0, 3.0, 8.0), "chat loss above the limit fails")
check(not verdict(False, True, 0.0, 0.0, 3.0, 8.0), "different text fails")
check(not verdict(True, False, 0.0, 0.0, 3.0, 8.0), "different long text fails")
print("rc_gate verdict:", "FAIL" if failures else "PASS")
sys.exit(1 if failures else 0)
