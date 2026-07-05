"""
Regression tests for the gpiod polling-thread stuck-HIGH edge-recovery fix.

These tests are designed to FAIL on dev and PASS on fix/gpiod-polling-edge-recovery.

The race condition being tested:
  1. A CRC error arrives — polling thread detects rising edge, sets last_state=HIGH,
     fires the IRQ callback.
  2. _handle_interrupt clears the IRQ register — DIO1 goes LOW.
  3. A new packet arrives before the polling thread's next tick — DIO1 goes HIGH again.
  4. Polling thread ticks: current=HIGH, last_state=HIGH — no edge detected.
  5. Radio is permanently deaf until restarted.

The fix: when the polling thread sees current=HIGH with last_state=HIGH, it resets
last_state=False. On the next tick it sees current=HIGH, last_state=False — a normal
rising edge — and fires the callback correctly.
"""

import threading
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_polling_manager():
    """
    Instantiate GPIOPinManager with just enough internal state for
    _monitor_polling to run. Bypasses __init__ to avoid needing real hardware.
    """
    from openhop_core.hardware.gpio_manager import GPIOPinManager

    gm = object.__new__(GPIOPinManager)
    gm._pins = {}
    gm._input_callbacks = {}
    return gm


def _run_polling(gm, pin, reads, interval=0.001):
    """
    Run _monitor_polling with a controlled read sequence.

    The stop_event is set automatically after the read list is exhausted.
    Returns the list of callback invocations (one entry per call).
    """
    stop_event = threading.Event()
    read_idx = [0]

    def read_pin():
        i = read_idx[0]
        read_idx[0] += 1
        if i >= len(reads):
            stop_event.set()
            return False
        return reads[i]

    mock_pin = MagicMock()
    mock_pin.read.side_effect = read_pin
    gm._pins[pin] = mock_pin

    callbacks = []
    gm._input_callbacks[pin] = lambda: callbacks.append(1)

    thread = threading.Thread(
        target=gm._monitor_polling,
        args=(pin, stop_event, interval),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "Polling thread did not terminate within 2s"
    return callbacks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPollingEdgeRecovery:
    """Verify _monitor_polling recovers from the stuck-HIGH state."""

    def test_stuck_high_fires_recovery_callback(self):
        """
        Pin sequence: LOW → HIGH (edge 1) → HIGH (stuck) → HIGH → (stop)

        Tick 1: LOW             → no edge
        Tick 2: HIGH/last=LOW   → rising edge, callback fires, last_state=HIGH
        Tick 3: HIGH/last=HIGH  → dev: nothing | fix: last_state=False, continue
        Tick 4: HIGH/last=LOW   → fix only: rising edge, callback fires
        Tick 5: (stop)

        Expected callbacks — dev: 1,  fix: 2
        """
        gm = _make_polling_manager()
        callbacks = _run_polling(gm, pin=16, reads=[False, True, True, True])

        assert len(callbacks) == 2, (
            f"Expected 2 callbacks (initial edge + stuck-HIGH recovery), got {len(callbacks)}.\n"
            "If this is 1, the stuck-HIGH race is NOT fixed on this branch."
        )

    def test_normal_rising_edge_fires_exactly_once(self):
        """
        A clean LOW → HIGH → LOW cycle must still produce exactly one callback.
        Ensures the fix does not introduce spurious duplicates in normal operation.
        """
        gm = _make_polling_manager()
        callbacks = _run_polling(gm, pin=16, reads=[False, True, False])

        assert len(callbacks) == 1, (
            f"Expected exactly 1 callback for a clean rising edge, got {len(callbacks)}.\n"
            "The fix must not cause duplicate callbacks in normal operation."
        )

    def test_sustained_low_fires_no_callbacks(self):
        """Pin stays LOW throughout — no callbacks should ever fire."""
        gm = _make_polling_manager()
        callbacks = _run_polling(gm, pin=16, reads=[False, False, False, False])

        assert len(callbacks) == 0, (
            f"Expected 0 callbacks on a continuously LOW pin, got {len(callbacks)}."
        )

    def test_multiple_clean_edges_each_fire_once(self):
        """
        Two separate clean rising edges (with LOW between them) each fire exactly once.
        Verifies last_state resets correctly between edges under normal operation.
        """
        gm = _make_polling_manager()
        # LOW → HIGH (edge 1) → LOW → HIGH (edge 2) → (stop)
        callbacks = _run_polling(gm, pin=16, reads=[False, True, False, True])

        assert len(callbacks) == 2, (
            f"Expected 2 callbacks for 2 clean rising edges, got {len(callbacks)}."
        )

    def test_recovery_does_not_fire_on_every_tick(self):
        """
        In the stuck-HIGH state, the recovery mechanism fires every 2 ticks
        (reset tick + edge tick), not on every single tick.

        Sequence: LOW → HIGH (edge) → HIGH (reset) → HIGH (recovery edge) → (stop)
        Expected: exactly 2 callbacks in 4 reads.

        If it fired on every tick it would produce 3 callbacks in 4 reads.
        """
        gm = _make_polling_manager()
        # Exactly 3 consecutive HIGHs after the LOW — fits precisely in 2-tick recovery cycle
        callbacks = _run_polling(gm, pin=16, reads=[False, True, True, True])

        assert len(callbacks) == 2, (
            f"Expected exactly 2 callbacks in a 3-HIGH sequence, got {len(callbacks)}. "
            "Recovery must fire every 2 ticks (reset + edge), not every tick."
        )
