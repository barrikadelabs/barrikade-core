import asyncio
import time
from pathlib import Path

import broker_demo
import pytest


def test_missing_action_rejected_at_boundary():
    # a permission broker never guesses the verb — omitting action is a caller error
    with pytest.raises(TypeError):
        asyncio.run(broker_demo.request_credentials("github", reason="x"))


def test_audit_log_anchored_to_module_dir():
    # the audit trail must land next to the broker regardless of the launch cwd
    log_path = Path(broker_demo._handler.baseFilename)
    assert log_path.parent == Path(broker_demo.__file__).parent
    assert log_path.name == "audit.log"


def test_audit_timestamps_are_utc():
    # %(asctime)s renders via the formatter's converter; local time would skew
    # host vs container lines from the same run (seen live: one hour apart)
    assert broker_demo._formatter.converter is time.gmtime
