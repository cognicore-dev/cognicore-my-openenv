from pathlib import Path
import re


def test_healthbot_schedule_is_weekly_wednesday():
    workflow = Path(".github/workflows/healthbot.yml").read_text(encoding="utf-8")
    assert re.search(r"cron:\s*['\"]0 0 \* \* 3['\"]", workflow)
