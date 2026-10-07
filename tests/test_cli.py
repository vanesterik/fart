from pathlib import Path

import pytest

from fartt import cli
from tests.fakes import HOUR_MS, FakeExchange, candle


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeExchange:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)
    monkeypatch.setattr(cli, "CcxtExchange", lambda exchange_id: exchange)
    monkeypatch.setattr(cli, "DEFAULT_HISTORY_START_MS", 0)
    monkeypatch.chdir(tmp_path)  # main() writes logs/cli.log relative to the cwd
    return exchange


def test_download_is_a_subcommand(fake_exchange: FakeExchange, tmp_path: Path) -> None:
    cli.main(
        [
            "download",
            "--assets-dir",
            str(tmp_path),
            "--market",
            "BTC/EUR",
            "--interval",
            "1h",
        ]
    )

    lines = (tmp_path / "BTC-EUR-1h.csv").read_text().splitlines()
    assert lines[0] == "Timestamp,Open,High,Low,Close,Volume"
    assert len(lines) == 4


def test_download_rejects_unknown_market_with_hint(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "download",
                "--assets-dir",
                str(tmp_path),
                "--market",
                "BTC-EUR",
                "--interval",
                "1h",
            ]
        )

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "BTC-EUR" in error
    assert "BTC/EUR" in error
    assert not (tmp_path / "BTC-EUR-1h.csv").exists()


def test_no_command_prints_usage_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])

    assert exit_info.value.code == 2
    assert "download" in capsys.readouterr().err
