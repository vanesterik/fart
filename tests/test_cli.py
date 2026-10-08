from pathlib import Path
from typing import Any

import pytest

from fartt import cli
from fartt.forecast import RepeatLastReturn, Settings
from tests.fakes import HOUR_MS, FakeExchange, candle


@pytest.fixture
def fake_exchange(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeExchange:
    exchange = FakeExchange([candle(h) for h in range(3)], now_ms=3 * HOUR_MS)

    def make_exchange(exchange_id: str) -> FakeExchange:
        if exchange_id != "bitvavo":
            raise ValueError(f"Exchange '{exchange_id}' is not supported by ccxt")
        return exchange

    monkeypatch.setattr(cli, "CcxtExchange", make_exchange)
    monkeypatch.setattr(cli, "DEFAULT_HISTORY_START_MS", 0)
    monkeypatch.chdir(tmp_path)  # main() writes logs/cli.log relative to the cwd
    return exchange


class FakeServer:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    built: dict[str, Any] = {}

    def fake_build_server(
        cache: Any, market: str, interval: str, forecaster: Any, settings: Any
    ) -> FakeServer:
        built.update(
            cache=cache,
            market=market,
            interval=interval,
            forecaster=forecaster,
            settings=settings,
            server=FakeServer(),
        )
        return built["server"]

    monkeypatch.setattr(cli, "build_server", fake_build_server)
    return built


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


@pytest.mark.parametrize("interval", ["3h", "1x"])
def test_download_rejects_an_interval_the_exchange_does_not_offer(
    fake_exchange: FakeExchange,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    interval: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--interval", interval])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert f"interval '{interval}'" in error
    assert "1m, 1h" in error


def test_download_rejects_an_unknown_exchange(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--exchange", "bitvavoo"])

    assert exit_info.value.code == 2
    assert "bitvavoo" in capsys.readouterr().err


def test_download_exits_cleanly_when_the_exchange_is_unreachable(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_exchange.reachable = False

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--interval", "1h"])

    assert exit_info.value.code == 1
    error = capsys.readouterr().err
    assert "unavailable" in error
    assert "Traceback" not in error


def test_serve_runs_the_server_for_the_configured_market(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(
        [
            "serve",
            "--assets-dir",
            str(tmp_path),
            "--market",
            "BTC/EUR",
            "--interval",
            "1h",
        ]
    )

    assert served["market"] == "BTC/EUR"
    assert served["interval"] == "1h"
    assert served["cache"].filepath == tmp_path / "BTC-EUR-1h.csv"
    assert served["forecaster"] == RepeatLastReturn()
    assert served["settings"] == Settings()
    assert "fee 0.25% per leg" in capsys.readouterr().err
    assert served["server"].ran


def test_serve_starts_when_the_exchange_is_unreachable(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    fake_exchange.reachable = False

    cli.main(["serve", "--assets-dir", str(tmp_path), "--interval", "1h"])

    assert served["server"].ran


def test_serve_rejects_an_unknown_market(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "serve",
                "--assets-dir",
                str(tmp_path),
                "--market",
                "BTC-EUR",
                "--interval",
                "1h",
            ]
        )

    assert exit_info.value.code == 2
    assert "BTC/EUR" in capsys.readouterr().err
    assert not served


def test_download_rejects_an_offered_interval_the_cache_cannot_handle(
    fake_exchange: FakeExchange, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Some exchanges offer intervals such as binance's lowercase `1w`.
    fake_exchange.intervals = ["1h", "1w"]

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["download", "--assets-dir", str(tmp_path), "--interval", "1w"])

    assert exit_info.value.code == 2
    error = capsys.readouterr().err
    assert "interval '1w'" in error
    assert "Traceback" not in error


def test_serve_passes_the_analysis_options(
    fake_exchange: FakeExchange, served: dict[str, Any], tmp_path: Path
) -> None:
    cli.main(
        [
            "serve",
            "--assets-dir",
            str(tmp_path),
            "--interval",
            "1h",
            "--fee",
            "0.001",
            "--slippage",
            "0",
            "--threshold",
            "0.01",
            "--hit-rate-window",
            "50",
        ]
    )

    assert served["settings"] == Settings(
        fee=0.001, slippage=0.0, threshold=0.01, hit_rate_window=50
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--fee", "-0.001"),
        ("--fee", "nan"),
        ("--slippage", "inf"),
        ("--threshold", "-0.01"),
        ("--hit-rate-window", "0"),
    ],
)
def test_serve_rejects_nonsense_analysis_options(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--assets-dir", str(tmp_path), option, value])

    assert exit_info.value.code == 2
    assert option in capsys.readouterr().err
    assert not served


def test_serve_warns_about_a_threshold_below_the_round_trip_cost(
    fake_exchange: FakeExchange,
    served: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(["serve", "--assets-dir", str(tmp_path), "--threshold", "0.001"])

    assert "below the round-trip cost" in capsys.readouterr().err
    assert served["server"].ran
