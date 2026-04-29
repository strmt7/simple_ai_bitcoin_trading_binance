from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path

from simple_ai_bitcoin_trading_binance import cli
from simple_ai_bitcoin_trading_binance.api import Candle
from simple_ai_bitcoin_trading_binance.config import save_runtime, save_strategy
from simple_ai_bitcoin_trading_binance.external_signals import ExternalSignalComponent, ExternalSignalReport
from simple_ai_bitcoin_trading_binance.market_store import MarketDataStore
from simple_ai_bitcoin_trading_binance.data_downloader import MarketDataSyncResult
from simple_ai_bitcoin_trading_binance.types import RuntimeConfig, StrategyConfig


def _candle(index: int, close: float = 100.0) -> Candle:
    open_time = index * 60_000
    return Candle(open_time, close, close + 1, close - 1, close, 1.0, open_time + 59_000)


def _sync_args(**overrides):
    defaults = {
        "db": "data/market_data.sqlite",
        "symbol": None,
        "interval": None,
        "market": None,
        "rows": 5,
        "batch_size": 5,
        "include_futures_metrics": True,
        "loop": False,
        "iterations": 1,
        "sleep": 0,
        "background": False,
        "pid_file": "data/sync.pid",
        "log_file": "data/sync.log",
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _report(status: str = "ok", *, adjustment: float = 0.0, fresh: int = 2, risk: float = 1.0) -> ExternalSignalReport:
    return ExternalSignalReport(
        status=status,
        score_adjustment=adjustment,
        raw_score=adjustment,
        risk_multiplier=risk,
        provider_count=2,
        fresh_count=fresh,
        stale_count=0,
        known_at_ms=123,
        cache_path="cache.json",
        warnings=[],
        components=[
            ExternalSignalComponent("x", "ok", adjustment, 1.0, adjustment, "detail", 123),
            ExternalSignalComponent("y", "ok", 0.0, 1.0, 0.0, "detail", 123),
        ],
    )


def test_command_data_sync_foreground_json_loop_and_failure(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(market_type="spot"))
    builds: list[str] = []

    def fake_build(runtime):
        builds.append(runtime.market_type)
        return object()

    calls: list[object] = []

    def fake_sync(client, config, *, futures_client=None):
        calls.append((client, config, futures_client))
        return MarketDataSyncResult("ok", str(tmp_path / "m.sqlite"), "BTCUSDC", "15m", "spot", 1, 2, 60_000, 2, [], {})

    monkeypatch.setattr(cli, "_build_client", fake_build)
    monkeypatch.setattr(cli, "sync_market_data", fake_sync)
    assert cli.command_data_sync(_sync_args(db=str(tmp_path / "m.sqlite"), loop=True, iterations=2, json=True)) == 0
    out = capsys.readouterr().out
    assert out.count('"status": "ok"') == 2
    assert builds == ["spot", "futures"]
    assert len(calls) == 2

    def fail_sync(*_args, **_kwargs):
        raise ValueError("bad interval")

    monkeypatch.setattr(cli, "sync_market_data", fail_sync)
    assert cli.command_data_sync(_sync_args(db=str(tmp_path / "m.sqlite"))) == 2
    assert "Market data sync failed" in capsys.readouterr().err

    monkeypatch.setattr(
        cli,
        "sync_market_data",
        lambda *_args, **_kwargs: MarketDataSyncResult(
            "fail", str(tmp_path / "m.sqlite"), "BTCUSDC", "15m", "futures", 0, 0, None, 0, ["empty"], {}
        ),
    )
    assert cli.command_data_sync(_sync_args(db=str(tmp_path / "m.sqlite"), market="futures", include_futures_metrics=True)) == 2
    assert "warning: empty" in capsys.readouterr().out


def test_command_data_sync_background_builds_detached_process(tmp_path, monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class _Process:
        pid = 4321

    def fake_popen(command, *, stdout, stderr, start_new_session):
        captured["command"] = command
        captured["stderr"] = stderr
        captured["start_new_session"] = start_new_session
        stdout.write(b"")
        return _Process()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    args = _sync_args(
        db=str(tmp_path / "m.sqlite"),
        symbol="BTCUSDC",
        interval="1m",
        market="spot",
        include_futures_metrics=False,
        background=True,
        pid_file=str(tmp_path / "sync.pid"),
        log_file=str(tmp_path / "sync.log"),
        iterations=0,
    )
    assert cli.command_data_sync(args) == 0
    assert (tmp_path / "sync.pid").read_text(encoding="utf-8") == "4321\n"
    command = captured["command"]
    assert "--no-include-futures-metrics" in command
    assert "--interval" in command
    assert "started market data downloader" in capsys.readouterr().out

    args = _sync_args(
        db=str(tmp_path / "m2.sqlite"),
        background=True,
        pid_file=str(tmp_path / "sync2.pid"),
        log_file=str(tmp_path / "sync2.log"),
        include_futures_metrics=True,
    )
    assert cli.command_data_sync(args) == 0
    assert "--symbol" not in captured["command"]
    assert "--no-include-futures-metrics" not in captured["command"]


def test_data_sync_compatibility_wrappers_delegate_to_structured_module(tmp_path, monkeypatch) -> None:
    runtime = RuntimeConfig(market_type="spot", interval="15m")
    futures_runtime = cli._runtime_with_market(runtime, "futures")
    config = cli._data_sync_config_from_args(_sync_args(db=str(tmp_path / "m.sqlite"), interval="1m"), runtime)
    captured: dict[str, object] = {}

    class _Process:
        pid = 9876

    def fake_popen(command, *, stdout, stderr, start_new_session):
        captured["command"] = command
        stdout.write(b"")
        return _Process()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    assert futures_runtime.market_type == "futures"
    assert config.interval == "1m"
    assert config.db_path == str(tmp_path / "m.sqlite")
    assert cli._start_background_data_sync(
        _sync_args(
            db=str(tmp_path / "m.sqlite"),
            background=True,
            pid_file=str(tmp_path / "sync.pid"),
            log_file=str(tmp_path / "sync.log"),
        )
    ) == 0
    assert "data-sync" in captured["command"]


def test_training_data_loader_uses_db_and_download_prompt(tmp_path, monkeypatch) -> None:
    save_runtime(RuntimeConfig(market_type="spot", interval="15m"))
    db = tmp_path / "m.sqlite"
    with MarketDataStore(db) as store:
        store.upsert_candles("BTCUSDC", "spot", "15m", [_candle(i) for i in range(130)])

    args = argparse.Namespace(
        source="db",
        input=str(tmp_path / "missing.json"),
        db=str(db),
        interval="15m",
        market="spot",
        min_rows=120,
        download_missing=False,
    )
    candles, source = cli._load_training_candles(args, RuntimeConfig(market_type="spot", interval="15m"))
    assert source == "db"
    assert candles is not None and len(candles) == 130

    empty_db = tmp_path / "empty.sqlite"
    args.db = str(empty_db)
    assert cli._load_training_candles(args, RuntimeConfig(market_type="spot", interval="15m")) == (None, "missing")

    class _Tty:
        def isatty(self) -> bool:
            return True

    def fake_sync(sync_args):
        with MarketDataStore(sync_args.db) as store:
            store.upsert_candles("BTCUSDC", "spot", "15m", [_candle(i) for i in range(125)])
        return 0

    monkeypatch.setattr(cli.sys, "stdin", _Tty())
    monkeypatch.setattr(builtins, "input", lambda _prompt: "yes")
    monkeypatch.setattr(cli, "command_data_sync", fake_sync)
    args.db = str(tmp_path / "downloaded.sqlite")
    candles, source = cli._load_training_candles(args, RuntimeConfig(market_type="spot", interval="15m"))
    assert source == "db_downloaded"
    assert candles is not None and len(candles) == 125

    def no_write_sync(_sync_args):
        return 0

    args.download_missing = True
    args.db = str(tmp_path / "still-empty.sqlite")
    monkeypatch.setattr(cli, "command_data_sync", no_write_sync)
    assert cli._load_training_candles(args, RuntimeConfig(market_type="spot", interval="15m")) == (None, "missing")


def test_training_db_loader_warns_about_coverage_gaps(tmp_path, capsys) -> None:
    db = tmp_path / "gappy.sqlite"
    with MarketDataStore(db) as store:
        store.upsert_candles("BTCUSDC", "spot", "1m", [_candle(0), _candle(2)])

    candles = cli._load_training_candles_from_db(
        db,
        RuntimeConfig(market_type="spot", interval="1m"),
        interval="1m",
        market_type="spot",
        min_rows=2,
    )

    assert candles is not None and len(candles) == 2
    assert "missing intervals" in capsys.readouterr().err


def test_command_signals_and_external_score_helpers(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig())
    save_strategy(StrategyConfig())
    strategy_args = argparse.Namespace(
        profile="custom",
        leverage=None,
        risk=None,
        max_position=None,
        stop=None,
        take=None,
        cooldown=None,
        max_open=None,
        max_trades_per_day=None,
        signal_threshold=None,
        max_drawdown=None,
        taker_fee_bps=None,
        slippage_bps=None,
        label_threshold=None,
        model_lookback=None,
        training_epochs=None,
        confidence_beta=None,
        feature_window_short=None,
        feature_window_long=None,
        set_features=None,
        enable_feature=None,
        disable_feature=None,
        external_signals=True,
        external_signal_max_adjustment=0.5,
        external_signal_min_providers=9,
        external_signal_ttl=-1,
        external_signal_timeout=99.0,
    )
    assert cli.command_strategy(strategy_args) == 0
    saved = cli.load_strategy()
    assert saved.external_signals_enabled is True
    assert saved.external_signal_max_adjustment == 0.2
    assert saved.external_signal_min_providers == 4
    assert saved.external_signal_ttl_seconds == 0
    assert saved.external_signal_timeout_seconds == 30.0
    report = _report("ok", adjustment=-0.03, fresh=2, risk=0.5)
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: report)
    args = argparse.Namespace(
        model=str(tmp_path / "model.json"),
        cache=None,
        ttl=10,
        timeout=1.0,
        max_adjustment=0.1,
        min_providers=2,
        refresh=True,
        json=False,
    )
    assert cli.command_signals(args) == 0
    assert "External signal report" in capsys.readouterr().out
    args.json = True
    assert cli.command_signals(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: _report("fail"))
    assert cli.command_signals(args) == 2
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    assert cli.command_signals(args) == 2
    assert "External signal collection failed" in capsys.readouterr().err
    assert cli._external_signal_cache_path(Path("data/model.json")) == Path("data/signals/external_signals.json")
    adjusted, effective, applied = cli._apply_external_signal_to_score(0.6, StrategyConfig(), report)
    assert adjusted == 0.57
    assert applied == -0.03
    assert effective.risk_per_trade == 0.005


class _LiveClient:
    def __init__(self) -> None:
        self.orders: list[dict[str, object]] = []

    def get_symbol_constraints(self, _symbol):
        return None

    def get_klines(self, *_args, **_kwargs):
        return [_candle(i, 100.0 + i) for i in range(80)]

    def place_order(self, symbol, side, size, *, dry_run, leverage):
        self.orders.append({"symbol": symbol, "side": side, "size": size, "dry_run": dry_run, "leverage": leverage})
        return {"ok": True, "side": side, "size": size}


class _ScoreModel:
    def __init__(self, score: float) -> None:
        self.score = score
        self.feature_signature = "sig"

    def predict_proba(self, _features) -> float:
        return self.score


def test_live_external_signals_gate_positive_boost_and_continue_on_error(tmp_path, monkeypatch, capsys) -> None:
    save_runtime(RuntimeConfig(market_type="spot", dry_run=True))
    save_strategy(StrategyConfig(external_signals_enabled=True, external_signal_min_providers=2, confidence_beta=1.0))
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    client = _LiveClient()
    monkeypatch.setattr(cli, "_build_client", lambda _runtime: client)
    monkeypatch.setattr(cli, "_load_runtime_model", lambda _path, _cfg: _ScoreModel(0.57))
    monkeypatch.setattr(cli, "collect_external_signals", lambda **_kwargs: _report("warn", adjustment=0.05, fresh=1))
    args = argparse.Namespace(
        steps=1,
        sleep=0,
        paper=True,
        live=False,
        model=str(model_path),
        leverage=None,
        retrain_interval=0,
        retrain_window=1,
        retrain_min_rows=1,
        external_signals=None,
    )
    assert cli.command_live(args) == 0
    assert client.orders == []
    assert "external signals" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_load_runtime_model", lambda _path, _cfg: _ScoreModel(0.8))

    def fail_signals(**_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(cli, "collect_external_signals", fail_signals)
    args.external_signals = True
    assert cli.command_live(args) == 0
    assert client.orders and client.orders[-1]["side"] == "BUY"
    assert "external signals unavailable" in capsys.readouterr().err
