"""Minimal, reproducible Binance UM mean-reversion study.

The script deliberately keeps the accounting path small: one-minute public
klines are converted to monthly Parquet, five-minute bars are built locally,
and all strategies use the same delayed execution and ledger.
"""
from __future__ import annotations

import argparse, concurrent.futures, hashlib, io, json, math, os, re, sys, time, zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TMP = ROOT / "tmp"
SYMBOLS = "BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT TRXUSDT LINKUSDT SUIUSDT AVAXUSDT LTCUSDT BCHUSDT DOTUSDT HBARUSDT XLMUSDT FILUSDT UNIUSDT NEARUSDT AAVEUSDT".split()
MONTHS = [f"2026-{m:02d}" for m in range(3, 9)]
START = pd.Timestamp("2026-03-01", tz="UTC")
END = pd.Timestamp("2026-09-01", tz="UTC")
KEEP = ["open_time_utc_ms", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume"]
URL = "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{m}.zip"


def mkdirs() -> None:
    for p in (DATA, RESULTS, TMP):
        p.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def request_bytes(url: str, timeout: int = 60, attempts: int = 4) -> tuple[bytes, int]:
    last = ""
    for n in range(attempts):
        try:
            r = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "crypto-mean-reversion-research/1.0"})
            status = r.status_code
            if status == 404:
                raise FileNotFoundError(url)
            if status == 429 or status >= 500:
                last = f"HTTP {status}"; time.sleep(2 ** n); continue
            r.raise_for_status()
            chunks = []
            for c in r.iter_content(1 << 20):
                if c: chunks.append(c)
            b = b"".join(chunks)
            if b[:1] == b"<" or b"<html" in b[:512].lower():
                last = "HTML response"; time.sleep(2 ** n); continue
            return b, status
        except FileNotFoundError:
            raise
        except Exception as e:
            last = repr(e); time.sleep(2 ** n)
    raise RuntimeError(last or "download failed")


def expected_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(month + "-01", tz="UTC")
    b = a + pd.offsets.MonthBegin(1)
    return a, b


def convert_zip(zip_path: Path, out_path: Path, symbol: str, month: str) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV, got {names}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, usecols=lambda c: c in {"open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume", "taker_buy_quote_volume", "volume", "taker_buy_volume"})
    # Old Binance archives sometimes contain no header. Detect and re-read.
    if "open_time" not in df.columns:
        with zipfile.ZipFile(zip_path) as z, z.open(names[0]) as f:
            cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
            df = pd.read_csv(f, header=None, names=cols, usecols=cols)
    rename = {"open_time": "open_time_utc_ms", "taker_buy_quote_volume": "taker_buy_quote_volume"}
    if "quote_volume" not in df and "volume" in df:
        raise ValueError("archive has no quote_volume")
    df = df.rename(columns=rename)[KEEP]
    for c in KEEP[1:]: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["open_time_utc_ms"] = pd.to_numeric(df["open_time_utc_ms"], errors="coerce").astype("int64")
    a, b = expected_bounds(month)
    ts = pd.to_datetime(df.open_time_utc_ms, unit="ms", utc=True)
    if ts.min() != a or ts.max() >= b or (ts.dt.second != 0).any() or (ts.dt.microsecond != 0).any():
        raise ValueError(f"coverage/alignment failure {symbol} {month}: {ts.min()}..{ts.max()}")
    if df.open_time_utc_ms.duplicated().any() or not df.open_time_utc_ms.is_monotonic_increasing:
        raise ValueError("duplicate or unsorted minute rows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    df.to_parquet(tmp, index=False, compression="zstd")
    check = pd.read_parquet(tmp, columns=KEEP)
    if len(check) != len(df) or check.open_time_utc_ms.iloc[0] != df.open_time_utc_ms.iloc[0]:
        raise ValueError("Parquet read-back mismatch")
    tmp.replace(out_path)
    return {"rows": int(len(df)), "start": str(ts.min()), "end": str(ts.max()), "parquet_bytes": out_path.stat().st_size}


def download_one(symbol: str, month: str) -> dict:
    out = DATA / "klines" / f"symbol={symbol}" / f"month={month}.parquet"
    url = URL.format(s=symbol, m=month)
    rec = {"symbol": symbol, "month": month, "url": url, "target": str(out.relative_to(ROOT)), "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "conversion_version": "1.0"}
    if out.exists():
        try:
            q = pd.read_parquet(out, columns=["open_time_utc_ms"])
            a, b = expected_bounds(month)
            if len(q) and pd.to_datetime(q.open_time_utc_ms.iloc[0], unit="ms", utc=True) == a and pd.to_datetime(q.open_time_utc_ms.iloc[-1], unit="ms", utc=True) < b:
                rec.update({"status": "skipped_existing", "rows": int(len(q)), "parquet_bytes": out.stat().st_size})
                return rec
        except Exception:
            pass
    try:
        b, status = request_bytes(url)
        official, _ = request_bytes(url + ".CHECKSUM", timeout=30, attempts=3)
        official_hash = official.decode("ascii", "replace").split()[0].lower()
        part = TMP / f"{symbol}-{month}.zip.part"
        part.write_bytes(b)
        local = sha256(part)
        if local != official_hash:
            raise ValueError(f"checksum mismatch {local} != {official_hash}")
        zpath = part.with_suffix("")
        part.replace(zpath)
        rec.update({"http_status": status, "download_bytes": len(b), "official_sha256": official_hash, "local_sha256": local})
        rec.update(convert_zip(zpath, out, symbol, month))
        zpath.unlink(missing_ok=True)
        rec["status"] = "ok"
    except FileNotFoundError:
        rec.update({"status": "missing_404", "error": "HTTP 404"})
    except Exception as e:
        rec.update({"status": "error", "error": repr(e)})
    return rec


def download(symbols: list[str] = SYMBOLS, months: list[str] = MONTHS, workers: int = 4) -> None:
    mkdirs(); jobs = [(s, m) for s in symbols for m in months]
    t0 = time.time(); records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(lambda x: download_one(*x), jobs):
            print(r["symbol"], r["month"], r["status"], flush=True); records.append(r)
    manifest = RESULTS / "source_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (RESULTS / "download_run.json").write_text(json.dumps({"seconds": time.time()-t0, "requested": len(jobs), "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_exchange_info() -> dict:
    """Public metadata; failures are retained as evidence, never fabricated."""
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20)
        r.raise_for_status(); j = r.json()
        (RESULTS / "exchange_info.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
        rows = [{k: s.get(k) for k in ("symbol", "contractType", "quoteAsset", "marginAsset", "status", "onboardDate")} for s in j.get("symbols", []) if s.get("symbol") in SYMBOLS]
        (RESULTS / "universe_check.json").write_text(json.dumps({"requested": SYMBOLS, "found": rows, "missing": [s for s in SYMBOLS if s not in {x['symbol'] for x in rows}]}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "ok", "rows": rows}
    except Exception as e:
        out = {"status": "unavailable", "error": repr(e), "requested": SYMBOLS}
        (RESULTS / "universe_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def fetch_funding(symbol: str) -> dict:
    out = DATA / "funding" / f"{symbol}.parquet"; out.parent.mkdir(parents=True, exist_ok=True)
    rows = []; archive_errors = []
    # The official archive contains actual settlement times/rates and interval.
    # It does not contain markPrice; that missing field is kept as NaN below.
    for month in MONTHS:
        u = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
        try:
            b, _ = request_bytes(u, timeout=30, attempts=3)
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                name = next(n for n in z.namelist() if n.endswith(".csv"))
                with z.open(name) as f:
                    d = pd.read_csv(f)
            if {"calc_time", "last_funding_rate"} - set(d.columns): raise ValueError("unexpected funding columns")
            d["symbol"] = symbol; d["funding_time_utc_ms"] = d.calc_time.astype("int64")
            d["funding_rate"] = d.last_funding_rate.astype(float); d["mark_price"] = np.nan
            rows.append(d[["symbol", "funding_time_utc_ms", "funding_rate", "mark_price"]])
        except Exception as e:
            archive_errors.append({"month": month, "error": repr(e)})
    try:
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if df.empty: raise RuntimeError("all official funding archives unavailable")
        df = df.drop_duplicates("funding_time_utc_ms").sort_values("funding_time_utc_ms")
        df.to_parquet(out, index=False, compression="zstd")
        return {"symbol": symbol, "status": "archive_rates_mark_price_missing", "rows": len(df), "path": str(out.relative_to(ROOT)), "archive_errors": archive_errors}
    except Exception as e:
        return {"symbol": symbol, "status": "unavailable", "error": repr(e), "archive_errors": archive_errors}


def funding() -> None:
    mkdirs(); t0 = time.time(); results = [fetch_funding(s) for s in SYMBOLS]
    (RESULTS / "funding_status.json").write_text(json.dumps({"seconds": time.time()-t0, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def load_minute() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    opens1, opens5, close5, volume5 = {}, {}, {}, {}
    for s in SYMBOLS:
        frames = []
        for m in MONTHS:
            p = DATA / "klines" / f"symbol={s}" / f"month={m}.parquet"
            if p.exists(): frames.append(pd.read_parquet(p, columns=KEEP))
        if not frames: continue
        d = pd.concat(frames, ignore_index=True)
        ix = pd.to_datetime(d.open_time_utc_ms, unit="ms", utc=True)
        x = d.set_index(ix)
        opens1[s] = x.open.astype(float)
        opens5[s] = x.open.astype(float).resample("5min", label="left", closed="left").first()
        close5[s] = x.close.astype(float).resample("5min", label="left", closed="left").last()
        volume5[s] = x.quote_volume.astype(float).resample("5min", label="left", closed="left").sum(min_count=5)
    idx = pd.date_range(START, END, freq="5min", inclusive="left", tz="UTC")
    minute_idx = pd.date_range(START, END, freq="1min", inclusive="left", tz="UTC")
    return idx, pd.DataFrame(opens1).reindex(minute_idx), pd.DataFrame(opens5).reindex(idx), pd.DataFrame(close5).reindex(idx), pd.DataFrame(volume5).reindex(idx)


def quality() -> dict:
    mkdirs(); out = {"symbols": {}, "total_rows": 0, "expected_rows": 184*1440*20}
    for s in SYMBOLS:
        rows = 0; issues = []; miss = 0; zero = 0
        for m in MONTHS:
            p = DATA / "klines" / f"symbol={s}" / f"month={m}.parquet"
            if not p.exists(): issues.append(f"missing_partition:{m}"); continue
            d = pd.read_parquet(p); rows += len(d)
            ts = pd.to_datetime(d.open_time_utc_ms, unit="ms", utc=True)
            a,b = expected_bounds(m); expected = int((b-a).total_seconds()/60)
            miss += max(0, expected-len(d)); zero += int((d.quote_volume == 0).sum())
            checks = {"duplicate": d.open_time_utc_ms.duplicated().sum(), "non_monotonic": int(not d.open_time_utc_ms.is_monotonic_increasing), "minute_gaps": int((d.open_time_utc_ms.diff().dropna() != 60000).sum()), "nan_values": int(d[KEEP].isna().any(axis=1).sum()), "bad_price": int((d[["open","high","low","close"]] <= 0).any().any()), "bad_ohlc": int(((d.high < d[["open","close"]].max(axis=1)) | (d.low > d[["open","close"]].min(axis=1))).sum()), "negative_volume": int((d.quote_volume < 0).sum()), "taker_gt_quote": int((d.taker_buy_quote_volume > d.quote_volume + 1e-8).sum())}
            for k,v in checks.items():
                if v: issues.append(f"{m}:{k}={v}")
        out["symbols"][s] = {"rows": rows, "missing_minutes": miss, "zero_volume_minutes": zero, "issues": issues}; out["total_rows"] += rows
    (RESULTS / "quality.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def pca_matrix(ret: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return D, V and M used by the documented standardized-space mapping."""
    mu = np.nanmean(ret, axis=0); sd = np.nanstd(ret, axis=0, ddof=1); sd = np.where(sd > 1e-12, sd, 1.0)
    z = (ret - mu) / sd
    _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    v = vt[:k].T
    d = np.diag(sd); m = np.eye(ret.shape[1]) - d @ v @ v.T @ np.linalg.inv(d)
    return d, v, m


def ewma_z(series: pd.Series, center_hours: float = 4, scale_days: float = 7) -> pd.Series:
    center = series.ewm(span=int(center_hours*12), adjust=False, min_periods=1).mean().shift(1)
    dev = series - center
    scale = dev.rolling(int(scale_days*24*12), min_periods=288).std().shift(1)
    return dev / scale.replace(0, np.nan)


def metrics(eq: pd.Series, trades: pd.DataFrame, initial: float) -> dict:
    daily = eq.resample("1D").last().pct_change().dropna(); peak = eq.cummax(); dd = eq/peak-1
    gross = float(trades.price_pnl.sum()) if len(trades) else 0.0
    known_funding = trades.funding.dropna() if len(trades) else pd.Series(dtype=float)
    return {"start_equity": initial, "end_equity": float(eq.iloc[-1]), "interval_return": float(eq.iloc[-1]/initial-1), "daily_sharpe_sqrt365": float(daily.mean()/daily.std(ddof=1)*math.sqrt(365)) if len(daily)>1 and daily.std(ddof=1)>0 else None, "max_drawdown": float(dd.min()), "trades": int(len(trades)), "gross_price_pnl": gross, "fees": float(trades.fee.sum()) if len(trades) else 0.0, "slippage": float(trades.slippage.sum()) if len(trades) else 0.0, "funding": float(known_funding.sum()) if len(known_funding) else None, "turnover": float(trades.turnover.sum()) if len(trades) else 0.0, "break_even_one_way_bp": float(gross/(trades.turnover.sum())*10000) if len(trades) and trades.turnover.sum() else None}


def funding_cashflow(quantity: float, mark_price: float, rate: float) -> float:
    """Linear contract cash flow; positive quantity is a long."""
    return -float(quantity) * float(mark_price) * float(rate)


def total_cost(turnover: float, one_way_bp: float) -> float:
    return float(turnover) * float(one_way_bp) / 10000.0


def build_signals(close: pd.DataFrame, model: str, k: int = 3) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    logp = np.log(close); ret = logp.diff(); z = pd.DataFrame(index=close.index, columns=SYMBOLS, dtype=float); weights = {}
    if model == "B0":
        for s in SYMBOLS: z[s] = ewma_z(logp[s])
        for s in SYMBOLS: weights[s] = np.eye(len(SYMBOLS))[SYMBOLS.index(s)]
    elif model == "B1":
        for s in SYMBOLS:
            ref = (ret.sum(axis=1) - ret[s]) / max(1, len(SYMBOLS)-1)
            spread = (ret[s] - ref).cumsum(); z[s] = ewma_z(spread)
            w = np.full(len(SYMBOLS), 1/(len(SYMBOLS)-1)); w[SYMBOLS.index(s)] = -1; weights[s] = w
    else:
        # One daily frozen PCA estimate; the estimate uses only the preceding 28 days.
        level = np.zeros(len(SYMBOLS)); level_rows = []
        for day, g in ret.groupby(ret.index.floor("D")):
            end = pd.Timestamp(day); end = end.tz_localize("UTC") if end.tzinfo is None else end
            hist = ret.loc[:end - pd.Timedelta(minutes=5)].tail(28*24*12).dropna()
            if len(hist) < 28*24*12: continue
            _, _, m = pca_matrix(hist.to_numpy(), k); weights[str(day.date())] = m
            transformed = ret.loc[g.index].to_numpy() @ m.T
            for i in range(len(g)):
                level += np.nan_to_num(transformed[i], nan=0.0)
                level_rows.append((g.index[i], level.copy()))
        if not weights: return z, {}
        levels = pd.DataFrame({s: {t: v[j] for t, v in level_rows} for j, s in enumerate(SYMBOLS)}).sort_index()
        for s in SYMBOLS: z[s] = ewma_z(levels[s])
    return z, weights


def backtest(close: pd.DataFrame, open1: pd.DataFrame, z: pd.DataFrame, model: str, weights, entry_sigma=2.0, hold_hours=4.0, cost_bp=7.0, initial=100000.0) -> tuple[pd.Series,pd.DataFrame]:
    # A single account with at most three frozen positions; all legs are netted before costs.
    idx = close.index; equity = initial; cash = initial; active = []; rows = []; eq = []
    maxbars = int(hold_hours*12); cost = cost_bp/10000.0
    for n,t in enumerate(idx):
        px = close.loc[t]
        # mark and evaluate exits at the close; execution is the next 5-minute open proxy
        new_active=[]
        for p in active:
            age=n-p["n"]; value=float(np.nansum(p["qty"]*px.to_numpy()))
            rel=float((value-p["entry_value"])/p["gross"]) if p["gross"] else 0
            zz = abs(float(z.loc[t,p["signal"]])) if pd.notna(z.loc[t,p["signal"]]) else np.inf
            if age >= maxbars or zz <= .5 or rel <= -.03:
                ex_t = t + pd.Timedelta(minutes=6); ep = open1.reindex([ex_t]).iloc[0] if ex_t in open1.index else px
                proceeds=float(np.nansum(p["qty"]*ep.to_numpy())); turnover=float(np.nansum(np.abs(p["qty"]*ep.to_numpy())))+p["entry_turnover"]; fee=turnover*cost; slippage=0.0
                price_pnl=proceeds-p["entry_value"]; cash += price_pnl-fee-slippage
                rows.append({"model":model,"signal":p["signal"],"entry_time":p["entry_time"],"exit_time":ex_t,"gross":p["gross"],"turnover":turnover,"price_pnl":price_pnl,"fee":fee,"slippage":slippage,"funding":np.nan,"net_pnl":price_pnl-fee-slippage,"reason":"timeout" if age>=maxbars else ("stop" if rel<=-.03 else "mean")})
            else: new_active.append(p)
        active=new_active
        # choose the largest valid deviation, with frozen weights at entry
        if len(active)<3 and n+1 < len(idx):
            zz=z.loc[t].abs().dropna()
            if len(zz):
                s=str(zz.idxmax()); val=float(z.loc[t,s])
                if abs(val)>=entry_sigma and not any(p["signal"]==s for p in active):
                    ep_t=t+pd.Timedelta(minutes=6); ep=open1.reindex([ep_t]).iloc[0] if ep_t in open1.index else None
                    if ep is not None and ep.notna().all():
                        if model=="B2": w=weights.get(str(t.date()), np.eye(len(SYMBOLS))[SYMBOLS.index(s)])[SYMBOLS.index(s)]
                        elif model=="B1": w=weights[s]
                        else: w=np.eye(len(SYMBOLS))[SYMBOLS.index(s)]
                        w=-np.sign(val)*np.asarray(w,float); w=w/(np.abs(w).sum() or 1); gross=equity*.2; qty=gross*w/ep.to_numpy(); entry_value=float(np.nansum(qty*ep.to_numpy())); active.append({"signal":s,"n":n,"entry_time":ep_t,"qty":qty,"gross":gross,"entry_value":entry_value,"entry_turnover":float(np.nansum(np.abs(qty*ep.to_numpy())))})
        mark=float(np.nansum([p["qty"] @ px.to_numpy() - p["entry_value"] for p in active])); eq.append((t,cash+mark))
    if active:
        t=idx[-1]; px=close.loc[t];
        for p in active:
            proceeds=float(np.nansum(p["qty"]*px.to_numpy())); turnover=float(np.nansum(np.abs(p["qty"]*px.to_numpy())))+p["entry_turnover"]; fee=turnover*cost; slippage=0.0; price_pnl=proceeds-p["entry_value"]
            rows.append({"model":model,"signal":p["signal"],"entry_time":p["entry_time"],"exit_time":t,"gross":p["gross"],"turnover":turnover,"price_pnl":price_pnl,"fee":fee,"slippage":slippage,"funding":np.nan,"net_pnl":price_pnl-fee-slippage,"reason":"end_of_sample"})
            cash += price_pnl-fee-slippage
        eq[-1] = (idx[-1], cash)
    return pd.Series(dict(eq)).sort_index(), pd.DataFrame(rows)


def conditional(close: pd.DataFrame, z: pd.DataFrame, open1: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for s in SYMBOLS:
        for threshold, lo in [("1-2",1),("2-3",2),("3-4",3),("4+",4)]:
            mask=z[s].abs().between(lo, 2 if lo==1 else (3 if lo==2 else (4 if lo==3 else 99)), inclusive="left")
            for mins in [5,15,30,60,120,240,1440]:
                delay = pd.Timedelta(minutes=6); entry = open1[s].reindex(close.index + delay).set_axis(close.index)
                target = (close.index + delay + pd.Timedelta(minutes=mins)).floor("5min")
                future = close[s].reindex(target).set_axis(close.index)
                fwd=future/entry-1; x=fwd[mask & fwd.notna()]
                rows.append({"symbol":s,"band":threshold,"minutes":mins,"n":int(len(x)),"mean":float(x.mean()) if len(x) else None,"median":float(x.median()) if len(x) else None,"positive_rate":float((x>0).mean()) if len(x) else None,"q05":float(x.quantile(.05)) if len(x) else None})
    out=pd.DataFrame(rows); out.to_csv(RESULTS/"conditional_returns.csv",index=False); return out


def plot_results(cond: pd.DataFrame, summaries: pd.DataFrame, equity: dict[str,pd.Series]) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4)); x=cond[(cond.band=="2-3") & (cond.minutes.isin([5,15,30,60,120,240,1440]))].groupby("minutes").mean(numeric_only=True); 
    if "mean" in x: plt.plot(x.index,x["mean"]*100,marker="o")
    plt.xlabel("持仓分钟"); plt.ylabel("未来收益均值 %"); plt.tight_layout(); plt.savefig(RESULTS/"conditional_horizon.png",dpi=140); plt.close()
    plt.figure(figsize=(8,4));
    for s,g in summaries.groupby("model"): plt.plot(g.cost_bp,g.interval_return*100,marker="o",label=s)
    plt.xlabel("单边成本(bp)"); plt.ylabel("区间净收益 %"); plt.legend(); plt.tight_layout(); plt.savefig(RESULTS/"cost_sensitivity.png",dpi=140); plt.close()
    plt.figure(figsize=(9,4));
    for k,v in equity.items(): plt.plot(v.index,v/v.iloc[0],label=k)
    plt.legend(); plt.ylabel("归一化权益"); plt.tight_layout(); plt.savefig(RESULTS/"equity_drawdown.png",dpi=140); plt.close()


def analyze() -> None:
    mkdirs(); q=quality(); idx,open1,open5,close,vol=load_minute(); available=[s for s in SYMBOLS if s in close]
    if len(available)<20: print(f"warning: only {len(available)}/20 symbols available")
    close=close[available]; open1=open1[available]; open5=open5[available]; zall={}; summary=[]; equity={}; all_cond=[]
    # The split is frozen before evaluation; all three models use the same account path.
    for model,k in [("B0",0),("B1",0),("B2",3)]:
        z,w=build_signals(close,model,k); zall[model]=z
        if model=="B1": pass
        for bp in [0,2,5,7,10]:
            eq,tr=backtest(close,open1,z,model,w,2,4,bp)
            if len(tr): tr.to_csv(RESULTS/f"trades_{model}_{bp}bp.csv",index=False)
            mm=metrics(eq,tr,100000); mm.update({"model":model,"cost_bp":bp}); summary.append(mm)
            if bp==7: equity[model]=eq
        if model=="B1": all_cond.append(conditional(close,z,open1).assign(model=model))
        elif model=="B0": all_cond.append(conditional(close,z,open1).assign(model=model))
    s=pd.DataFrame(summary); s.to_csv(RESULTS/"experiment_summary.csv",index=False); pd.DataFrame(zall["B2"]).to_parquet(RESULTS/"signals_b2.parquet",compression="zstd")
    # Frozen final hold-out: July-August is evaluated once after the above code/configuration is fixed.
    test_mask = close.index >= pd.Timestamp("2026-07-01", tz="UTC")
    held = []
    for model in ("B0", "B1", "B2"):
        z,w=build_signals(close, model, 3)
        for bp in (0, 7, 10):
            eq,tr=backtest(close.loc[test_mask], open1, z.loc[test_mask], model, w, 2, 4, bp)
            mm=metrics(eq,tr,100000); mm.update({"model":model,"cost_bp":bp,"period":"2026-07-01/2026-09-01"}); held.append(mm)
    pd.DataFrame(held).to_csv(RESULTS/"heldout_summary.csv",index=False)
    c=pd.concat(all_cond,ignore_index=True) if all_cond else pd.DataFrame(); c.to_csv(RESULTS/"conditional_returns_all.csv",index=False)
    plot_results(c,s,equity)
    (RESULTS/"research_run.json").write_text(json.dumps({"symbols":available,"rows_1m":q["total_rows"],"rows_5m":int(len(close)*len(close.columns)),"index_start":str(idx.min()),"index_end":str(idx.max()),"models":["B0","B1","B2"],"cost_bp":[0,2,5,7,10]},ensure_ascii=False,indent=2),encoding="utf-8")


def explore() -> None:
    """Run the small, predeclared sequential ablation set at 7 bp."""
    _, open1, _, close, _ = load_minute(); close = close[SYMBOLS]; open1 = open1[SYMBOLS]
    configs = [("B0", 0, 3, 4), ("B0", 0, 2, 1), ("B0", 0, 2, 24), ("B2", 1, 2, 4), ("B2", 5, 2, 4)]
    rows = []
    for model, k, entry, hold in configs:
        z, w = build_signals(close, model, k); eq, tr = backtest(close, open1, z, model, w, entry, hold, 7)
        m = metrics(eq, tr, 100000); m.update({"model": model, "pca_k": k, "entry_sigma": entry, "hold_hours": hold, "cost_bp": 7}); rows.append(m)
    pd.DataFrame(rows).to_csv(RESULTS / "parameter_exploration.csv", index=False)
    z, _ = build_signals(close, "B2", 3)
    conditional(close, z, open1).assign(model="B2").to_csv(RESULTS / "conditional_returns_b2.csv", index=False)
    # Store compact PCA diagnostics (variance share and projection residual) without
    # writing any per-parameter feature matrices.
    ret = np.log(close).diff(); diag = []
    for day, _ in ret.groupby(ret.index.floor("D")):
        end = pd.Timestamp(day); end = end.tz_localize("UTC") if end.tzinfo is None else end
        hist = ret.loc[:end - pd.Timedelta(minutes=5)].tail(28 * 24 * 12).dropna()
        if len(hist) < 28 * 24 * 12: continue
        mu = hist.to_numpy().mean(0); sd = hist.to_numpy().std(0, ddof=1); sd = np.where(sd > 1e-12, sd, 1.0)
        zz = (hist.to_numpy() - mu) / sd; _, sv, vt = np.linalg.svd(zz - zz.mean(0), full_matrices=False)
        var = sv**2 / np.sum(sv**2); d = np.diag(sd)
        for k in (1, 3, 5):
            v = vt[:k].T; m = np.eye(len(SYMBOLS)) - d @ v @ v.T @ np.linalg.inv(d)
            diag.append({"day": str(day), "k": k, "explained_variance": float(var[:k].sum()), "projection_residual_norm": float(np.max(np.abs(m @ d @ v)))})
    pd.DataFrame(diag).to_csv(RESULTS / "pca_diagnostics.csv", index=False)


def main() -> None:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    d=sub.add_parser("download"); d.add_argument("--sample",action="store_true"); d.add_argument("--workers",type=int,default=4)
    sub.add_parser("metadata"); sub.add_parser("funding"); sub.add_parser("quality"); sub.add_parser("analyze"); sub.add_parser("explore")
    a=ap.parse_args(); mkdirs()
    if a.cmd=="download": download((SYMBOLS if not a.sample else ["BTCUSDT","ETHUSDT","SUIUSDT"]),(MONTHS if not a.sample else ["2026-03"]),a.workers)
    elif a.cmd=="metadata": print(json.dumps(fetch_exchange_info(),ensure_ascii=False,indent=2))
    elif a.cmd=="funding": funding()
    elif a.cmd=="quality": print(json.dumps(quality(),ensure_ascii=False,indent=2))
    elif a.cmd=="analyze": analyze()
    elif a.cmd=="explore": explore()

if __name__=="__main__": main()
