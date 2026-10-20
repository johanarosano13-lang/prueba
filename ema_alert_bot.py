"""Bot de alertas swing por Telegram. Solo avisa; no opera.
Uso: python ema_alert_bot.py --once
     python ema_alert_bot.py --test-telegram
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import requests

WATCHLIST = [
    ("crypto", "BTC/USD"), ("crypto", "ETH/USD"),
    ("yf", "TSLA"), ("yf", "NVDA"), ("yf", "COIN"), ("yf", "MSTR"),
    ("yf", "EURUSD=X"), ("yf", "USDJPY=X"),
]
OB_LOOKBACK, OB_IMPULSE_ATR, OB_IMPULSE_CANDLES = 120, 3.0, 5
RSI_LEN, PIVOT_N, DIV_LOOKBACK = 14, 5, 120
DIV_MIN_GAP, DIV_MIN_RSI_DIFF, DIV_MIN_PRICE_ATR = 12, 5, 0.5
FAST, SLOW, GATE_1H, GATE_DIV_HOURS = 5, 10, True, 168
STATE_FILE = "alert_state.json"
TOKEN, CHAT_ID = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
NAME = {"bull": "ALCISTA", "bear": "BAJISTA"}
DELTA = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1), "1wk": pd.Timedelta(days=7)}
YF_PERIOD = {"1h": "2y", "1d": "5y", "1wk": "10y"}
OB_CACHE, _kraken = {}, None


def candles(kind, sym, tf):
    global _kraken
    if kind == "crypto":
        if _kraken is None:
            import ccxt
            _kraken = ccxt.kraken()
        # Kraken/CCXT usa 1w (no 1wk) para velas semanales.
        exchange_tf = "1w" if tf == "1wk" else tf
        raw = _kraken.fetch_ohlcv(sym, exchange_tf, limit=720)
        df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "vol"])
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    else:
        import yfinance as yf
        df = yf.download(sym, period=YF_PERIOD[tf], interval=tf, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"Yahoo Finance no devolvió velas para {sym} ({tf})")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df = df.dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + DELTA[tf] <= now]  # usar solo velas cerradas


def atr(df, n=14):
    c = df["close"]
    tr = pd.concat([df.high-df.low, (df.high-c.shift()).abs(),
                    (df.low-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=RSI_LEN):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)


def pivots(x, n, kind):
    out = []
    for i in range(n, len(x)-n):
        w = x[i-n:i+n+1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            if not out or i-out[-1] > n:
                out.append(i)
    return out


def find_obs(df):
    d = df.iloc[-OB_LOOKBACK:]
    if len(d) < 17:
        return []
    a = atr(d).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    obs = []
    for i in range(14, len(d)-2):
        if np.isnan(a[i]):
            continue
        end = min(i+1+OB_IMPULSE_CANDLES, len(d))
        if c[i] < o[i] and c[i+1] > o[i+1]:
            if h[i+1:end].max()-c[i] >= OB_IMPULSE_ATR*a[i] and not (c[i+1:] < l[i]).any():
                obs.append(("bull", l[i], h[i], d.index[i]))
        elif c[i] > o[i] and c[i+1] < o[i+1]:
            if c[i]-l[i+1:end].min() >= OB_IMPULSE_ATR*a[i] and not (c[i+1:] > h[i]).any():
                obs.append(("bear", l[i], h[i], d.index[i]))
    return obs


def send(msg):
    """Envía mensaje e informa errores sin imprimir credenciales."""
    if not TOKEN or not CHAT_ID:
        print("[ERROR] Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID.")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.ok and data.get("ok") is True:
            print("[OK] Telegram aceptó el mensaje.")
            return True
        print(f"[ERROR TELEGRAM] HTTP {r.status_code}: {data.get('description', 'respuesta no reconocida')}")
    except requests.RequestException as e:
        print(f"[ERROR DE CONEXIÓN TELEGRAM] {type(e).__name__}: {e}")
    return False


def check_ob(sym, px, obs, st, ctx):
    prev, new = st["ob"].get(sym, {}), {}
    ctx["ob"] = None
    for direction, lo, hi, ts in obs:
        oid, inside = f"{direction}|{ts:%Y-%m-%d}", bool(lo <= px <= hi)
        new[oid] = inside
        if inside:
            ctx["ob"] = direction
            if not prev.get(oid, False):
                send(f"{sym} · SEMANAL\nPrecio dentro de OB {NAME[direction]} grande\nZona: {lo:.5g} – {hi:.5g}\nFormado: {ts:%Y-%m-%d}\nPrecio: {px:.5g}")
    st["ob"][sym] = new


def find_divergence(df):
    r = rsi(df.close).iloc[-DIV_LOOKBACK:].values
    d = df.iloc[-DIV_LOOKBACK:]
    a, n = atr(df).iloc[-1], len(d)
    if n < 40 or np.isnan(a):
        return None
    recent = lambda i: i >= n-1-PIVOT_N-2
    lo, hi = d.low.values, d.high.values
    pl = pivots(lo, PIVOT_N, "low")
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        if (recent(i2) and i2-i1 >= DIV_MIN_GAP and lo[i1]-lo[i2] >= DIV_MIN_PRICE_ATR*a
                and r[i2]-r[i1] >= DIV_MIN_RSI_DIFF and r[i1] < 40):
            return "bull", d.index[i2], r[i1], r[i2], lo[i1], lo[i2]
    ph = pivots(hi, PIVOT_N, "high")
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        if (recent(i2) and i2-i1 >= DIV_MIN_GAP and hi[i2]-hi[i1] >= DIV_MIN_PRICE_ATR*a
                and r[i1]-r[i2] >= DIV_MIN_RSI_DIFF and r[i1] > 60):
            return "bear", d.index[i2], r[i1], r[i2], hi[i1], hi[i2]
    return None


def check_div(sym, d1, st, ctx):
    res = find_divergence(d1)
    if not res:
        return
    direction, ts, r1, r2, p1, p2 = res
    ctx["div"] = {"dir": direction, "seen": pd.Timestamp.now(tz="UTC").isoformat()}
    key = f"{direction}|{ts}"
    if st["last"].get(f"{sym}|1d") == key:
        return
    st["last"][f"{sym}|1d"] = key
    send(f"{sym} · DIARIO\nDivergencia {NAME[direction]} grande en RSI\nPrecio: {p1:.5g} → {p2:.5g}\nRSI: {r1:.1f} → {r2:.1f}\nPivote: {ts:%Y-%m-%d %H:%M} UTC")


def check_cross(sym, h1, st, ctx):
    c = h1.close
    gap = c.ewm(span=FAST, adjust=False).mean()-c.ewm(span=SLOW, adjust=False).mean()
    g0, g1 = gap.iloc[-2], gap.iloc[-1]
    direction = "bull" if g0 <= 0 < g1 else "bear" if g0 >= 0 > g1 else None
    if not direction:
        return
    ts, key = h1.index[-1], f"{direction}|{h1.index[-1]}"
    if st["last"].get(f"{sym}|1h") == key:
        return
    st["last"][f"{sym}|1h"] = key
    why = []
    if ctx.get("ob") == direction:
        why.append("precio en OB semanal " + NAME[direction].lower())
    div = ctx.get("div")
    if div and div["dir"] == direction and pd.Timestamp.now(tz="UTC")-pd.Timestamp(div["seen"]) <= pd.Timedelta(hours=GATE_DIV_HOURS):
        why.append("divergencia RSI diaria " + NAME[direction].lower())
    if GATE_1H and not why:
        return
    send(f"{sym} · HORARIO\nCruce {NAME[direction]} EMA{FAST}/EMA{SLOW}\nPrecio: {c.iloc[-1]:.5g}\nContexto: {', '.join(why) if why else 'sin confluencia'}\nVela: {ts:%Y-%m-%d %H:%M} UTC")


def run_symbol(kind, sym, st):
    ctx = st["ctx"].setdefault(sym, {})
    h1 = candles(kind, sym, "1h")
    if len(h1) < SLOW+2:
        raise ValueError("No hay suficientes velas horarias cerradas")
    d1 = candles(kind, sym, "1d")
    if d1.empty:
        raise ValueError("No hay velas diarias cerradas")
    OB_CACHE[sym] = find_obs(candles(kind, sym, "1wk"))
    check_ob(sym, float(h1.close.iloc[-1]), OB_CACHE[sym], st, ctx)
    check_div(sym, d1, st, ctx)
    check_cross(sym, h1, st, ctx)


def load_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            st = json.load(f)
    except (OSError, ValueError):
        st = {}
    for k in ("last", "ob", "ctx"):
        st.setdefault(k, {})
    return st


def main():
    if "--test-telegram" in sys.argv:
        sys.exit(0 if send("Prueba de Telegram: el bot EMA puede enviar mensajes.") else 1)
    once, st = "--once" in sys.argv, load_state()
    while True:
        now = pd.Timestamp.now(tz="UTC")
        for kind, sym in WATCHLIST:
            try:
                run_symbol(kind, sym, st)
            except Exception as e:
                print(f"Error con {sym}: {type(e).__name__}: {e}")
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        if once:
            break
        nxt = now.floor("h") + pd.Timedelta(hours=1, minutes=5)
        time.sleep(max(5, (nxt-pd.Timestamp.now(tz="UTC")).total_seconds()))


if __name__ == "__main__":
    main()
