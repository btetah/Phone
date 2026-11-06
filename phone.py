#!/usr/bin/env python3
"""
phonemodem.py - send a picture through an ordinary phone call.
DSP is numpy only. Pillow is used just to open/save images. WAV via the stdlib.

Commands
  test-tx  out.wav                  channel test signal (play it into the call)
  test-rx  rec.wav                  what did the call do to each frequency?

  sstv-tx  photo.jpg out.wav        v1 analog: brightness -> frequency (~31 s)
  sstv-rx  rec.wav out.png

  fsk-tx   photo.jpg out.wav        v2 digital: 16 tones = 16 grey levels
  fsk-rx   rec.wav out.png          (use the same --size / --sym on both sides)

  simulate in.wav out.wav           fake phone line (300-3400 Hz, 8 kHz, mu-law,
                                    clock drift, echo, AGC, noise) for testing at home

Recordings must be WAV (PCM). Convert phone recordings (m4a/amr/mp3) first,
e.g. with Audacity: File > Export > WAV (16-bit PCM).
"""
import argparse
import wave

import numpy as np
from PIL import Image, ImageOps

TX_SR = 48000          # sample rate of generated audio
RX_SR = 16000          # everything received is resampled to this first

# ----------------------------------------------------------------- WAV I/O

def write_wav(path, x, sr=TX_SR):
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) or 1.0
    pcm = np.round(0.8 * x / peak * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    print(f"wrote {path}  ({len(x) / sr:.1f} s)")


def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float64) - 128) / 128
    elif sw == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float64) / 2 ** 31
    else:
        raise ValueError("unsupported WAV sample width")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr

# ------------------------------------------------------------ DSP helpers

def resample(x, sr, target):
    """FFT low-pass, then linear interpolation onto the new time grid."""
    if sr == target:
        return x.copy()
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X[f > 0.45 * min(sr, target)] = 0
    y = np.fft.irfft(X, len(x))
    t_new = np.arange(int(len(x) * target / sr)) / target
    return np.interp(t_new, np.arange(len(x)) / sr, y)


def bandpass(x, sr, lo, hi):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def analytic(x):
    """Analytic signal (Hilbert transform) with numpy's FFT."""
    N = len(x)
    h = np.zeros(N)
    h[0] = 1
    if N % 2 == 0:
        h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[1:(N + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(x) * h)


def fm(freqs, sr=TX_SR):
    """Continuous-phase oscillator: one frequency value per output sample."""
    return np.sin(2 * np.pi * np.cumsum(freqs) / sr)


def tone(f, dur, sr=TX_SR):
    return np.full(int(round(dur * sr)), float(f))


def sweep(f0, f1, dur, sr=TX_SR):
    return np.linspace(f0, f1, int(round(dur * sr)))


def find_chirp(x, sr, f0, f1, dur):
    """Matched filter. Returns (start index, peak quality). Phase/polarity proof."""
    ref = np.exp(2j * np.pi * np.cumsum(sweep(f0, f1, dur, sr)) / sr)
    n = 1 << int(np.ceil(np.log2(len(x) + len(ref))))
    c = np.abs(np.fft.ifft(np.fft.fft(x, n) * np.conj(np.fft.fft(ref, n)))[:len(x)])
    k = int(np.argmax(c))
    return k, c[k] / (np.median(c) + 1e-12)


def locate(x, sr, up, down, gap):
    """Find start chirp and end chirp. `gap` = seconds between their starts at TX.
    Returns (start index, samples per TX-second) - the second value absorbs clock drift."""
    k1, q1 = find_chirp(x, sr, *up)
    k2, q2 = find_chirp(x, sr, *down)
    ratio = (k2 - k1) / (gap * sr)
    print(f"start chirp at {k1 / sr:.2f} s (quality {q1:.0f}),  "
          f"end chirp at {k2 / sr:.2f} s (quality {q2:.0f})")
    if not 0.97 < ratio < 1.03:
        print("  warning: end chirp not found cleanly, assuming no clock drift")
        ratio = 1.0
    else:
        print(f"  clock drift: {(ratio - 1) * 1e6:+.0f} ppm")
    if q1 < 8:
        print("  warning: start chirp is weak - is this the right recording?")
    return k1, sr * ratio


def load_gray(path, w, h):
    im = ImageOps.fit(Image.open(path).convert("L"), (w, h))
    return np.asarray(im, dtype=np.float64) / 255


def save_gray(path, g, scale=4):
    im = Image.fromarray(np.uint8(np.clip(g, 0, 1) * 255))
    im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path)
    print(f"wrote {path}")

# ------------------------------------------------------------ channel test

TEST_TONES = np.arange(300, 4001, 100)
TEST_UP = (300, 4000, 1.0)
TEST_DOWN = (4000, 300, 1.0)
TEST_STEP = 0.3


def cmd_test_tx(a):
    parts = [tone(1000, 1.0), sweep(*TEST_UP), tone(1000, 0.2)]
    parts += [tone(f, TEST_STEP) for f in TEST_TONES]
    parts += [tone(1000, 0.2), sweep(*TEST_DOWN), tone(1000, 0.5)]
    write_wav(a.out, fm(np.concatenate(parts)))


def cmd_test_rx(a):
    x, sr = read_wav(a.rec)
    x, sr = resample(x, sr, RX_SR), RX_SR
    gap = 1.0 + 0.2 + len(TEST_TONES) * TEST_STEP + 0.2
    k1, s = locate(x, sr, TEST_UP, TEST_DOWN, gap)
    rows = []
    for i, f in enumerate(TEST_TONES):
        t0 = 1.2 + i * TEST_STEP
        lo, hi = int(k1 + (t0 + 0.05) * s), int(k1 + (t0 + TEST_STEP - 0.05) * s)
        seg = x[lo:hi]
        n = np.arange(len(seg))
        X = np.sum(seg * np.exp(-2j * np.pi * f * n / sr))
        power = abs(X) ** 2 / len(seg)
        purity = 2 * abs(X) ** 2 / (len(seg) * np.sum(seg ** 2) + 1e-12)
        rows.append((f, power, purity))
    pmax = max(r[1] for r in rows) + 1e-20
    print("\n freq    level   purity")
    good = []
    for f, p, pur in rows:
        db = 10 * np.log10(p / pmax + 1e-12)
        bar = "#" * max(0, int(40 + db))  # one # per dB above -40 dB
        print(f"{f:5d} Hz {db:6.1f} dB  {pur * 100:4.0f}%  {bar}")
        if db > -10 and pur > 0.5:
            good.append(f)
    if good:
        print(f"\nusable band (within 10 dB, >50% pure): {good[0]}-{good[-1]} Hz")

# ----------------------------------------------- v1: analog (SSTV-like)

SW, SH = 160, 120                 # picture size
S_SYNC, S_PORCH, S_PIX = 0.008, 0.002, 0.0015
S_LINE = S_SYNC + S_PORCH + SW * S_PIX
S_UP, S_DOWN = (1000, 2500, 0.5), (2500, 1000, 0.5)
F_SYNC, F_BLACK, F_WHITE = 1200, 1500, 2300


def cmd_sstv_tx(a):
    g = load_gray(a.img, SW, SH)
    parts = [tone(1900, 0.3), sweep(*S_UP), tone(F_BLACK, 0.05)]
    for row in g:
        parts += [tone(F_SYNC, S_SYNC), tone(F_BLACK, S_PORCH),
                  np.repeat(F_BLACK + (F_WHITE - F_BLACK) * row, int(round(S_PIX * TX_SR)))]
    parts += [tone(F_BLACK, 0.05), sweep(*S_DOWN), tone(1900, 0.3)]
    write_wav(a.out, fm(np.concatenate(parts)))


def cmd_sstv_rx(a):
    x, sr = read_wav(a.rec)
    x, sr = resample(x, sr, RX_SR), RX_SR
    head = S_UP[2] + 0.05
    k1, s = locate(x, sr, S_UP, S_DOWN, head + SH * S_LINE + 0.05)

    z = analytic(bandpass(x, sr, 1000, 2600))
    f = np.diff(np.unwrap(np.angle(z))) * sr / (2 * np.pi)
    f = np.append(f, f[-1])
    w = max(1, int(0.0004 * sr))
    f = np.convolve(f, np.ones(w) / w, mode="same")

    in_sync = np.concatenate([[0], np.cumsum(f < (F_SYNC + F_BLACK) / 2)])
    n_sync = int(round(S_SYNC * s))
    search = int(0.003 * sr)
    img = np.zeros((SH, SW))
    locked = 0
    for i in range(SH):
        exp = int(round(k1 + (head + i * S_LINE) * s))
        best, score = exp, -1.0
        for d in range(-search, search + 1):
            p = exp + d
            if 0 <= p and p + n_sync < len(in_sync):
                sc = (in_sync[p + n_sync] - in_sync[p]) / n_sync
                if sc > score:
                    best, score = p, sc
        start = best if score > 0.6 else exp
        locked += score > 0.6
        for j in range(SW):
            p0 = start + (S_SYNC + S_PORCH + j * S_PIX) * s
            lo, hi = int(p0 + 0.2 * S_PIX * s), int(p0 + 0.8 * S_PIX * s) + 1
            if hi <= len(f):
                img[i, j] = (np.median(f[lo:hi]) - F_BLACK) / (F_WHITE - F_BLACK)
    print(f"line syncs found: {locked}/{SH}")
    save_gray(a.out, img)

# ------------------------------------------ v2: digital (16-FSK pixels)

F_BASE, F_STEP, N_TONES = 600, 150, 16
D_UP, D_DOWN = (600, 2900, 0.5), (2900, 600, 0.5)
TRAIN = np.arange(N_TONES)        # known symbols sent first: a channel check


def fsk_gap(size, sym):
    return D_UP[2] + 0.05 + (len(TRAIN) + size * size) * sym + 0.05


def cmd_fsk_tx(a):
    g = load_gray(a.img, a.size, a.size)
    levels = np.clip(np.round(g * (N_TONES - 1)), 0, N_TONES - 1).astype(int).ravel()
    symbols = np.concatenate([TRAIN, levels])
    parts = [tone(1900, 0.3), sweep(*D_UP), tone(1900, 0.05),
             np.repeat(F_BASE + F_STEP * symbols, int(round(a.sym * TX_SR))),
             tone(1900, 0.05), sweep(*D_DOWN), tone(1900, 0.3)]
    write_wav(a.out, fm(np.concatenate(parts)))
    print(f"  remember: fsk-rx --size {a.size} --sym {a.sym}")


def cmd_fsk_rx(a):
    x, sr = read_wav(a.rec)
    x, sr = resample(x, sr, RX_SR), RX_SR
    k1, s = locate(x, sr, D_UP, D_DOWN, fsk_gap(a.size, a.sym))
    n_sym = len(TRAIN) + a.size * a.size

    # look only at the middle 60% of every symbol (edges are smeared by the codec/echo)
    L = int(0.6 * a.sym * s)
    starts = np.round(k1 + (D_UP[2] + 0.05 + (np.arange(n_sym) + 0.2) * a.sym) * s).astype(int)
    starts = np.clip(starts, 0, len(x) - L - 1)
    segs = x[starts[:, None] + np.arange(L)]                       # (n_sym, L)
    freqs = F_BASE + F_STEP * np.arange(N_TONES)
    basis = np.exp(-2j * np.pi * np.outer(np.arange(L), freqs) / sr)  # (L, 16)
    P = np.abs(segs @ basis) ** 2                                  # tone energies
    sym = np.argmax(P, axis=1)
    top2 = np.sort(P, axis=1)[:, -2:]
    margin = 10 * np.log10(top2[:, 1] / (top2[:, 0] + 1e-20))

    ok = int(np.sum(sym[:len(TRAIN)] == TRAIN))
    print(f"training symbols correct: {ok}/{len(TRAIN)}")
    print(f"median decision margin: {np.median(margin):.1f} dB "
          f"(weak symbols < 3 dB: {np.mean(margin < 3) * 100:.1f}%)")
    pix = sym[len(TRAIN):].reshape(a.size, a.size) / (N_TONES - 1)
    save_gray(a.out, pix, scale=max(1, 256 // a.size))

# ------------------------------------------------------ fake phone line

def cmd_simulate(a):
    rng = np.random.default_rng(a.seed)
    x, sr = read_wav(a.inp)
    y = bandpass(x, sr, 300, 3400)
    y = np.interp(np.arange(int(len(y) * a.drift)) / a.drift, np.arange(len(y)), y)  # clock drift
    y = resample(y, sr, 8000)
    sr = 8000
    y = y / np.max(np.abs(y)) * 0.7
    mu = 255.0                                                      # G.711 mu-law, 8 bit
    q = np.round(np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu) * 127) / 127
    y = np.sign(q) * np.expm1(np.abs(q) * np.log1p(mu)) / mu
    d = int(0.007 * sr)                                             # one room echo
    y[d:] += 0.25 * y[:-d]
    t = np.arange(len(y)) / sr
    y *= 1 + 0.3 * np.sin(2 * np.pi * 0.3 * t)                      # AGC breathing
    y += rng.standard_normal(len(y)) * np.sqrt(np.mean(y ** 2) / 10 ** (a.snr / 10))
    pad = rng.standard_normal(int(1.5 * sr)) * 0.01
    y = -np.concatenate([pad, y, pad])                              # flipped polarity
    write_wav(a.out, y, sr)

# --------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("test-tx"); s.add_argument("out"); s.set_defaults(fn=cmd_test_tx)
    s = sub.add_parser("test-rx"); s.add_argument("rec"); s.set_defaults(fn=cmd_test_rx)
    s = sub.add_parser("sstv-tx"); s.add_argument("img"); s.add_argument("out"); s.set_defaults(fn=cmd_sstv_tx)
    s = sub.add_parser("sstv-rx"); s.add_argument("rec"); s.add_argument("out"); s.set_defaults(fn=cmd_sstv_rx)
    for name, fn, args in (("fsk-tx", cmd_fsk_tx, ("img", "out")), ("fsk-rx", cmd_fsk_rx, ("rec", "out"))):
        s = sub.add_parser(name)
        for arg in args:
            s.add_argument(arg)
        s.add_argument("--size", type=int, default=64, help="picture is size x size (default 64)")
        s.add_argument("--sym", type=float, default=0.04, help="seconds per symbol (default 0.04)")
        s.set_defaults(fn=fn)
    s = sub.add_parser("simulate"); s.add_argument("inp"); s.add_argument("out")
    s.add_argument("--snr", type=float, default=25); s.add_argument("--drift", type=float, default=1.0003)
    s.add_argument("--seed", type=int, default=1); s.set_defaults(fn=cmd_simulate)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
