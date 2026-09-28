#!/usr/bin/env python3
"""
算了嘛，警官 —— 用 Hopf 极限环解释驾照为什么永远拿不出来。

对话只有四句，循环往复：
    警察：你有没有驾照？   司机：有。
    警察：拿出来。         司机：算了嘛，警官。

模型（Hopf 标准型，极坐标）：
    dr/dt = r (μ − r²)
    dθ/dt = ω = 2π / 5.05 s          （一轮对话约 5 秒）

    μ = 司机嘴硬程度 − 警察强硬程度
    原点 r = 0 是「出示驾照」。μ > 0 时它不稳定，
    所有轨迹都被吸到半径 √μ 的极限环上，四句台词无限循环。

用法：
    python render.py                 # 渲染 media/suanlema_hopf.mp4
    python render.py --still 20      # 只导出第 20 秒的一帧，调版面用
    python render.py --gif           # 渲染完再截一段做 README 预览 GIF
"""
import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 模型参数
PERIOD = 5.05                 # 一轮「问—有—拿出来—算了嘛」的秒数
OMEGA = 2 * math.pi / PERIOD
MU = 0.8                      # 司机嘴硬程度 − 警察强硬程度，全程 > 0
R_VIEW = 1.75                 # 相平面显示范围 ±R_VIEW
R_FLOOR = 3e-3                # 数值噪声地板，免得点永远卡在原点
EPS = 0.01                    # P(t) 里「离原点多近算快掏出来了」的尺度
SULE_R = 0.02                 # 只排除数值噪声；再小声的「算了嘛」也算

SIM_START = 2.2               # 片头开始淡出时开始积分
# 警察两次加大力度：最早时刻、推到的半径。都卡在进入「拿出来」那一段时出手
KICKS = [(19.0, 0.12), (27.0, 0.05)]
KICK_LEN = 0.6                # 一次推的过程（秒）

LINES = [("警察", "你有没有驾照？"), ("司机", "有。"),
         ("警察", "拿出来。"), ("司机", "算了嘛，警官。")]
SECTOR_LABELS = [("问", 0, 1.58), ("有", -1.58, 0), ("拿出来", 0, -1.58), ("算了嘛", 1.58, 0)]

# ---------------------------------------------------------------- 配色（深色）
BG = "#1a1a19"
PANEL = "#222220"
GRID = "#34342f"
TEXT = "#f0efec"
SEC = "#c3c2b7"
MUTED = "#898781"
DRIVER = "#F0997B"            # 司机 / 主轨迹
POLICE = "#85B7EB"            # 警察
CYCLE = "#AFA9EC"             # 极限环
UNSTABLE = "#F09595"          # 不稳定平衡点
GHOST = "#B4B2A9"             # 其他司机

W_PX, H_PX, DPI = 1920, 1080, 100


# ---------------------------------------------------------------- 数值部分
def field(p):
    x, y = p[:, 0], p[:, 1]
    r2 = x * x + y * y
    return np.stack([MU * x - OMEGA * y - x * r2, OMEGA * x + MU * y - y * r2], axis=1)


def rk4(p, h):
    k1 = field(p)
    k2 = field(p + h / 2 * k1)
    k3 = field(p + h / 2 * k2)
    k4 = field(p + h * k3)
    p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r = np.hypot(p[:, 0], p[:, 1])
    small = r < R_FLOOR
    if small.any():
        p[small] *= (R_FLOOR / np.maximum(r[small], 1e-12))[:, None]
    return p


def phase(x, y):
    """把角度换成对话进度 u∈[0,1)：0 问 / .25 有 / .5 拿出来 / .75 算了嘛（逆时针）"""
    th = math.atan2(y, x)
    return ((th - math.pi / 4) % (2 * math.pi)) / (2 * math.pi)


def bump(u):
    """一圈里的掏驾照进度：一路往上爬，「算了嘛」出口瞬间清零"""
    if u < 0.75:
        return 0.03 + 0.82 * (u / 0.75) ** 1.6
    return 0.03 + 0.82 * math.exp(-(u - 0.75) / 0.018)


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def simulate(fps, t_max, substeps=6):
    starts = [(0.08, math.pi / 2)]                       # 主角：刚被拦下，离出示驾照只差一点
    starts += [(r, 0.4 + 2.1 * i) for i, r in enumerate([0.02, 0.35, 0.7, 1.05, 1.45, 1.65])]
    p = np.array([[r * math.cos(a), r * math.sin(a)] for r, a in starts])

    n = int(round(t_max * fps))
    out = {k: np.zeros(n) for k in ("P", "r")}
    out["pos"] = np.zeros((n, len(p), 2))
    out["sec"] = np.zeros(n, int)
    out["sule"] = np.zeros(n, int)
    out["kick"] = np.zeros(n, bool)
    kick_starts = []

    h = 1 / fps / substeps
    prev, count = -1, 0
    pending = list(KICKS)
    active = None                                         # (开始时刻, 起始半径, 目标半径)
    for i in range(n):
        t0 = (i - 1) / fps
        tv = i / fps
        if i > 0 and t0 >= SIM_START:
            for _ in range(substeps):
                p = rk4(p, h)
            if active:
                ts, r0, r1 = active
                x, y = p[0]
                r_now = math.hypot(x, y)
                target = r0 + (r1 - r0) * smoothstep((tv - ts) / KICK_LEN)
                p[0] *= target / r_now
                if tv - ts >= KICK_LEN:
                    active = None
        x, y = p[0]
        r = math.hypot(x, y)
        u = phase(x, y)
        s = min(3, int(u * 4))
        if pending and tv >= pending[0][0] and prev == 1 and s == 2 and not active:
            active = (tv, r, pending.pop(0)[1])
            kick_starts.append(tv)
        if prev == 2 and s == 3 and r > SULE_R:
            count += 1
        prev = s
        w = math.exp(-r * r / EPS)
        out["pos"][i] = p
        out["r"][i] = r
        out["sec"][i] = s
        out["sule"][i] = count
        out["kick"][i] = active is not None
        out["P"][i] = (1 - w) * bump(u) + w
    out["kick_starts"] = kick_starts
    return out


# ---------------------------------------------------------------- 字体
CJK_CANDIDATES = ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "Noto Sans SC",
                  "Source Han Sans SC", "Microsoft YaHei", "WenQuanYi Zen Hei", "Arial Unicode MS"]


def setup_fonts():
    have = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in CJK_CANDIDATES if n in have]
    if not cjk:
        sys.exit("没找到中文字体。装一个 Noto Sans CJK SC（或思源黑体）再跑。")
    plt.rcParams.update({
        "font.family": [cjk[0], "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })


# ---------------------------------------------------------------- 画面
def ramp(t, a, b):
    return min(1.0, max(0.0, (t - a) / (b - a)))


class Scene:
    def __init__(self, sim, fps):
        self.sim, self.fps = sim, fps
        k1, k2 = sim["kick_starts"]
        self.end_in = k2 + 8.5
        self.duration = self.end_in + 5.0
        floquet = math.exp(-2 * MU * PERIOD) * 100
        self.captions = [
            (3.0, 9.0, "μ > 0：原点「出示驾照」不稳定，差一点点也会被甩出去"),
            (9.0, 14.0, "不管从哪出发，所有司机最后都落到同一个圈上：极限环 r = √μ"),
            (14.0, k1, "每一圈：有没有驾照？→ 有 → 拿出来 → 算了嘛，警官（约 5 秒一圈）"),
            (k1, k1 + 3.0, "警察加大力度：一把推到离「出示驾照」很近的地方……"),
            (k1 + 3.0, k2, "……可原点不稳定，几圈就被甩回原来的循环"),
            (k2, k2 + 3.0, "再推一次，更近了——"),
            (k2 + 3.0, self.end_in,
             rf"还是「算了嘛」。偏离每转一圈只剩 $e^{{-2\mu T}} \approx {floquet:.2f}\%$"),
        ]
        self.fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
        self._build()

    # ---- 搭建所有图元
    def _build(self):
        fig = self.fig
        ax = fig.add_axes([0.035, 0.10, 0.461, 0.82])
        ax.set_facecolor(BG)
        ax.set_xlim(-R_VIEW, R_VIEW)
        ax.set_ylim(-R_VIEW, R_VIEW)
        ax.set_aspect("equal")
        ax.set_axis_off()
        self.ax = ax

        g = -1.625 + 0.25 * np.arange(14)
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        v = field(pts)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        ax.quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], color=MUTED, alpha=0.38,
                  angles="xy", scale_units="xy", scale=1 / 0.11, pivot="mid",
                  width=0.0022, headwidth=4, headlength=4.5, headaxislength=4, zorder=1)

        for sgn in (1, -1):
            ax.plot([-R_VIEW, R_VIEW], [-sgn * R_VIEW, sgn * R_VIEW], color=GRID,
                    lw=1.2, ls=(0, (3, 4)), zorder=1)
        th = np.linspace(0, 2 * np.pi, 400)
        rc = math.sqrt(MU)
        ax.plot(rc * np.cos(th), rc * np.sin(th), color=CYCLE, lw=2.2, ls=(0, (6, 5)), zorder=2)
        ax.text(rc * math.cos(2.45) - 0.05, rc * math.sin(2.45) + 0.1, "极限环", color=CYCLE,
                fontsize=15, ha="right", va="bottom", zorder=2)

        self.sector_txt = [ax.text(x, y, s, ha="center", va="center", fontsize=22, color=MUTED, zorder=3)
                           for s, x, y in SECTOR_LABELS]

        self.ghost_lc = LineCollection([], linewidths=1.3, zorder=3, capstyle="round")
        self.main_lc = LineCollection([], linewidths=3.0, zorder=4, capstyle="round")
        ax.add_collection(self.ghost_lc)
        ax.add_collection(self.main_lc)

        ax.plot([0], [0], marker="o", ms=15, mew=2.5, mfc="none", mec=UNSTABLE, zorder=5)
        ax.text(0, 0.1, "出示驾照（不稳定）", color=UNSTABLE, fontsize=16, ha="center", va="bottom", zorder=5)

        self.ghost_dots, = ax.plot([], [], "o", color=GHOST, ms=6, alpha=0.85, zorder=6)
        self.halo, = ax.plot([], [], "o", color=DRIVER, ms=28, alpha=0.18, mew=0, zorder=6)
        self.push_ring, = ax.plot([], [], "o", mfc="none", mec=POLICE, ms=40, mew=2.5, zorder=6)
        self.main_dot, = ax.plot([], [], "o", color=DRIVER, ms=13, zorder=7)

        ax.text(R_VIEW - 0.05, -R_VIEW + 0.05, "x：警察施压 →", color=MUTED, fontsize=15, ha="right", va="bottom")
        ax.text(-R_VIEW + 0.05, R_VIEW - 0.05, "↑ y：司机心虚", color=MUTED, fontsize=15, ha="left", va="top")

        # 右栏：μ 读数
        fig.text(0.545, 0.905, f"μ = {MU:+.2f}", family="DejaVu Sans Mono", fontsize=34, color=TEXT, va="center")
        fig.text(0.69, 0.905, "极限环：无限循环", fontsize=26, color=CYCLE, va="center")

        # 右栏：台词框
        box = FancyBboxPatch((0.545, 0.655), 0.42, 0.20, transform=fig.transFigure,
                             boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=W_PX / H_PX,
                             facecolor=PANEL, edgecolor=GRID, lw=1.2)
        fig.patches.append(box)
        self.who = fig.text(0.565, 0.815, "", fontsize=20, color=MUTED, va="center")
        self.line = fig.text(0.565, 0.748, "", fontsize=40, color=TEXT, va="center")
        self.stat = fig.text(0.565, 0.685, "", fontsize=18, color=SEC, va="center")

        # 右栏：分岔图
        fig.text(0.545, 0.60, "分岔图：μ > 0 这一侧只有极限环是稳定的", fontsize=18, color=SEC, va="center")
        axb = fig.add_axes([0.565, 0.355, 0.39, 0.215])
        axb.set_facecolor(BG)
        axb.set_axis_off()
        axb.set_xlim(-1, 1.5)
        axb.set_ylim(-1.4, 1.4)
        axb.plot([0, 0], [-1.4, 1.4], color=GRID, lw=1)
        axb.plot([-1, 0], [0, 0], color="#5DCAA5", lw=2.4)
        axb.plot([0, 1.5], [0, 0], color=UNSTABLE, lw=2.4, ls=(0, (4, 4)))
        m = np.linspace(0, 1.5, 200)
        for sgn in (1, -1):
            axb.plot(m, sgn * np.sqrt(m), color=CYCLE, lw=2.4)
        axb.plot([MU, MU], [-1.4, 1.4], color=SEC, lw=1, ls=(0, (2, 3)))
        axb.text(-0.5, 0.1, "出示驾照（仅 μ<0 稳定）", color="#5DCAA5", fontsize=13, ha="center", va="bottom")
        axb.text(1.5, 0.75, "极限环 ±√μ", color=CYCLE, fontsize=14, ha="right", va="top")
        axb.text(0, -1.4, "μ = 0（Hopf 点）", color=MUTED, fontsize=13, ha="center", va="top")
        axb.text(MU, 1.4, "这位司机", color=DRIVER, fontsize=13, ha="center", va="bottom")
        self.bif_dot, = axb.plot([], [], "o", color=DRIVER, ms=10, zorder=5)

        # 右栏：P(t)
        fig.text(0.545, 0.29, "掏驾照进度 P(t)：「拿出来」时往上爬，一句「算了嘛」清零",
                 fontsize=18, color=SEC, va="center")
        axp = fig.add_axes([0.58, 0.11, 0.375, 0.155])
        axp.set_facecolor(BG)
        axp.set_axis_off()
        axp.set_xlim(-20, 0)
        axp.set_ylim(-0.05, 1.1)
        for yv in (0, 0.5, 1):
            axp.plot([-20, 0], [yv, yv], color=GRID, lw=1)
        axp.text(-20.4, 1, "掏出", color=MUTED, fontsize=13, ha="right", va="center")
        axp.text(-20.4, 0, "0", color=MUTED, fontsize=13, ha="right", va="center")
        axp.text(-20, -0.12, "20 秒前", color=MUTED, fontsize=13, ha="left", va="top")
        axp.text(0, -0.12, "现在", color=MUTED, fontsize=13, ha="right", va="top")
        self.p_line, = axp.plot([], [], color=DRIVER, lw=2)

        self.caption = fig.text(0.5, 0.04, "", fontsize=24, color=TEXT, ha="center", va="center")

        # 片头 / 片尾覆盖层
        ov = fig.add_axes([0, 0, 1, 1], zorder=50)
        ov.set_axis_off()
        ov.set_xlim(0, 1)
        ov.set_ylim(0, 1)
        self.cover = Rectangle((0, 0), 1, 1, transform=ov.transAxes, color=BG, zorder=0)
        ov.add_patch(self.cover)
        kw = dict(ha="center", va="center", transform=ov.transAxes)
        self.title_txt = [
            ov.text(0.5, 0.60, "算了嘛，警官", fontsize=80, color=TEXT, **kw),
            ov.text(0.5, 0.48, "一个 Hopf 极限环：驾照为什么永远拿不出来", fontsize=32, color=SEC, **kw),
            ov.text(0.5, 0.38, r"$\dot r = r(\mu - r^2),\qquad \dot\theta = \omega$", fontsize=30, color=MUTED, **kw),
        ]
        self.end_txt = [
            ov.text(0.5, 0.66, "这个系统的吸引子是「算了嘛」，不是「出示驾照」", fontsize=42, color=TEXT, **kw),
            ov.text(0.5, 0.53, "有没有驾照？→ 有 → 拿出来 → 算了嘛，警官 → 有没有驾照？→ …",
                    fontsize=30, color=CYCLE, **kw),
            ov.text(0.5, 0.42, "推得再近也会被甩回去；要他掏出来，得让 μ < 0", fontsize=28, color=SEC, **kw),
            ov.text(0.5, 0.30, "μ = 司机嘴硬程度 − 警察强硬程度", fontsize=26, color=MUTED, **kw),
            ov.text(0.5, 0.10, r"$\dot r = r(\mu - r^2),\quad \dot\theta = 2\pi / 5.05\,\mathrm{s},\quad \mu = 0.8$",
                    fontsize=22, color=MUTED, **kw),
        ]
        self.last_key = None

    # ---- 更新第 i 帧
    def update(self, i):
        s, fps = self.sim, self.fps
        tv = i / fps
        pos = s["pos"]
        mx, my = pos[i, 0]
        r = s["r"][i]

        k_main, k_ghost = int(3.2 * fps), int(2.0 * fps)
        j = max(0, i - k_main)
        pts = pos[j:i + 1, 0]
        segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.zeros((0, 2, 2))
        col = np.tile(to_rgba(DRIVER), (len(segs), 1))
        col[:, 3] = np.linspace(0, 0.95, len(segs)) if len(segs) else col[:, 3]
        self.main_lc.set_segments(segs)
        self.main_lc.set_color(col)

        j = max(0, i - k_ghost)
        gsegs, gcol = [], []
        for g in range(1, pos.shape[1]):
            gp = pos[j:i + 1, g]
            if len(gp) > 1:
                gsegs.append(np.stack([gp[:-1], gp[1:]], axis=1))
                c = np.tile(to_rgba(GHOST), (len(gp) - 1, 1))
                c[:, 3] = np.linspace(0, 0.5, len(gp) - 1)
                gcol.append(c)
        self.ghost_lc.set_segments(np.concatenate(gsegs) if gsegs else [])
        if gcol:
            self.ghost_lc.set_color(np.concatenate(gcol))
        self.ghost_dots.set_data(pos[i, 1:, 0], pos[i, 1:, 1])

        self.main_dot.set_data([mx], [my])
        self.halo.set_data([mx], [my])
        since = min((tv - k for k in s["kick_starts"] if tv >= k), default=None)
        pushing = since is not None and since < 1.2
        if pushing:
            self.push_ring.set_data([mx], [my])
            self.push_ring.set_alpha(1 - since / 1.2)
        else:
            self.push_ring.set_data([], [])

        sec = s["sec"][i]
        for k, t in enumerate(self.sector_txt):
            on = k == sec
            t.set_color(DRIVER if on else MUTED)
            t.set_fontsize(25 if on else 22)

        who, line = LINES[sec]
        if (who, line) != self.last_key:
            self.last_key = (who, line)
            self.who.set_text(who)
            self.who.set_color(POLICE if who == "警察" else DRIVER)
            self.line.set_text(line)
        if who == "司机":                       # 离原点越近，司机越没底气
            conf = min(1.0, max(0.0, r / math.sqrt(MU)))
            self.line.set_fontsize(40 * (0.55 + 0.45 * conf))
            self.line.set_alpha(0.45 + 0.55 * conf)
            self.line.set_color(TEXT)
        else:                                   # 警察加大力度时字变大变蓝
            self.line.set_fontsize(48 if pushing else 40)
            self.line.set_alpha(1.0)
            self.line.set_color(POLICE if pushing else TEXT)
        self.stat.set_text(f"「算了嘛」× {s['sule'][i]}　　　出示驾照 × 0")

        self.bif_dot.set_data([MU], [r])

        i0 = int(math.ceil(SIM_START * fps))
        if i >= i0:
            j = max(i0, i - 20 * fps)
            self.p_line.set_data((np.arange(j, i + 1) - i) / fps, s["P"][j:i + 1])

        text, alpha = "", 0.0
        for a, b, c in self.captions:
            if a <= tv < b:
                text, alpha = c, ramp(tv, a, a + 0.35) * (1 - ramp(tv, b - 0.35, b))
        self.caption.set_text(text)
        self.caption.set_alpha(alpha)

        ta = 1 - ramp(tv, SIM_START, SIM_START + 0.8)
        ea = ramp(tv, self.end_in, self.end_in + 0.8)
        self.cover.set_alpha(max(ta, ea))
        for t in self.title_txt:
            t.set_alpha(ta)
            t.set_visible(ta > 0)
        for t in self.end_txt:
            t.set_alpha(ea)
            t.set_visible(ea > 0)


def make_gif(mp4, gif, start, length, width=720, fps=15):
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(mp4),
                    "-filter_complex", f"{vf},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)


def main():
    ap = argparse.ArgumentParser(description="渲染「算了嘛，警官」Hopf 极限环动画")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "media" / "suanlema_hopf.mp4")
    ap.add_argument("--still", type=float, help="只导出这一秒的单帧 PNG")
    ap.add_argument("--gif", action="store_true", help="渲染后额外输出 media/preview.gif")
    args = ap.parse_args()

    setup_fonts()
    sim = simulate(args.fps, t_max=60.0)
    scene = Scene(sim, args.fps)
    n = int(round(scene.duration * args.fps))
    k1, k2 = sim["kick_starts"]
    print(f"时长 {scene.duration:.1f}s，{n} 帧；两次施压在 {k1:.2f}s / {k2:.2f}s，"
          f"「算了嘛」共 {sim['sule'][n - 1]} 次")

    if args.still is not None:
        i = min(n - 1, int(round(args.still * args.fps)))
        scene.update(i)
        out = ROOT / "media" / f"still_{args.still:g}s.png"
        out.parent.mkdir(exist_ok=True)
        scene.fig.savefig(out, dpi=DPI, facecolor=BG)
        print(out)
        return

    if not shutil.which("ffmpeg"):
        sys.exit("需要 ffmpeg（brew install ffmpeg / apt install ffmpeg）")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow",
                                      "-movflags", "+faststart"])
    with writer.saving(scene.fig, str(args.out), dpi=DPI):
        for i in range(n):
            scene.update(i)
            writer.grab_frame(facecolor=BG)
            if i % (args.fps * 5) == 0:
                print(f"  {i / args.fps:5.1f}s / {scene.duration:.1f}s", flush=True)
    print(args.out)

    if args.gif:
        gif = args.out.parent / "preview.gif"
        make_gif(args.out, gif, start=k1 - 1.5, length=2 * PERIOD)
        print(gif)


if __name__ == "__main__":
    main()
