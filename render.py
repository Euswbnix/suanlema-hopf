#!/usr/bin/env python3
"""
算了嘛，警官 —— 一个 Hopf 极限环，以及它是怎么被打破的。

视频里的对话（原文顺序）：
    请出示驾照 → 算了嘛警官 → 你有没有驾照 → 有 → 有就请出示驾照 → 算了嘛警官
    → 不是，你有没有驾照 → 有 → 有就出示驾照 → 算了嘛警官
    → 有没有驾照 → 有 → 你出示驾照 → 算了嘛警官
    → 师傅，请你尊重法律，我作为一个交警我现在郑重的请你出示驾照，
      你不要觉得我在跟你开玩笑。 → （司机拿出驾照）

模型（Hopf 标准型，极坐标）：
    dr/dt = r (μ − r²)
    dθ/dt = ω = 2π / 5.05 s          （一轮「盘问—有—请出示—算了嘛」约 5 秒）

    μ = 司机嘴硬程度 − 警察强硬程度
    原点 r = 0 是「出示驾照」。μ > 0 时它是不稳定焦点（特征值 μ ± iω），所有轨迹被吸到半径 √μ 的
    极限环上，问多少遍都是同一个圈；郑重声明是一股强大的力量，把 μ 压到 −2，
    极限环消失（Hopf 分岔），原点变成稳定焦点，系统收敛：拿出驾照。

用法：
    python render.py                 # 渲染 media/suanlema_hopf.mp4
    python render.py --still 25      # 只导出第 25 秒的一帧，调版面用
    python render.py --gif           # 渲染完再把整段视频转成 README 预览 GIF
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 模型参数
PERIOD = 5.05                 # 一轮「盘问—有—请出示—算了嘛」的秒数
OMEGA = 2 * math.pi / PERIOD
MU0 = 0.8                     # 司机嘴硬程度 − 警察强硬程度：前面一直 > 0
MU_FORCED = -2.0              # 郑重声明之后：一股强大的力量把 μ 压到很负
R_VIEW = 1.75                 # 相平面显示范围 ±R_VIEW
R_FLOOR = 3e-3                # 数值噪声地板，免得点永远卡在原点
EPS = 0.01                    # 离原点多近，出示概率就被顶得多高
SULE_R = 0.02                 # 只排除数值噪声；再小声的「算了嘛」也算

SIM_START = 2.2               # 片头开始淡出时开始积分
START_U = 0.55                # 主角从「请出示」这一段开始（视频第一句就是请出示驾照）

# 每一段对应的出示概率台阶：盘问 / 有 / 请出示 / 算了嘛
LEVELS = [0.25, 0.55, 0.8, 0.0]

# 原视频前 14 句，四句一轮
SCRIPT = [
    ("警察", "请出示驾照。"),
    ("司机", "算了嘛，警官。"),
    ("警察", "你有没有驾照？"),
    ("司机", "有。"),
    ("警察", "有就请出示驾照。"),
    ("司机", "算了嘛，警官。"),
    ("警察", "不是，你有没有驾照？"),
    ("司机", "有。"),
    ("警察", "有就出示驾照。"),
    ("司机", "算了嘛，警官。"),
    ("警察", "有没有驾照？"),
    ("司机", "有。"),
    ("警察", "你出示驾照。"),
    ("司机", "算了嘛，警官。"),
]
# 最后的郑重声明，分句出现：(文字, 时长, 出示概率台阶)
STATEMENT = [
    ("师傅，请你尊重法律。", 1.8, 0.25),
    ("我作为一个交警，", 1.5, 0.5),
    ("我现在郑重的请你出示驾照，", 2.2, 0.75),
    ("你不要觉得我在跟你开玩笑。", 2.2, 0.9),
]
FORCE_AT = 2                  # 从第几句（0 起）开始施加那股力量：「郑重的请你出示驾照」
FORCE_LEN = 2.5               # μ 从 MU0 降到 MU_FORCED 用的秒数
SHOW_DELAY = 0.3              # 说完之后多久拿出驾照

SECTOR_LABELS = [("盘问", 0, 1.58), ("有", -1.58, 0), ("请出示", 0, -1.58), ("算了嘛", 1.58, 0)]

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
UNSTABLE = "#F09595"          # 不稳定的「出示驾照」
STABLE = "#5DCAA5"            # 稳定的「出示驾照」
GHOST = "#B4B2A9"             # 其他司机

W_PX, H_PX, DPI = 1920, 1080, 100


# ---------------------------------------------------------------- 数值部分
def field(p, mu):
    x, y = p[:, 0], p[:, 1]
    r2 = x * x + y * y
    return np.stack([mu * x - OMEGA * y - x * r2, OMEGA * x + mu * y - y * r2], axis=1)


def rk4(p, h, mu):
    k1 = field(p, mu)
    k2 = field(p + h / 2 * k1, mu)
    k3 = field(p + h / 2 * k2, mu)
    k4 = field(p + h * k3, mu)
    p = p + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r = np.hypot(p[:, 0], p[:, 1])
    small = r < R_FLOOR
    if small.any():
        p[small] *= (R_FLOOR / np.maximum(r[small], 1e-12))[:, None]
    return p


def phase(x, y):
    """把角度换成对话进度 u∈[0,1)：0 盘问 / .25 有 / .5 请出示 / .75 算了嘛（逆时针）"""
    th = math.atan2(y, x)
    return ((th - math.pi / 4) % (2 * math.pi)) / (2 * math.pi)


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def staircase(u):
    """循环里的出示概率：每句话上一个台阶，「算了嘛」直接清零"""
    s = min(3, int(u * 4))
    prev, cur = LEVELS[(s - 1) % 4], LEVELS[s]
    width = 0.02 if s == 3 else 0.04          # 清零比上台阶快
    return prev + (cur - prev) * smoothstep((u - s / 4) / width)


def simulate(fps, t_max, substeps=6):
    a0 = math.pi / 4 + 2 * math.pi * START_U
    starts = [(0.08, a0)]                                 # 主角：刚被拦下，离出示驾照只差一点
    starts += [(r, 0.4 + 2.1 * i) for i, r in enumerate([0.02, 0.35, 0.7, 1.05, 1.45, 1.65])]
    p = np.array([[r * math.cos(a), r * math.sin(a)] for r, a in starts])

    n = int(round(t_max * fps))
    out = {"P": np.zeros(n), "r": np.zeros(n), "mu": np.zeros(n), "pos": np.zeros((n, len(p), 2)),
           "sec": np.zeros(n, int), "line": np.zeros(n, int), "sule": np.zeros(n, int),
           "phrase": np.full(n, -1), "shown": np.zeros(n, bool)}
    sule_times = []
    starts_at = np.cumsum([0] + [d for _, d, _ in STATEMENT])   # 各句相对声明开始的时刻
    t_stmt = None                                                 # 声明开始 = 原文 14 句播完

    def mu_at(t):
        if t_stmt is None:
            return MU0
        u = smoothstep((t - t_stmt - starts_at[FORCE_AT]) / FORCE_LEN)
        return MU0 + (MU_FORCED - MU0) * u

    h = 1 / fps / substeps
    prev, count, line = -1, 0, 0
    P_hold = 0.0
    for i in range(n):
        t0 = (i - 1) / fps
        tv = i / fps
        if i > 0 and t0 >= SIM_START:
            for k in range(substeps):
                p = rk4(p, h, mu_at(t0 + (k + 0.5) * h))
        x, y = p[0]
        r = math.hypot(x, y)
        u = phase(x, y)
        s = min(3, int(u * 4))
        if prev != -1 and s != prev and t_stmt is None:
            line += 1
            if s == 3 and r > SULE_R:
                count += 1
                sule_times.append(tv)
            if line == len(SCRIPT):
                t_stmt = tv
        prev = s
        if t_stmt is None:
            assert s == (2 + line) % 4, "台词和相位对不上"
            w = math.exp(-r * r / EPS)
            P = (1 - w) * staircase(u) + w
            phrase, shown = -1, False
        else:
            d = tv - t_stmt
            phrase = int(np.searchsorted(starts_at, d, side="right")) - 1
            shown = d >= starts_at[-1] + SHOW_DELAY
            if shown:
                assert r < 0.08, "拿出驾照的时候还没收敛，调 FORCE_* 参数"
                lo, hi, t_step = STATEMENT[-1][2], 1.0, starts_at[-1] + SHOW_DELAY
            else:
                phrase = min(phrase, len(STATEMENT) - 1)
                lo = STATEMENT[phrase - 1][2] if phrase > 0 else P_hold
                hi, t_step = STATEMENT[phrase][2], starts_at[phrase]
            P = lo + (hi - lo) * smoothstep((d - t_step) / 0.3)
        if t_stmt is None:
            P_hold = P
        out["pos"][i] = p
        out["r"][i] = r
        out["mu"][i] = mu_at(tv)
        out["sec"][i] = s
        out["line"][i] = line
        out["sule"][i] = count
        out["phrase"][i] = phrase
        out["shown"][i] = shown
        out["P"][i] = P
    out["sule_times"] = sule_times
    out["t_stmt"] = t_stmt
    out["t_show"] = t_stmt + starts_at[-1] + SHOW_DELAY
    out["t_force"] = t_stmt + starts_at[FORCE_AT]
    out["t_mu0"] = next(t for t, m in zip(np.arange(n) / fps, out["mu"]) if m <= 0)
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


def mix(c1, c2, t):
    a, b = np.array(to_rgba(c1)), np.array(to_rgba(c2))
    return tuple(a + (b - a) * t)


class Scene:
    def __init__(self, sim, fps):
        self.sim, self.fps = sim, fps
        ts, tf, tw = sim["t_stmt"], sim["t_force"], sim["t_show"]
        self.end_in = tw + 5.0
        self.duration = self.end_in + 5.0
        self.captions = [
            (3.0, 9.0, "μ > 0：原点「出示驾照」不稳定，差一点点也会被甩出去"),
            (9.0, 14.0, "不管从哪出发，所有司机最后都落到同一个圈上：极限环 r = √μ"),
            (14.0, ts, "问多少遍都是同一个圈：有没有驾照 → 有 → 请出示 → 算了嘛警官"),
            (ts, tf, "警察不再重复同一个问题，开始郑重声明……"),
            (tf, tf + 2.2, f"一股强大的力量把 μ 往下压到 {MU_FORCED:g}：流场翻转，全部指向原点".replace("-", "−")),
            (tf + 2.2, tw, "μ 穿过 0 的那一刻极限环消失（Hopf 分岔），系统收敛"),
            (tw, self.end_in, "拿出来了。改变结局的不是再问一遍，而是 μ 变了号"),
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

        g = -1.625 + 0.25 * np.arange(14)
        gx, gy = np.meshgrid(g, g)
        self.grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self.quiver = ax.quiver(self.grid_pts[:, 0], self.grid_pts[:, 1],
                                np.zeros(len(self.grid_pts)), np.zeros(len(self.grid_pts)),
                                color=MUTED, alpha=0.38, angles="xy", scale_units="xy", scale=1 / 0.11,
                                pivot="mid", width=0.0022, headwidth=4, headlength=4.5, headaxislength=4, zorder=1)
        self.quiver_mu = None

        for sgn in (1, -1):
            ax.plot([-R_VIEW, R_VIEW], [-sgn * R_VIEW, sgn * R_VIEW], color=GRID,
                    lw=1.2, ls=(0, (3, 4)), zorder=1)
        self.circle_th = np.linspace(0, 2 * np.pi, 400)
        self.cycle_line, = ax.plot([], [], color=CYCLE, lw=2.2, ls=(0, (6, 5)), zorder=2)
        self.cycle_lab = ax.text(0, 0, "极限环", color=CYCLE, fontsize=15, ha="right", va="bottom", zorder=2)

        self.sector_txt = [ax.text(x, y, s, ha="center", va="center", fontsize=22, color=MUTED, zorder=3)
                           for s, x, y in SECTOR_LABELS]

        self.ghost_lc = LineCollection([], linewidths=1.3, zorder=3, capstyle="round")
        self.main_lc = LineCollection([], linewidths=3.0, zorder=4, capstyle="round")
        ax.add_collection(self.ghost_lc)
        ax.add_collection(self.main_lc)

        self.fp, = ax.plot([0], [0], marker="o", ms=15, mew=2.5, zorder=5)
        self.fp_lab = ax.text(0, 0.1, "", fontsize=16, ha="center", va="bottom", zorder=5)

        self.ghost_dots, = ax.plot([], [], "o", color=GHOST, ms=6, alpha=0.85, zorder=6)
        self.halo, = ax.plot([], [], "o", color=DRIVER, ms=28, alpha=0.18, mew=0, zorder=6)
        self.main_dot, = ax.plot([], [], "o", color=DRIVER, ms=13, zorder=7)

        ax.text(R_VIEW - 0.05, -R_VIEW + 0.05, "x：警察施压 →", color=MUTED, fontsize=15, ha="right", va="bottom")
        ax.text(-R_VIEW + 0.05, R_VIEW - 0.05, "↑ y：司机心虚", color=MUTED, fontsize=15, ha="left", va="top")

        # 右栏：μ 读数
        self.mu_txt = fig.text(0.545, 0.94, "", family="DejaVu Sans Mono", fontsize=34, color=TEXT, va="center")
        self.state_txt = fig.text(0.69, 0.94, "", fontsize=26, va="center")

        # 右栏：台词框
        box = FancyBboxPatch((0.545, 0.735), 0.42, 0.165, transform=fig.transFigure,
                             boxstyle="round,pad=0,rounding_size=0.012", mutation_aspect=W_PX / H_PX,
                             facecolor=PANEL, edgecolor=GRID, lw=1.2)
        fig.patches.append(box)
        self.who = fig.text(0.565, 0.872, "", fontsize=20, color=MUTED, va="center")
        self.line = fig.text(0.565, 0.815, "", fontsize=40, color=TEXT, va="center")
        self.stat = fig.text(0.565, 0.76, "", fontsize=18, color=SEC, va="center")

        # 右栏：分岔图
        fig.text(0.77, 0.695, "分岔图：极限环 ±√μ", fontsize=18, color=SEC, va="center")
        axb = fig.add_axes([0.775, 0.455, 0.185, 0.205])
        axb.set_facecolor(BG)
        axb.set_axis_off()
        axb.set_xlim(-2.2, 1.5)
        axb.set_ylim(-1.4, 1.4)
        axb.plot([0, 0], [-1.4, 1.4], color=GRID, lw=1)
        axb.plot([-2.2, 0], [0, 0], color=STABLE, lw=2.4)
        axb.plot([0, 1.5], [0, 0], color=UNSTABLE, lw=2.4, ls=(0, (4, 4)))
        m = np.linspace(0, 1.5, 200)
        for sgn in (1, -1):
            axb.plot(m, sgn * np.sqrt(m), color=CYCLE, lw=2.4)
        axb.text(-1.1, 0.12, "稳定焦点", color=STABLE, fontsize=13, ha="center", va="bottom")
        axb.text(1.0, -0.12, "不稳定焦点", color=UNSTABLE, fontsize=13, ha="center", va="top")
        axb.text(1.5, 0.45, "极限环", color=CYCLE, fontsize=14, ha="right", va="top")
        axb.text(0, -1.45, "Hopf 点", color=MUTED, fontsize=13, ha="center", va="top")
        self.bif_vline, = axb.plot([], [], color=SEC, lw=1, ls=(0, (2, 3)))
        self.bif_lab = axb.text(0, 1.4, "这位司机", color=DRIVER, fontsize=13, ha="center", va="bottom")
        self.bif_dot, = axb.plot([], [], "o", color=DRIVER, ms=10, zorder=5)

        # 右栏：马尔可夫链。节点位置和相平面四段一一对应，「出示驾照」在正中间 = 原点
        fig.text(0.545, 0.695, "状态转移（马尔可夫链）", fontsize=18, color=SEC, va="center")
        axm = fig.add_axes([0.545, 0.43, 0.205, 0.245])
        axm.set_facecolor(BG)
        axm.set_axis_off()
        yl = 1.45
        xl = yl * (0.205 * W_PX) / (0.245 * H_PX)      # 让 x、y 单位等长，圆是圆的
        axm.set_xlim(-xl, xl)
        axm.set_ylim(-yl, yl)
        nodes = [(0, 1.1), (-1.4, 0), (0, -1.1), (1.4, 0)]   # 盘问 / 有 / 请出示 / 算了嘛
        owner = [POLICE, DRIVER, POLICE, DRIVER]
        r_node, r_mid = 0.32, 0.38
        self.mk_nodes = []
        for (x, y), (lab, _, _), col in zip(nodes, SECTOR_LABELS, owner):
            c = Circle((x, y), r_node, facecolor=BG, edgecolor=col, lw=2, zorder=3)
            axm.add_patch(c)
            self.mk_nodes.append((c, axm.text(x, y, lab, ha="center", va="center", fontsize=12,
                                              color=TEXT, zorder=4), col))
        self.mk_mid = Circle((0, 0), r_mid, facecolor=BG, edgecolor=UNSTABLE, lw=2, ls=(0, (3, 2)), zorder=3)
        axm.add_patch(self.mk_mid)
        self.mk_mid_t = axm.text(0, 0, "出示驾照", ha="center", va="center", fontsize=11, color=UNSTABLE, zorder=4)

        def ends(a, b, ra, rb, gap=0.04):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            ux, uy = (b[0] - a[0]) / d, (b[1] - a[1]) / d
            return ((a[0] + ux * (ra + gap), a[1] + uy * (ra + gap)),
                    (b[0] - ux * (rb + gap), b[1] - uy * (rb + gap)))

        self.mk_edges = []
        for k in range(4):
            a, b = nodes[k], nodes[(k + 1) % 4]
            arr = FancyArrowPatch(*ends(a, b, r_node, r_node), arrowstyle="-|>", mutation_scale=14,
                                  color=SEC, lw=1.6, shrinkA=0, shrinkB=0, zorder=2)
            axm.add_patch(arr)
            mx_, my_ = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            nrm = math.hypot(mx_, my_)
            lab = axm.text(mx_ + 0.3 * mx_ / nrm, my_ + 0.3 * my_ / nrm, "", ha="center", va="center",
                           fontsize=12, color=SEC)
            self.mk_edges.append((arr, lab))
        self.mk_out = FancyArrowPatch(*ends(nodes[2], (0, 0), r_node, r_mid), arrowstyle="-|>", mutation_scale=14,
                                      color=UNSTABLE, lw=1.6, ls=(0, (3, 2)), shrinkA=0, shrinkB=0, zorder=2)
        axm.add_patch(self.mk_out)
        self.mk_out_lab = axm.text(0.12, -0.47, "", ha="left", va="center", fontsize=10, color=UNSTABLE)
        self.mk_info0 = axm.text(xl - 0.05, yl - 0.05, "周期 4", ha="right", va="top", fontsize=11, color=MUTED)
        self.mk_info1 = axm.text(-xl + 0.05, yl - 0.05, "", ha="left", va="top", fontsize=11, color=MUTED)
        self.mk_info2 = axm.text(xl - 0.05, -yl + 0.05, "", ha="right", va="bottom", fontsize=11, color=MUTED)

        # 右栏：出示概率 P(t)
        fig.text(0.545, 0.395, "驾照出示概率 P(t)：每句话上一个台阶，「算了嘛」清零",
                 fontsize=18, color=SEC, va="center")
        axp = fig.add_axes([0.58, 0.11, 0.375, 0.25])
        axp.set_facecolor(BG)
        axp.set_axis_off()
        axp.set_xlim(-20, 0)
        axp.set_ylim(-0.05, 1.3)
        for yv in (0, 0.5):
            axp.plot([-20, 0], [yv, yv], color=GRID, lw=1)
        self.p_goal, = axp.plot([-20, 0], [1, 1], color=UNSTABLE, lw=1.4, ls=(0, (5, 4)))
        self.p_goal_lab = axp.text(-19.8, 1.05, "出示驾照（从未到达）", color=UNSTABLE, fontsize=13,
                                   ha="left", va="bottom")
        for yv, lab in ((1, "1.0"), (0.5, "0.5"), (0, "0.0")):
            axp.text(-20.4, yv, lab, color=MUTED, fontsize=13, ha="right", va="center")
        axp.text(-20, -0.1, "20 秒前", color=MUTED, fontsize=13, ha="left", va="top")
        axp.text(0, -0.1, "现在", color=MUTED, fontsize=13, ha="right", va="top")
        self.p_line, = axp.plot([], [], color=POLICE, lw=2.2)
        self.p_head, = axp.plot([], [], "o", color=POLICE, ms=8)
        self.sule_marks = [(axp.plot([], [], "v", color=DRIVER, ms=9)[0],
                            axp.text(0, 1.07, "算了", color=DRIVER, fontsize=12, ha="center", va="bottom"))
                           for _ in range(6)]

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
            ov.text(0.5, 0.48, "一个 Hopf 极限环，以及它是怎么被打破的", fontsize=32, color=SEC, **kw),
            ov.text(0.5, 0.38, r"$\dot r = r(\mu - r^2),\qquad \dot\theta = \omega$", fontsize=30, color=MUTED, **kw),
        ]
        self.end_txt = [
            ov.text(0.5, 0.68, "问一百遍都是同一个圈，郑重一次就收敛了", fontsize=44, color=TEXT, **kw),
            ov.text(0.5, 0.56, "μ > 0：有没有驾照 → 有 → 请出示 → 算了嘛警官 → …（极限环）",
                    fontsize=30, color=CYCLE, **kw),
            ov.text(0.5, 0.47, "μ < 0：一股强大的力量让系统收敛 → 拿出驾照（稳定焦点）",
                    fontsize=30, color=STABLE, **kw),
            ov.text(0.5, 0.34, "μ = 司机嘴硬程度 − 警察强硬程度　·　μ 穿过 0 = Hopf 分岔", fontsize=26, color=MUTED, **kw),
            ov.text(0.5, 0.10, rf"$\dot r = r(\mu - r^2),\quad \dot\theta = 2\pi / 5.05\,\mathrm{{s}},"
                               rf"\quad \mu: {MU0:g} \to {MU_FORCED:g}$", fontsize=22, color=MUTED, **kw),
        ]
        self.last_line = None

    def _set_mu(self, mu):
        """μ 变了才重画流场、极限环、平衡点"""
        if self.quiver_mu is not None and abs(mu - self.quiver_mu) < 1e-4:
            return
        self.quiver_mu = mu
        v = field(self.grid_pts, mu)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.quiver.set_UVC(v[:, 0], v[:, 1])
        k = min(1.0, max(0.0, -mu / 1.5))               # 力量越大，流场越绿越亮
        self.quiver.set_color(mix(MUTED, STABLE, k))
        self.quiver.set_alpha(0.38 + 0.3 * k)

        if mu > 0:
            rc = math.sqrt(mu)
            self.cycle_line.set_data(rc * np.cos(self.circle_th), rc * np.sin(self.circle_th))
            self.cycle_lab.set_position((rc * math.cos(2.45) - 0.05, rc * math.sin(2.45) + 0.1))
            self.cycle_lab.set_visible(rc > 0.45)
        else:
            self.cycle_line.set_data([], [])
            self.cycle_lab.set_visible(False)

        if mu > 0.02:
            c, fill, lab, state, sc = UNSTABLE, "none", "出示驾照（不稳定焦点）", "极限环：无限循环", CYCLE
        elif mu < -0.02:
            c, fill, lab, state, sc = STABLE, STABLE, "出示驾照（稳定焦点）", "稳定焦点：系统收敛", STABLE
        else:
            c, fill, lab, state, sc = MUTED, "none", "出示驾照（临界）", "Hopf 分岔点", SEC
        self.fp.set_markeredgecolor(c)
        self.fp.set_markerfacecolor(fill)
        self.fp_lab.set_text(lab)
        self.fp_lab.set_color(c)
        self.mu_txt.set_text(f"μ = {mu:+.2f}")
        self.state_txt.set_text(state)
        self.state_txt.set_color(sc)
        self.bif_vline.set_data([mu, mu], [-1.4, 1.4])
        self.bif_lab.set_x(mu)

    # ---- 更新第 i 帧
    def update(self, i):
        s, fps = self.sim, self.fps
        tv = i / fps
        pos = s["pos"]
        mx, my = pos[i, 0]
        r, mu = s["r"][i], s["mu"][i]
        self._set_mu(mu)

        k_main, k_ghost = int(3.2 * fps), int(2.0 * fps)
        j = max(0, i - k_main)
        pts = pos[j:i + 1, 0]
        segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.zeros((0, 2, 2))
        col = np.tile(to_rgba(DRIVER), (len(segs), 1))
        if len(segs):
            col[:, 3] = np.linspace(0, 0.95, len(segs))
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

        phrase, shown = s["phrase"][i], s["shown"][i]
        looping = phrase < 0
        sec = s["sec"][i]
        for k, t in enumerate(self.sector_txt):
            on = looping and k == sec
            t.set_color(DRIVER if on else MUTED)
            t.set_fontsize(25 if on else 22)

        if shown:
            who, line = "司机", "（拿出驾照）"
        elif looping:
            who, line = SCRIPT[s["line"][i]]
        else:
            who, line = "警察（郑重）", STATEMENT[phrase][0]
        if (who, line) != self.last_line:
            self.last_line = (who, line)
            self.who.set_text(who)
            self.who.set_color(DRIVER if who == "司机" else POLICE)
            self.line.set_text(line)
        if shown:
            self.line.set_fontsize(40)
            self.line.set_alpha(1.0)
            self.line.set_color(STABLE)
        elif looping and who == "司机":         # 离原点越近，司机越没底气
            conf = min(1.0, max(0.0, r / math.sqrt(MU0)))
            self.line.set_fontsize(40 * (0.55 + 0.45 * conf))
            self.line.set_alpha(0.45 + 0.55 * conf)
            self.line.set_color(TEXT)
        else:
            self.line.set_fontsize(40 if looping else 36)
            self.line.set_alpha(1.0)
            self.line.set_color(TEXT if looping else POLICE)
        n = s["sule"][i]
        self.stat.set_text(f"已完成 {max(0, n - 1)} 个周期 · 「算了嘛」× {n} · 驾照出示 {int(shown)} 次")
        self.stat.set_color(STABLE if shown else SEC)

        self.bif_dot.set_data([mu], [r])

        # 马尔可夫链：请出示之后去哪，由 μ 决定；μ < 0 时出示驾照变成吸收态
        p_out = smoothstep(-mu / 0.6)
        for k, (arr, lab) in enumerate(self.mk_edges):
            p = 1 - p_out if k == 2 else 1.0
            lab.set_text(rf"$p={p:.2f}$")
            arr.set_alpha(0.25 + 0.75 * p)
            lab.set_alpha(0.4 + 0.6 * p)
        absorbing = p_out >= 0.5
        oc = STABLE if absorbing else UNSTABLE
        ols = "-" if absorbing else (0, (3, 2))
        self.mk_out.set_color(oc)
        self.mk_out.set_linestyle(ols)
        self.mk_out_lab.set_text(rf"$p={p_out:.2f}$")
        self.mk_out_lab.set_color(oc)
        self.mk_mid.set_edgecolor(oc)
        self.mk_mid.set_linestyle(ols)
        self.mk_mid.set_facecolor(STABLE if shown else BG)
        self.mk_mid_t.set_color(BG if shown else oc)
        active = sec if looping else (None if shown else 2)
        for k, (c, t, col) in enumerate(self.mk_nodes):
            on = k == active
            c.set_facecolor(col if on else BG)
            t.set_color(BG if on else TEXT)
        if absorbing:
            self.mk_info0.set_visible(False)
            self.mk_info1.set_text("吸收概率 = 1")
            self.mk_info2.set_text("出示驾照：吸收态")
            self.mk_info2.set_color(STABLE)
        else:
            self.mk_info0.set_visible(True)
            self.mk_info1.set_text("平稳分布各 1/4")
            self.mk_info2.set_text("出示驾照：不可达")
            self.mk_info2.set_color(MUTED)

        i0 = int(math.ceil(SIM_START * fps))
        if i >= i0:
            j = max(i0, i - 20 * fps)
            self.p_line.set_data((np.arange(j, i + 1) - i) / fps, s["P"][j:i + 1])
            self.p_head.set_data([0], [s["P"][i]])
        goal = STABLE if shown else UNSTABLE
        self.p_goal.set_color(goal)
        self.p_goal_lab.set_color(goal)
        self.p_goal_lab.set_text("出示驾照（到达）" if shown else "出示驾照（从未到达）")
        events = [te - tv for te in s["sule_times"] if tv - 20 < te <= tv]
        for k, (mark, lab) in enumerate(self.sule_marks):
            if k < len(events):
                x = events[k]
                a = ramp(x, -14.2, -13.2)       # 滑到左边「出示驾照（从未到达）」字样之前淡出
                mark.set_data([x], [1.0])
                mark.set_alpha(a)
                lab.set_x(x)
                lab.set_alpha(a)
                lab.set_visible(a > 0)
            else:
                mark.set_data([], [])
                lab.set_visible(False)

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


def make_gif(mp4, gif, start, length, width=800, fps=12):
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(mp4),
                    "-filter_complex", f"{vf},split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(gif)], check=True)


def main():
    ap = argparse.ArgumentParser(description="渲染「算了嘛，警官」Hopf 极限环动画")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "media" / "suanlema_hopf.mp4")
    ap.add_argument("--still", type=float, help="只导出这一秒的单帧 PNG")
    ap.add_argument("--gif", action="store_true", help="渲染后额外输出整段视频的 media/preview.gif")
    args = ap.parse_args()

    setup_fonts()
    sim = simulate(args.fps, t_max=60.0)
    scene = Scene(sim, args.fps)
    n = int(round(scene.duration * args.fps))
    print(f"时长 {scene.duration:.1f}s，{n} 帧；原文 14 句播完 / 郑重声明开始 {sim['t_stmt']:.2f}s；"
          f"施力 {sim['t_force']:.2f}s；μ 过 0 {sim['t_mu0']:.2f}s；拿出驾照 {sim['t_show']:.2f}s；"
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
        make_gif(args.out, gif, start=0, length=scene.duration)
        print(gif)


if __name__ == "__main__":
    main()
