"""Build docs/meetings/2026-11-17_advisor_structured-closure.pptx (not in git) from the figures of
advisor_2026_11_17_figs.py, plus layout previews (docs/meetings/figures_2026-11-17/preview_slideN.png) drawn
from the same coordinates (no LibreOffice on HiPerGator).  Needs python-pptx (PYTHONPATH to a --target install).
"""
import os, textwrap
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
FIG = os.path.join(ROOT, "docs/meetings/figures_2026-11-17")
OUT = os.path.join(ROOT, "docs/meetings/2026-11-17_advisor_structured-closure.pptx")
NAVY, TEAL, CORAL, INK, MUTED, TINT, WHITE, ICE = "0B3C5D", "1D7874", "E4572E", "1F2D35", "5C6B73", "EAF2F4", "FFFFFF", "CFE3EA"
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
PREVIEW = []          # per slide: list of elements for the layout preview


def new_slide(dark):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = rgb(NAVY if dark else WHITE)
    PREVIEW.append(dict(dark=dark, items=[]))
    return s


def text(s, x, y, w, h, runs, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None, para_after=6):
    """runs: str, or list of paragraphs; a paragraph is a str or list of (text, dict(bold, color, size, italic))."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name: tb.name = name
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04); tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = runs if isinstance(runs, list) else [runs]
    flat = []
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(para_after)
        for t, o in ([(p, {})] if isinstance(p, str) else p):
            r = para.add_run(); r.text = t; f = r.font
            f.name = o.get("font", font); f.size = Pt(o.get("size", size)); f.bold = o.get("bold", bold)
            f.italic = o.get("italic", False); f.color.rgb = rgb(o.get("color", color))
        flat.append(("".join(t for t, _ in ([(p, {})] if isinstance(p, str) else p)), (p[0][1].get("size", size) if not isinstance(p, str) else size)))
    PREVIEW[-1]["items"].append(dict(kind="text", x=x, y=y, w=w, h=h, paras=flat, color=color, align=align))
    return tb


def card(s, x, y, w, h, fill=TINT, name=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06; sh.fill.solid(); sh.fill.fore_color.rgb = rgb(fill); sh.line.fill.background(); sh.shadow.inherit = False
    if name: sh.name = name
    PREVIEW[-1]["items"].append(dict(kind="card", x=x, y=y, w=w, h=h, fill=fill))
    return sh


def pic(s, fn, x, y, w=None, h=None, name=None):
    im = Image.open(os.path.join(FIG, fn)); ar = im.height / im.width
    if w is not None and h is not None:          # fit inside the box, keep aspect, centre
        if w * ar > h: w2, h2 = h / ar, h
        else: w2, h2 = w, w * ar
        x, y, w, h = x + (w - w2) / 2, y + (h - h2) / 2, w2, h2
    elif w is not None: h = w * ar
    else: w = h / ar
    p = s.shapes.add_picture(os.path.join(FIG, fn), Inches(x), Inches(y), Inches(w), Inches(h))
    if name: p.name = name
    PREVIEW[-1]["items"].append(dict(kind="pic", x=x, y=y, w=w, h=h, fn=fn))
    return p


def title(s, t, dark, sub=None):
    text(s, 0.6, 0.38, W - 1.2, 0.8, [[(t, dict(bold=True))]], size=34, color=WHITE if dark else NAVY, font=HEAD, name="Title")
    if sub:
        text(s, 0.6, 1.12, W - 1.2, 0.45, sub, size=16, color=ICE if dark else MUTED, name="Subtitle")


def stat(s, x, y, w, h, big, label, color=TEAL):
    card(s, x, y, w, h)
    text(s, x + 0.2, y + 0.1, w - 0.4, 0.6, [[(big, dict(bold=True, font=HEAD))]], size=30, color=color)
    text(s, x + 0.2, y + 0.72, w - 0.4, h - 0.8, label, size=14, color=INK)


# ---------------------------------------------------------------- slide 1
s = new_slide(True)
title(s, "Why the current ML closures drift, and the plan", True,
      "Two days: an exact linear reference, a closure right by construction, a first closed-loop test")
card(s, 0.6, 1.8, 7.3, 5.1, fill=WHITE); pic(s, "f1_ml_drift.png", 0.75, 1.9, w=7.0, h=4.9)
x0 = 8.25
text(s, x0, 1.8, 4.5, 0.4, [[("Phase A diagnosis: two defects", dict(bold=True))]], size=18, color=WHITE)
card(s, x0, 2.3, 4.5, 1.45, fill="134B70")
text(s, x0 + 0.2, 2.38, 4.1, 1.3, [[("Not zero on a flat film", dict(bold=True, color="F7B2A1", size=17))],
                                   "Flat states were masked out of training, so the closure shifts the base state and wave speed"], size=14, color=WHITE)
card(s, x0, 3.95, 4.5, 1.45, fill="134B70")
text(s, x0 + 0.2, 4.03, 4.1, 1.3, [[("Wrong linear growth", dict(bold=True, color="F7B2A1", size=17))],
                                   "20–28% off Orr–Sommerfeld; the separate-model MLP doubles the growth rate"], size=14, color=WHITE)
text(s, x0, 5.6, 4.5, 1.3, [[("Plan: ", dict(bold=True, color="8FD3C9")), ("a closure that preserves the flat film and stays hyperbolic by construction, "
                                                                           "trained with exact linear (OS) targets", {})]], size=15, color=WHITE)
s.notes_slide.notes_text_frame.text = (
    "Oracle case, ROM started from the DNS state. The current sum MLP restores the wave height for about 10 time units "
    "(as good as the exact closure), then drifts and by t = 100 is worse than Lavalle. Phase A traced two defects: the MLPs "
    "do not vanish on a flat film (flat_mask removed |h-1|<0.05 from training) and their linear growth is wrong. "
    "The structured-closure plan fixes both by construction and with Orr-Sommerfeld targets.")

# ---------------------------------------------------------------- slide 2
s = new_slide(False)
title(s, "An exact linear reference, confirmed by the DNS", False,
      "Orr–Sommerfeld (OS) verified against the literature; comb-forced DNS tests all frequencies at once")
pic(s, "f2_comb_os.png", 0.6, 1.75, w=12.1)
y = 5.75; ww = 3.85
stat(s, 0.6, y, ww, 1.4, "2–3%", "DNS growth vs OS, Re 80–200 (median, whole forced band)", TEAL)
stat(s, 0.6 + ww + 0.3, y, ww, 1.4, "1–3%", "extracted ΔMₓ, ΔT_w per frequency vs exact OS values", TEAL)
stat(s, 0.6 + 2 * (ww + 0.3), y, ww, 1.4, "24–37%", "Lavalle's growth deficit at Re 80–200", CORAL)
s.notes_slide.notes_text_frame.text = (
    "Comb inlet: 21-31 frequencies at 3e-4 each, period exactly 100, so each is separated exactly by projection. "
    "The DNS follows OS across the band; the phase speed is 1-1.5% below OS at every frequency (mesh). The extracted "
    "closure terms match the exact OS values, which validates the extraction pipeline where the truth is known. "
    "Re 40 Ca 0.04 agrees in an upstream window; at Re 20 Lavalle is already within 3%.")

# ---------------------------------------------------------------- slide 3
s = new_slide(False)
title(s, "Four coefficients fix Lavalle's linear error", False,
      "Lavalle's error is set by δ_lin = k_max Re; a closure whose linear part matches OS removes it")
pic(s, "f3_linear_error.png", 0.6, 1.75, w=7.4)
x0 = 8.4
stat(s, x0, 1.85, 4.3, 1.45, "85 / 85", "cases within 10% of OS growth with (a, b, c, d) over the forced band", TEAL)
card(s, x0, 3.55, 4.3, 2.6)
text(s, x0 + 0.2, 3.65, 3.9, 2.45, [
    [("Linear part of the closure", dict(bold=True, color=NAVY, size=16))],
    "S = a(ξ − 3η) + b ηₓ + c ξₓ + d ηₓₓ",
    "The h_xx term d is needed beyond δ ≈ 5",
    [("Used as a training constraint: ", dict(bold=True)), ("the OS dispersion is imposed at all 85 parameter points, no DNS needed", {})]],
    size=14, color=INK, para_after=8)
s.notes_slide.notes_text_frame.text = (
    "Lavalle is 7-18% too unstable at small delta and 25-50% too stable at large delta. Four coefficients, with the "
    "long-wave limit imposed exactly, match OS within 10% in every case. In training we impose the OS dispersion "
    "relation itself (20 frequencies per point) because the flux and relaxation parts see different derivative filters in the ROM.")

# ---------------------------------------------------------------- slide 4
s = new_slide(False)
title(s, "Structured closure works as designed in the ROM", False,
      "Lavalle + learned functions of h in a fixed form, trained with the OS constraint")
pic(s, "f4a_schematic.png", 0.6, 1.65, w=5.9)
card(s, 0.6, 5.1, 6.1, 1.9)
text(s, 0.8, 5.2, 5.8, 1.75, [
    [("Exact flat film: ", dict(bold=True, color=TEAL)), ("S = 0 for any weights", {})],
    [("Always hyperbolic: ", dict(bold=True, color=TEAL)), ("Φ ≥ 1 and increasing in h", {})],
    [("Starts at Lavalle: ", dict(bold=True, color=TEAL)), ("zero correction at initialisation", {})],
    [("ROM checks: ", dict(bold=True, color=TEAL)), ("flat film stays flat, no stiffness", {})]], size=14, color=INK, para_after=4)
pic(s, "f4b_rom_comb.png", 7.3, 1.65, h=3.9)
stat(s, 7.1, 5.6, 2.75, 1.4, "~10×", "smaller growth error with OS constraint", TEAL)
stat(s, 10.05, 5.6, 2.75, 1.4, "0.26", "held-out NRMSE, unchanged", NAVY)
s.notes_slide.notes_text_frame.text = (
    "Without the OS constraint the data leave linear growth free: errors up to 10x the OS peak and the comb run goes nonlinear. "
    "With it (lambda_lin = 10) the ROM growth error at Re 120-200 is 8-11% (Lavalle 27-36%) at no cost in held-out accuracy. "
    "The ROM reproduces the model's own analytic dispersion to 3-4%. The uniform-film test found a bug: the truncated Gaussian "
    "second derivative is not zero on constants (fixed with a zero-sum operator).")

# ---------------------------------------------------------------- slide 5
s = new_slide(True)
title(s, "Closed loop: the energy balance is the hard part", True,
      "Oracle case from the DNS state: good for ~5 time units, then crest growth and phase drift")
card(s, 0.6, 1.8, 5.35, 3.75, fill=WHITE); pic(s, "f5a_aposteriori.png", 0.7, 1.88, w=5.15, h=3.6)
card(s, 6.15, 1.8, 3.8, 3.75, fill=WHITE); pic(s, "f5b_power.png", 6.23, 1.88, w=3.64, h=3.6)
text(s, 10.2, 1.8, 2.55, 3.8, [
    [("Diagnosis", dict(bold=True, color="8FD3C9", size=17))],
    "Pointwise fit is fine (NRMSE ≈ 0.12 for all closures)",
    "Net energy input is ~0.1% of the front/back terms, and all learned closures get its sign wrong",
    "Matching it a priori alone does not fix the drift"], size=13.5, color=WHITE, para_after=7)
text(s, 0.6, 5.8, 1.6, 0.4, [[("Next", dict(bold=True))]], size=18, color=WHITE)
steps = [("1", "Seed spread", "how much is noise?"), ("2", "Flat-film-exact MLP", "running now"),
         ("3", "Richer forms", "V1.5 / V2: (h, q/h³)"), ("4", "Broadband held-out", "gate GD")]
for k, (n, a, b) in enumerate(steps):
    xk = 0.6 + k * 3.08
    card(s, xk, 6.2, 2.9, 0.85, fill="134B70")
    text(s, xk + 0.15, 6.25, 2.65, 0.8, [[(n + "  ", dict(bold=True, color="F7B2A1", size=16)), (a, dict(bold=True, size=14))],
                                         [(b, dict(size=12.5, color=ICE))]], size=14, color=WHITE, para_after=0)
s.notes_slide.notes_text_frame.text = (
    "Structured V1 (sigma 0.25) matches the current MLP for ~5 time units, then drifts somewhat faster (h error 0.60 vs 0.47 at t = 50). "
    "Split runs: the flux part alone drives crest growth, the relaxation part alone damps like Lavalle. A priori, the closure's net power "
    "into the waves <(-S) q'> is a tiny residual of large front (-0.32) and back (+0.19) contributions; the DNS value is -3.4e-4, every "
    "learned closure gives +1e-3 to +2.2e-3. A power penalty fixes this a priori but not a posteriori (phase drift). "
    "Next: seed spread, the flat-film-exact MLP (running), V1.5/V2, then the broadband held-out cases.")

prs.save(OUT)
print("deck ->", OUT)

# ---------------------------------------------------------------- layout previews
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
for i, sl in enumerate(PREVIEW, 1):
    fig = plt.figure(figsize=(W, H), dpi=80); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), W, H, color="#" + (NAVY if sl["dark"] else WHITE), zorder=0))
    for it in sl["items"]:
        if it["kind"] == "card":
            ax.add_patch(FancyBboxPatch((it["x"], it["y"]), it["w"], it["h"], boxstyle="round,pad=0,rounding_size=0.1", color="#" + it["fill"], zorder=2))
        elif it["kind"] == "pic":
            ax.imshow(plt.imread(os.path.join(FIG, it["fn"])), extent=(it["x"], it["x"] + it["w"], it["y"] + it["h"], it["y"]), aspect="auto", zorder=3)
        else:
            ax.add_patch(plt.Rectangle((it["x"], it["y"]), it["w"], it["h"], fill=False, ec="#999999", lw=0.6, ls=":"))
            yy = it["y"] + 0.05
            for t, sz in it["paras"]:
                cpl = max(8, int(it["w"] * 72 / (0.47 * sz)))   # DejaVu is ~10% wider than Calibri
                for line in textwrap.wrap(t, cpl) or [""]:
                    ax.text(it["x"] + 0.05, yy, line, fontsize=sz * 0.92, va="top", color="#" + it["color"], family="DejaVu Sans", zorder=5)
                    yy += sz * 1.2 / 72
                yy += 6 / 72
            if yy > it["y"] + it["h"] + 0.05:
                ax.add_patch(plt.Rectangle((it["x"], it["y"]), it["w"], it["h"], fill=False, ec="red", lw=2, zorder=6))
    fig.savefig(os.path.join(FIG, f"preview_slide{i}.png")); plt.close(fig)
print("previews ->", FIG)
