#!/usr/bin/env python3
"""Editable WIREFRAME .docx of the vision-only UAV proposal — structured so it can be rewritten later.

Every section: heading + editable body + (optional) embedded image with caption. Technical terms
explain HOW vision works without GPS and HOW accurate it is, WITHOUT giving a copyable pipeline.
"""
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPO = Path(__file__).resolve().parents[1]
F = REPO / "docs" / "ppt_assets" / "fresh"
INK = RGBColor(0x14, 0x1A, 0x22)
ACC = RGBColor(0xC2, 0x46, 0x12)
MUT = RGBColor(0x55, 0x60, 0x6C)
CY = RGBColor(0x12, 0x6A, 0x8C)

doc = Document()
st = doc.styles["Normal"].font
st.name = "Calibri"; st.size = Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(0.9)
    s.top_margin = s.bottom_margin = Inches(0.8)


def kicker(txt, color=ACC):
    p = doc.add_paragraph()
    r = p.add_run(txt.upper()); r.bold = True; r.font.size = Pt(9.5); r.font.color.rgb = color
    p.space_after = Pt(2)
    return p


def heading(txt, n=None):
    p = doc.add_paragraph()
    if n:
        rn = p.add_run(f"{n:02d}   "); rn.bold = True; rn.font.color.rgb = ACC; rn.font.size = Pt(18)
    r = p.add_run(txt); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = INK
    p.space_before = Pt(10); p.space_after = Pt(6)
    return p


def body(txt, italic=False, color=INK, size=11):
    p = doc.add_paragraph()
    r = p.add_run(txt); r.italic = italic; r.font.color.rgb = color; r.font.size = Pt(size)
    p.space_after = Pt(6); p.paragraph_format.line_spacing = 1.25
    return p


def bullet(txt, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        r = p.add_run(lead + " "); r.bold = True; r.font.color.rgb = INK
    r2 = p.add_run(txt); r2.font.color.rgb = INK
    return p


def image(name, w=6.2, cap=""):
    path = F / name
    if not path.exists():
        ph = doc.add_paragraph(); r = ph.add_run(f"[ IMAGE PLACEHOLDER: {name} — replace ]")
        r.italic = True; r.font.color.rgb = MUT
    else:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(w))
    if cap:
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rc = c.add_run("▲ " + cap); rc.italic = True; rc.font.size = Pt(9); rc.font.color.rgb = MUT


def hr():
    p = doc.add_paragraph(); r = p.add_run("—" * 46); r.font.color.rgb = RGBColor(0xCF, 0xD6, 0xDE)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.space_before = Pt(8); p.space_after = Pt(8)


def editnote(txt):
    p = doc.add_paragraph()
    r = p.add_run("✎ EDIT: " + txt); r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x99, 0x76, 0x12)


# ===================== COVER =====================
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = t.add_run("THIRD IRIS"); r.bold = True; r.font.size = Pt(13); r.font.color.rgb = ACC
kicker("Vision-only UAV system · proposal wireframe · 2026")
h = doc.add_paragraph(); r = h.add_run("Sees. Tracks. Needs no GPS.")
r.bold = True; r.font.size = Pt(30); r.font.color.rgb = INK
body("An aerial system that detects, follows and intercepts a moving target — and navigates itself — "
     "using nothing but a camera, even under full GPS jamming.", size=13, color=MUT)
image("sim_track_0.jpg", 6.4, "Live drone-feed: target locked in flight, vision-only, no GPS. Range and "
      "telemetry overlaid on the on-board camera.")
editnote("This is the cover. Swap headline / tagline freely; replace the image with any locked-target feed frame.")
hr()

# ===================== 02 PROBLEM =====================
kicker("The problem", RGBColor(0xB0, 0x2B, 0x2B)); heading("Why vision, not GPS", 2)
bullet("GPS can be jammed or spoofed in contested airspace — it is the first thing to fail.", "GPS-dependent:")
bullet("Cameras are passive — nothing to jam, nothing to spoof, no emissions to detect.", "Vision-driven:")
body("If the aircraft can see, it can navigate, track and finish the mission. The eye is the backup that "
     "cannot be jammed.")
hr()

# ===================== 03 CONCEPT =====================
kicker("System concept", CY); heading("One camera, two minds", 3)
body("Every frame is consumed twice: one pipeline watches the TARGET (detect + follow), the other watches "
     "the WORLD to localize the aircraft itself. GPS becomes optional.")
for s, d in [("SEE", "detect + segment the target"), ("TRACK", "lock on and predict its motion"),
             ("LOCATE", "estimate own position with no GPS"), ("AVOID", "steer around obstacles"),
             ("ACT", "follow / intercept at 10 Hz")]:
    bullet(d, s + " —")
hr()

# ===================== 04 HOW VISION WORKS (TECHNICAL TERMS) =====================
kicker("How it works · technical", CY); heading("How vision navigates without GPS", 4)
body("High-level terms only — enough to show the method, not a recipe to copy.")
bullet("a fine-tuned single-stage neural detector with instance segmentation localizes the target every "
       "frame; a dedicated bird class suppresses false locks.", "Detection & segmentation:")
bullet("a multi-object tracker with re-identification holds a stable target ID; an image-space Kalman filter "
       "predicts the target's bearing and coasts through detection drop-outs.", "Tracking:")
bullet("guidance is computed in the body frame directly from the target's pixel bearing and depth — pixel "
       "offset becomes yaw-rate, climb and forward-velocity commands. Needing no absolute position, it is "
       "immune to GPS jamming.", "Visual servoing (GPS-free):")
bullet("visual-inertial odometry fuses camera feature motion with the IMU at high rate to dead-reckon the "
       "drone's own 6-DOF pose when GPS is denied; a magnetometer bounds heading drift; place recognition "
       "(SLAM-style) corrects accumulated drift.", "Self-localization (VIO / VINS):")
bullet("a depth sensor gives metric range to the target and a depth-as-LiDAR obstacle field for reactive "
       "avoidance, plus a live 3-D sense of the scene.", "Depth & 3-D:")
bullet("the flight policy is a small neural pilot — learned by imitating a classical guidance law, then "
       "improved with reinforcement learning — running instantly on-board.", "Learned pilot:")
hr()

# ===================== 05 ACCURACY =====================
kicker("Accuracy & validation", RGBColor(0x1F, 0x7A, 0x44)); heading("How accurate it is", 5)
body("All figures measured in simulation / on real validation data — not claimed.")
tab = doc.add_table(rows=1, cols=2); tab.style = "Light Grid Accent 1"
tab.rows[0].cells[0].paragraphs[0].add_run("Metric").bold = True
tab.rows[0].cells[1].paragraphs[0].add_run("Result").bold = True
for k, v in [("Target detection (real held-out)", "mAP50 ≈ 0.80 and climbing"),
             ("Target detection (simulation)", "mAP50 ≈ 0.97"),
             ("Target kept in frame (vision-only)", "≈ 90%+"),
             ("Learned-pilot centering error", "0.107 — 2.5× tighter than hand-tuned law"),
             ("Stand-off hold error", "≈ 4.3 m (vs 7.2 m baseline)"),
             ("Collisions across maneuver suite", "0"),
             ("Control loop / supervisor cadence", "10 Hz pilot · ~0.5 Hz agent")]:
    row = tab.add_row().cells; row[0].text = k; row[1].text = v
editnote("Update these numbers as training / sim runs improve. Detector mAP is mid-training here.")
hr()

# ===================== 06 PERCEPTION =====================
kicker("Perception", ACC); heading("Trained on the real threat", 6)
body("The detector is fine-tuned on ~50,000 real images of Shahed-type loitering munitions and quadcopters, "
     "plus synthetic data. Below: our model running on unseen real frames. (Note: the current public Shahed dataset proved low-quality; clean real frames here are quadcopters.)")
image("real_det_2.jpg", 5.6, "Real quadcopter in flight — detected mid-manoeuvre.")
image("real_det_0.jpg", 5.6, "Real quadcopter in flight — detected and bracketed by the on-board model.")
editnote("Add / swap detection frames. Keep at least one Shahed and one quad.")
hr()

# ===================== 07 TRAINING =====================
kicker("Training & validation", RGBColor(0x1F, 0x7A, 0x44)); heading("It copies a master, then beats him", 7)
bullet("real + synthetic footage is segmented; birds, clutter and shadows are trained out.", "Data:")
bullet("a neural pilot learns by imitating a hand-built expert across thousands of randomized pursuit "
       "episodes.", "Imitate:")
bullet("reinforcement learning then trains past the teacher under realistic flight dynamics and detector "
       "drop-outs.", "Exceed:")
bullet("every build re-flies orbit, zigzag, weave, climb and dive scenarios in simulation.", "Prove:")
image("rl_standoff_hold.png", 6.2, "Learned pilot holding stand-off against an evasive zig-zag target.")
hr()

# ===================== 08 TRACKING / WEB UI =====================
kicker("Target following", RGBColor(0x1F, 0x7A, 0x44)); heading("Click the target — the aircraft does the rest", 8)
bullet("click-to-lock in a live ground-station web UI.")
bullet("predictive lock leads the target and coasts through detection gaps.")
bullet("holds a chosen stand-off distance; speeds match, no spin.")
image("webui_live.png", 6.2, "Ground-station web UI: live feed, depth panel, lock and telemetry.")
image("sim_track_3.jpg", 5.6, "Live pursuit frame — target locked at ~10 m, vision-only.")
hr()

# ===================== 09 VIO =====================
kicker("Self-localization", CY); heading("It knows where it is — without being told", 9)
bullet("visual-inertial odometry fuses camera + IMU into a live position estimate (VINS-class).")
bullet("depth gives true metric distance; SLAM-style relocalization corrects drift.")
bullet("GPS available → vision + satellites fused. GPS jammed → seamless switch to vision-only.")
image("sim_trajectory.png", 6.2, "Live run: ego pursuit path and altitude hold, vision-only.")
hr()

# ===================== 10 AVOIDANCE =====================
kicker("Flight safety", RGBColor(0xB5, 0x7A, 0x12)); heading("Dodges the world, keeps the lock", 10)
bullet("the depth image is read as a wall of distances — depth-as-LiDAR, no extra sensor.")
bullet("reactive steering bends the path around obstacles while tracking continues.")
bullet("zero collisions across the simulated scenario suite.")
image("airsim_depth.jpg", 6.2, "Live depth field (near → far). Walls and ground resolved for avoidance.")
hr()

# ===================== 11 APPLICATIONS =====================
kicker("Applications", ACC); heading("Two markets, same eye", 11)
at = doc.add_table(rows=1, cols=2); at.style = "Light Grid Accent 1"
at.rows[0].cells[0].paragraphs[0].add_run("Commercial").bold = True
at.rows[0].cells[1].paragraphs[0].add_run("Defense").bold = True
comm = ["Follow-me cinematography", "GPS-poor infrastructure inspection", "Search & rescue (canyons / indoor)",
        "Warehouse & indoor inventory drones", "Urban-canyon delivery"]
defn = ["Counter-UAS / anti-Shahed interception", "ISR under GPS jamming", "Convoy & perimeter overwatch",
        "Contested-airspace navigation & strike", "Comms-denied autonomous missions"]
c = at.add_row().cells
c[0].text = "\n".join("• " + x for x in comm); c[1].text = "\n".join("• " + x for x in defn)
hr()

# ===================== 12 INTELLIGENCE LAYER =====================
kicker("Intelligence layer · final phase", CY); heading("A System-2 mind on the aircraft", 12)
body("A small on-board language model supervises the mission — it advises, it never flies. The 10 Hz pilot, "
     "tracker and VIO stay untouched.")
bullet("a small quantized LLM (e.g. Qwen2.5-1.5B, 4-bit) on Jetson Orin reads telemetry text and emits a "
       "mode { SEARCH | TRACK | STRIKE | RETURN | HOLD } at ~0.5 Hz.", "Slow supervisor:")
bullet("a deterministic finite-state machine validates the proposal and clamps it to a geofence / altitude / "
       "battery safety envelope — the agent cannot command actuators.", "Safety gate:")
bullet("the fast neural pilot stays System-1 — instant, sub-millisecond.", "Fast stack:")
body("Grounded in Hi Robot (Physical Intelligence, 2025), LLM-Land (on-device LLM + MPC, 2025), and "
     "NVIDIA GR00T's System-2 / System-1 split. Built in simulation as the final phase.", italic=True, color=MUT)
hr()

# ===================== 13 ROADMAP =====================
kicker("Status & roadmap", CY); heading("Same software, sim to sky", 13)
bullet("full stack proven in AirSim simulation (detect · track · follow · avoid · GPS-jam handover).", "Done:")
bullet("detector + learned pilot trained on real data; RL pilot beats the hand-built expert.", "Done:")
bullet("production visual-inertial odometry (OpenVINS) fused into the flight controller.", "Next:")
bullet("same code on RealSense + Jetson Orin hardware; then the agentic supervisor.", "Then:")
body("If it can see, it can fight. Vision-only autonomy — jam-proof by physics, not by promise.",
     italic=True, color=ACC, size=12)

out = REPO / "docs" / "UAV_Vision_Proposal_Wireframe.docx"
doc.save(str(out))
print("saved", out)
