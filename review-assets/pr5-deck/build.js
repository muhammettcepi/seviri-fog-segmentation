// Walkthrough deck of PR #5 (station validation): what runs, in what order, with what data.
const nodePath = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/root/.claude/skills/synced/8028e69d-e62b-4d42-8b97-6d5b5f0ac3e0_86aafce7-baed-4f82-991f-dfebdad36547/pptx";
const { applyTheme } = require(nodePath.join(SKILL, "scripts/apply_theme.js"));

const OUT = process.argv[2] || "pr5-station-validation.pptx";

const THEME = {
  name: "Night Fog",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A41", lt1: "FFFFFF", dk2: "324A5F", lt2: "EEF2F5",
    accent1: "F2A541", accent2: "3E8E7E", accent3: "C8553D",
    accent4: "7A8FA6", accent5: "0C1821", accent6: "C9D6DF",
    hlink: "3E8E7E", folHlink: "7A8FA6",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Station validation of the fog labels (PR #5)";
pres.author = "seviri-fog-segmentation";

const C = pres.SchemeColor;
const TEXT = C.text1, SLATE = C.text2, WHITE = C.background1, FOG = C.background2;
const AMBER = C.accent1, TEAL = C.accent2, RUST = C.accent3, MIST = C.accent4, NIGHT = C.accent5, PALE = C.accent6;
const MONO = "Courier New";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: NIGHT },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 7.6, h: 1.9, fontSize: 40, bold: true, color: WHITE, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.1, w: 7.6, h: 1.2, fontSize: 18, color: PALE, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: NIGHT },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.8, y: 2.3, w: 8, h: 0.6, fontSize: 18, bold: true, color: AMBER, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.95, w: 11.5, h: 1.1, fontSize: 40, bold: true, color: WHITE, valign: "top", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.25, w: 10, h: 1.3, fontSize: 18, color: PALE, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: TEXT, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "PR #5 · station validation", options: { x: 0.6, y: 7.05, w: 5, h: 0.3, fontSize: 10, color: SLATE, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: SLATE, align: "right" },
});

// ---------- drawing helpers ----------
let objectCount = 0;
const nm = (base) => `${base}-${++objectCount}`;

function textOpts(o) {
  const out = {};
  for (const [k, v] of Object.entries(o)) if (v !== undefined) out[k] = v;
  return out;
}

// Rounded box with text. `text` is a string or an array of runs.
function box(s, x, y, w, h, text, o = {}) {
  const fill = o.fill || FOG;
  s.addText(text, textOpts({
    shape: o.shape || pres.shapes.ROUNDED_RECTANGLE,
    rectRadius: (o.shape && o.shape !== pres.shapes.ROUNDED_RECTANGLE) ? undefined : (o.r ?? 0.08),
    x, y, w, h,
    fill: { color: fill },
    line: { color: o.line || fill, width: o.lw || 1, dashType: o.dash },
    fontSize: o.size || 14, color: o.color || TEXT, bold: o.bold,
    fontFace: o.mono ? MONO : undefined,
    align: o.align || "center", valign: o.valign || "middle",
    margin: o.margin ?? 4, objectName: nm(o.name || "box"),
  }));
}

// A node: function name in mono on top, a description underneath.
function node(s, x, y, w, h, fn, sub, o = {}) {
  const color = o.color || TEXT;
  const runs = [{ text: fn, options: { fontFace: MONO, bold: true, fontSize: o.fnSize || 14, color, breakLine: !!sub } }];
  if (sub) runs.push({ text: sub, options: { fontSize: o.subSize || 12, color: o.subColor || color } });
  box(s, x, y, w, h, runs, { ...o, color });
}

function diamond(s, x, y, w, h, text, o = {}) {
  box(s, x, y, w, h, text, { shape: pres.shapes.DIAMOND, fill: o.fill || WHITE, line: o.line || AMBER, lw: 2, size: o.size || 12, margin: 2, name: "decision", ...o });
}

function line(s, x1, y1, x2, y2, o = {}) {
  s.addShape(pres.shapes.LINE, textOpts({
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1 ? true : undefined, flipV: y2 < y1 ? true : undefined,
    line: textOpts({ color: o.color || MIST, width: o.width || 1.75, dashType: o.dash, endArrowType: o.head === false ? undefined : "triangle" }),
    objectName: nm("arrow"),
  }));
}
const arrow = line;
// Polyline through points; only the last segment carries the arrowhead.
function path(s, pts, o = {}) {
  for (let i = 0; i < pts.length - 1; i++) {
    line(s, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], { ...o, head: i === pts.length - 2 ? o.head : false });
  }
}

function label(s, x, y, w, h, text, o = {}) {
  s.addText(text, textOpts({
    x, y, w, h, isTextBox: true, fontSize: o.size || 12, color: o.color || SLATE, bold: o.bold, italic: o.italic,
    fontFace: o.mono ? MONO : undefined, align: o.align || "left", valign: o.valign || "middle", margin: 0,
    objectName: nm(o.name || "label"),
  }));
}

function badge(s, x, y, n, o = {}) {
  const d = o.d || 0.5;
  box(s, x, y, d, d, String(n), { shape: pres.shapes.OVAL, fill: o.fill || AMBER, color: o.color || NIGHT, bold: true, size: o.size || 16, margin: 0, name: "badge" });
}

function bullets(s, x, y, w, h, items, o = {}) {
  const runs = items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } }));
  s.addText(runs, textOpts({ x, y, w, h, isTextBox: true, fontSize: o.size || 14, color: o.color || TEXT, valign: "top", paraSpaceAfter: 6, margin: 0, objectName: nm("bullets") }));
}

function content(title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: currentSection });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
let currentSection = "Overview";
function section(kicker, title, body, notes) {
  currentSection = title.replace(/^fogseg /, "fogseg ");
  pres.addSection({ title: currentSection });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: currentSection });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  s.addText(body, { placeholder: "body" });
  if (notes) s.addNotes(notes);
  return s;
}

// ============================================================
// 1. Title
pres.addSection({ title: "Overview" });
{
  const s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Overview" });
  s.addText("Checking the fog labels against weather stations", { placeholder: "title" });
  s.addText("PR #5 walkthrough: what runs, in what order, with what data", { placeholder: "body" });
  label(s, 0.8, 6.3, 8, 0.4, "feature/station-validation → develop", { color: MIST, size: 14, mono: true });
  // Motif: a pixel grid with stations (amber dots) on it
  const gx = 9.0, gy = 1.6, cell = 0.52, gap = 0.08;
  const fogCells = new Set(["1,2", "1,3", "2,1", "2,2", "2,3", "3,2", "4,4"]);
  for (let r = 0; r < 7; r++) for (let c = 0; c < 6; c++) {
    const fog = fogCells.has(`${r},${c}`);
    s.addShape(pres.shapes.RECTANGLE, { x: gx + c * (cell + gap), y: gy + r * (cell + gap), w: cell, h: cell,
      fill: { color: fog ? PALE : SLATE, transparency: fog ? 10 : 55 }, line: { color: NIGHT, width: 0 }, objectName: nm("pixel") });
  }
  for (const [r, c] of [[1, 2], [2, 4], [4, 1], [5, 3], [2, 1]]) {
    s.addShape(pres.shapes.OVAL, { x: gx + c * (cell + gap) + cell / 2 - 0.11, y: gy + r * (cell + gap) + cell / 2 - 0.11, w: 0.22, h: 0.22,
      fill: { color: AMBER }, line: { color: NIGHT, width: 1.5 }, objectName: nm("station") });
  }
  s.addNotes("This deck walks through PR #5 in the order the data moves: download the station reports, pair them with the labels, score them, and tune the BTD threshold. The grid on the right is the motif: satellite pixels (pale = labelled fog) and stations (amber dots).");
}

// 2. Big picture
{
  const s = content("Three commands, one pipeline", "Run stations once (it takes hours and can resume), then validate on the training season, then tune. The test season is validated once at the very end with --final.");
  const xs = [0.7, 4.95, 9.2], w = 3.45;
  const cmds = [["fogseg stations", "download and decode reports"], ["fogseg validate", "pair reports with labels, score"], ["fogseg tune", "relabel per threshold, score, decide"]];
  const ins = ["IEM (METAR) · Ogimet (SYNOP)\nNOAA station list", "satellite data (Zarr)\n+ one label store", "satellite data + label store\n+ grids in the config"];
  const outs = [["observations_metar.csv", "observations_synop.csv"], ["cases_<mode>_<season>.csv", "scores_<mode>_<season>.csv"], ["tuning_offset.csv", "tuning_btd_fixed.csv", "tuning_btd_fixed_fixed_mode.csv"]];
  for (let i = 0; i < 3; i++) {
    label(s, xs[i], 1.45, w, 0.3, "reads", { size: 12, color: SLATE, bold: true });
    box(s, xs[i], 1.75, w, 0.85, ins[i], { size: 14, name: "input" });
    arrow(s, xs[i] + w / 2, 2.6, xs[i] + w / 2, 3.05);
    node(s, xs[i], 3.05, w, 1.0, cmds[i][0], cmds[i][1], { fill: NIGHT, color: WHITE, subColor: PALE, fnSize: 18, subSize: 14, name: "command" });
    arrow(s, xs[i] + w / 2, 4.05, xs[i] + w / 2, 4.5);
    label(s, xs[i], 4.5, w, 0.3, "writes", { size: 12, color: SLATE, bold: true });
    box(s, xs[i], 4.8, w, 1.25, outs[i].join("\n"), { fill: WHITE, line: PALE, lw: 1.5, mono: true, size: 12, name: "output" });
  }
  arrow(s, xs[0] + w, 3.55, xs[1], 3.55, { color: AMBER, width: 2.25 });
  arrow(s, xs[1] + w, 3.55, xs[2], 3.55, { color: AMBER, width: 2.25 });
  label(s, 0.7, 6.3, 12, 0.45, "Order: stations (once, resumable) → validate --season 2023-24 → tune → validate --season 2024-25 --final (once, at the end)", { size: 14, color: TEXT });
}

// 3. Module map
{
  const s = content("Which file does what", "cli.py holds the three command functions. Each command leans on one main module. observations.py borrows select_stations() from validation.py so both commands pick stations the same way.");
  node(s, 5.35, 1.4, 2.6, 0.75, "cli.py", "the three commands", { fill: NIGHT, color: WHITE, subColor: PALE });
  const row2 = [[0.7, 2.85, "observations.py", "download, decode, filter"], [5.0, 3.35, "validation.py", "truth, matching, case table, views"], [9.3, 3.35, "tuning.py", "relabel per value, decide"]];
  for (const [x, w, fn, sub] of row2) {
    node(s, x, 2.85, w, 0.95, fn, sub, { fill: FOG });
    path(s, [[6.65, 2.15], [6.65, 2.5], [x + w / 2, 2.5], [x + w / 2, 2.85]]);
  }
  node(s, 0.7, 4.55, 1.35, 0.95, "metar.py", "visibility, ceiling", { fill: WHITE, line: PALE, lw: 1.5 });
  node(s, 2.2, 4.55, 1.35, 0.95, "synop.py", "visibility, ceiling", { fill: WHITE, line: PALE, lw: 1.5 });
  node(s, 5.0, 4.55, 3.35, 0.95, "scores.py", "counts, POD FAR CSI bias HSS PSS, bootstrap", { fill: WHITE, line: PALE, lw: 1.5 });
  box(s, 9.3, 4.55, 3.35, 0.95, [{ text: "labels.py", options: { fontFace: MONO, bold: true, breakLine: true } }, { text: "dynamic_threshold.py", options: { fontFace: MONO, bold: true, breakLine: true } }, { text: "existing labelling rule (reused)", options: { fontSize: 12 } }], { fill: WHITE, line: MIST, lw: 1.5, dash: "dash", size: 13 });
  arrow(s, 1.375, 3.8, 1.375, 4.55); arrow(s, 2.875, 3.8, 2.875, 4.55);
  arrow(s, 6.675, 3.8, 6.675, 4.55);
  arrow(s, 10.975, 3.8, 10.975, 4.55);
  path(s, [[10.0, 3.8], [10.0, 4.17], [8.0, 4.17], [8.0, 4.55]]);
  arrow(s, 3.55, 3.4, 5.0, 3.4, { dash: "dash", color: AMBER, width: 2 });
  label(s, 3.55, 3.05, 1.45, 0.3, "select_stations()", { size: 10, mono: true, color: SLATE, align: "center" });
  // legend
  line(s, 0.7, 6.2, 1.4, 6.2, {});
  label(s, 1.55, 6.05, 2.2, 0.3, "calls", { size: 12 });
  line(s, 3.0, 6.2, 3.7, 6.2, { dash: "dash", color: AMBER, width: 2 });
  label(s, 3.85, 6.05, 4.5, 0.3, "borrows one function", { size: 12 });
  box(s, 7.3, 6.0, 0.4, 0.35, "", { fill: WHITE, line: MIST, dash: "dash", lw: 1.5, name: "legend-existing" });
  label(s, 7.85, 6.05, 3, 0.3, "existing code, unchanged", { size: 12 });
}

// ---------- Part 1 ----------
section("Part 1", "fogseg stations", "Downloads METAR and SYNOP reports for the dataset nights, decodes visibility and cloud ceiling, and keeps the stations that sit on the pixel grid.",
  "Part 1 builds the station table that everything else uses.");

// 4. stations flow
{
  const s = content("fogseg stations, step by step", "collect_observations() runs these six steps for each source. Step 4 is what makes it resumable: a file already on disk is never fetched again.");
  const steps = [
    ["candidate_stations()", "NOAA list (SYNOP) · IEM networks (METAR)", "every station: station, source, lat, lon, elevation_m"],
    ["select_stations()", "nearest pixel ≤ 5 km · elevation ≤ 1000 m", "+ iy, ix, distance_km"],
    ["download_requests()", "METAR: station × season · SYNOP: block × night", "list of (url, raw file)"],
    ["download_all()", "skip files on disk, fetch the rest", "raw files + the number that failed"],
    ["parse_metar_csv() / parse_synop_text()", "metar.py / synop.py on every line", "rows: source, station, time, visibility_m, ceiling_m"],
    ["merge + is_near_scene()", "add coordinates, keep reports within 1 h of a scene", "observations_<source>.csv"],
  ];
  label(s, 1.4, 1.3, 5.5, 0.3, "step", { size: 12, bold: true });
  label(s, 7.75, 1.3, 5, 0.3, "what comes out", { size: 12, bold: true });
  steps.forEach(([fn, sub, out], i) => {
    const y = 1.62 + i * 0.86;
    badge(s, 0.7, y + 0.1, i + 1);
    node(s, 1.4, y, 5.6, 0.7, fn, sub, { fill: FOG, fnSize: 13 });
    arrow(s, 7.0, y + 0.35, 7.6, y + 0.35, { color: AMBER, width: 2 });
    box(s, 7.75, y, 4.95, 0.7, out, { fill: WHITE, line: PALE, lw: 1.5, size: 13, align: "left", margin: 8, mono: i === 5, name: "out" });
    if (i < steps.length - 1) arrow(s, 4.2, y + 0.7, 4.2, y + 0.86);
  });
}

// 5. download() flowchart
{
  const s = content("download(): resumable and polite", "Throttling servers answer with an empty body, so an empty answer means try again, not 'no data'. Writing to .part and renaming means a crash never leaves a half file that the next run would skip.");
  box(s, 0.6, 2.25, 1.9, 0.7, [{ text: "download(url, path)", options: { fontFace: MONO, fontSize: 12, bold: true } }], { r: 0.3, fill: NIGHT, color: WHITE, name: "start" });
  arrow(s, 2.5, 2.6, 2.85, 2.6);
  diamond(s, 2.85, 2.05, 2.0, 1.1, "file already\non disk?");
  path(s, [[3.85, 2.05], [3.85, 1.75]]);
  box(s, 2.85, 1.25, 2.0, 0.5, "return True (cached)", { fill: TEAL, color: WHITE, size: 12 });
  label(s, 3.95, 1.78, 0.5, 0.25, "yes", { size: 11 });
  arrow(s, 4.85, 2.6, 5.3, 2.6); label(s, 4.88, 2.3, 0.4, 0.25, "no", { size: 11 });
  node(s, 5.3, 2.25, 1.75, 0.7, "sleep", "pause_s", { fill: FOG, fnSize: 13 });
  arrow(s, 7.05, 2.6, 7.45, 2.6);
  node(s, 7.45, 2.2, 2.2, 0.8, "_answer(url)", "error → \"\"", { fill: FOG, fnSize: 13 });
  arrow(s, 9.65, 2.6, 10.05, 2.6);
  diamond(s, 10.05, 2.0, 2.4, 1.2, "is_valid(text)?");
  arrow(s, 11.25, 2.0, 11.25, 1.85);
  box(s, 9.75, 1.2, 3.0, 0.65, "write path.part → rename to path\nreturn True", { fill: TEAL, color: WHITE, size: 12 });
  label(s, 11.35, 1.86, 0.5, 0.2, "yes", { size: 11 });
  arrow(s, 11.25, 3.2, 11.25, 3.75); label(s, 11.35, 3.35, 0.4, 0.25, "no", { size: 11 });
  diamond(s, 10.05, 3.75, 2.4, 1.1, "tries left?\n(max_tries 3)");
  arrow(s, 10.05, 4.3, 9.65, 4.3); label(s, 9.68, 4.0, 0.5, 0.25, "yes", { size: 11 });
  node(s, 7.45, 3.95, 2.2, 0.7, "sleep", "6 × pause_s", { fill: FOG, fnSize: 13 });
  path(s, [[7.45, 4.3], [6.175, 4.3], [6.175, 2.95]]);
  arrow(s, 11.25, 4.85, 11.25, 5.4); label(s, 11.35, 4.95, 0.4, 0.25, "no", { size: 11 });
  box(s, 10.05, 5.4, 2.4, 0.55, "return False", { fill: RUST, color: WHITE, size: 12 });
  bullets(s, 0.6, 3.6, 5.2, 2.9, [
    "Valid answer: METAR starts with \"station,\" · SYNOP contains \"AAXX\"",
    "Pauses: IEM 5 s, Ogimet 22 s (it allows one query per 20 s)",
    "Failures are counted, logged, and fetched on the next run",
    "Station lists go through the same function, so a re-run uses the same stations",
  ]);
}

// 6. METAR
{
  const s = content("metar.py: two numbers from one message", "main_body() cuts the message to the part between the time group and the first trend or remark word. Visibility is the first 4-digit word. The ceiling is the first BKN, OVC or VV layer, read lowest first.");
  const tokens = ["EDDF", "080020Z", "24005KT", "0300", "FG", "VV001", "05/05", "Q1018", "NOSIG"];
  const body = new Set([2, 3, 4, 5, 6, 7]);
  let x = 0.7; const pos = [];
  tokens.forEach((t, i) => {
    const w = 0.42 + 0.14 * t.length;
    box(s, x, 1.5, w, 0.55, t, { mono: true, size: 14, bold: i === 3 || i === 5, fill: body.has(i) ? FOG : WHITE, line: i === 3 || i === 5 ? AMBER : PALE, lw: i === 3 || i === 5 ? 2.25 : 1, name: "token" });
    pos.push([x, w]); x += w + 0.12;
  });
  const bx1 = pos[2][0], bx2 = pos[7][0] + pos[7][1];
  path(s, [[bx1, 2.18], [bx1, 2.3], [bx2, 2.3], [bx2, 2.18]], { head: false, color: SLATE, width: 1.25 });
  label(s, bx1, 2.35, bx2 - bx1, 0.3, "main_body(): after the time group, before TEMPO / BECMG / NOSIG / RMK", { size: 12, align: "center" });
  const vis = pos[3][0] + pos[3][1] / 2, vv = pos[5][0] + pos[5][1] / 2;
  path(s, [[vis, 2.05], [vis, 2.15]], { head: false });
  path(s, [[vis, 2.7], [vis, 3.15]]);
  path(s, [[vv, 2.7], [vv, 3.15]]);
  node(s, vis - 2.65, 3.15, 3.2, 0.95, "prevailing_visibility_m()", "\"0300\" → 300 m", { fill: NIGHT, color: WHITE, subColor: AMBER, fnSize: 12, subSize: 16 });
  node(s, vv - 0.4, 3.15, 3.1, 0.95, "ceiling_m()", "VV001 = 1 × 100 ft → 30.5 m", { fill: NIGHT, color: WHITE, subColor: AMBER, fnSize: 12, subSize: 16 });
  // Return conventions
  const rows = [
    [{ text: "ceiling_m returns", options: { bold: true, color: WHITE, fill: { color: SLATE } } }, { text: "meaning", options: { bold: true, color: WHITE, fill: { color: SLATE } } }],
    [{ text: "a number", options: { fontFace: MONO } }, "base of the first BKN, OVC or VV layer, in metres"],
    [{ text: "inf", options: { fontFace: MONO } }, "cloud reported, but none covers the sky (FEW, SCT, NSC, CAVOK): known, no ceiling"],
    [{ text: "NaN", options: { fontFace: MONO } }, "unknown: no cloud group, a /// height, or an unknown ////// layer first"],
  ];
  s.addTable(rows, { x: 0.7, y: 4.5, w: 8.2, colW: [1.9, 6.3], fontSize: 13, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, fill: { color: WHITE }, rowH: 0.42, margin: 0.06, objectName: nm("table") });
  box(s, 9.3, 4.5, 3.35, 1.68, [
    { text: "Visibility", options: { bold: true, breakLine: true } },
    { text: "9999 or CAVOK → 10000 m", options: { breakLine: true } },
    { text: "//// → NaN", options: { breakLine: true } },
    { text: "nothing found → NaN", options: {} },
  ], { fill: FOG, size: 13, align: "left", margin: 10, name: "vis-rules" });
  label(s, 0.7, 6.35, 12, 0.35, "Why inf and NaN differ: the truth rule can say \"no fog\" only when the ceiling is known.", { size: 14, color: TEXT, italic: true });
}

// 7. SYNOP
{
  const s = content("synop.py: groups and the ceiling decision", "SYNOP gives a cloud base class, not a height, so the code takes the lower edge of the class. Classes below 1000 m stay below 1000 m, so the truth is not affected.");
  const tokens = [["AAXX", ""], ["08001", ""], ["06260", "station"], ["11503", "iR iX h VV\nh = 5 · VV = 03"], ["82000", "N dd ff\nN = 8"], ["10050", ""], ["…", ""], ["87500", "8 Nh CL CM CH\nNh = 7"]];
  let x = 0.7;
  tokens.forEach(([t, lab]) => {
    const w = t === "…" ? 0.45 : 1.0;
    const hot = lab !== "";
    box(s, x, 1.45, w, 0.55, t, { mono: true, size: 14, bold: hot, fill: hot ? FOG : WHITE, line: hot ? AMBER : PALE, lw: hot ? 2.25 : 1, name: "group" });
    if (lab) label(s, x - 0.3, 2.05, w + 0.6, 0.6, lab, { size: 12, align: "center", mono: false });
    x += w + 0.12;
  });
  box(s, 9.75, 1.45, 2.9, 1.2, [
    { text: "visibility_m(\"03\")", options: { fontFace: MONO, bold: true, breakLine: true } },
    { text: "codes 0–50 × 100 m → ", options: {} }, { text: "300 m", options: { bold: true, color: AMBER } },
  ], { fill: NIGHT, color: WHITE, size: 13, name: "vis" });
  // Ceiling flowchart
  label(s, 0.7, 3.0, 8, 0.35, "_ceiling_m(): is there a low deck, and how high is it?", { size: 16, bold: true, color: TEXT });
  const y = 3.55, dh = 1.15, dy = y;
  box(s, 0.7, dy + 0.15, 2.0, 0.85, "cover = Nh of the 8-group\n(else N)", { fill: FOG, size: 12 });
  arrow(s, 2.7, dy + 0.575, 3.0, dy + 0.575);
  diamond(s, 3.0, dy, 2.0, dh, "N or Nh = 9?");
  arrow(s, 5.0, dy + 0.575, 5.4, dy + 0.575); label(s, 5.02, dy + 0.28, 0.4, 0.25, "no", { size: 11 });
  diamond(s, 5.4, dy, 2.0, dh, "cover ≤ 5\noktas?");
  arrow(s, 7.4, dy + 0.575, 7.8, dy + 0.575); label(s, 7.42, dy + 0.28, 0.4, 0.25, "no", { size: 11 });
  diamond(s, 7.8, dy, 2.0, dh, "base class h\nknown?");
  arrow(s, 9.8, dy + 0.575, 10.2, dy + 0.575); label(s, 9.82, dy + 0.28, 0.4, 0.25, "yes", { size: 11 });
  box(s, 10.2, dy + 0.1, 2.45, 0.95, [{ text: "lower edge of class h", options: { breakLine: true } }, { text: "h = 5 → 600 m", options: { bold: true } }], { fill: NIGHT, color: WHITE, size: 13 });
  const outs = [[4.0, "0 m (sky obscured)", "yes", TEAL], [6.4, "inf (no ceiling)", "yes", SLATE], [8.8, "NaN (unknown)", "no", MIST]];
  for (const [cx, t, yn, col] of outs) {
    arrow(s, cx, dy + dh, cx, 5.25);
    label(s, cx + 0.08, dy + dh + 0.08, 0.5, 0.25, yn, { size: 11 });
    box(s, cx - 1.0, 5.25, 2.0, 0.55, t, { fill: col, color: WHITE, size: 13 });
  }
  label(s, 0.7, 6.2, 12, 0.4, "Result for the example: Synop(station=\"06260\", visibility_m=300.0, ceiling_m=600.0)", { size: 14, mono: true, color: TEXT });
}

// ---------- Part 2 ----------
section("Part 2", "fogseg validate", "Pairs every station report with the label at its pixel for one season, then scores the labels with bootstrap intervals over nights.",
  "Part 2 turns reports into cases and cases into scores.");

// 8. validate flow
{
  const s = content("fogseg validate, step by step", "The guard keeps the test season unseen. stored_params() reads the thresholds saved inside the label store, so the hidden test matches the labels even after a config edit.");
  diamond(s, 0.6, 1.5, 2.7, 1.35, "training season\nor --final?", { size: 13 });
  arrow(s, 1.95, 2.85, 1.95, 3.35); label(s, 2.05, 2.95, 0.4, 0.25, "no", { size: 11 });
  box(s, 0.85, 3.35, 2.2, 0.55, "PermissionError", { fill: RUST, color: WHITE, mono: true, size: 12 });
  arrow(s, 3.3, 2.175, 3.7, 2.175); label(s, 3.3, 1.85, 0.4, 0.25, "yes", { size: 11 });
  node(s, 3.7, 1.75, 2.65, 0.85, "read_observations()", "both CSVs → one table", { fill: FOG, fnSize: 12 });
  arrow(s, 6.35, 2.175, 6.75, 2.175);
  node(s, 6.75, 1.75, 2.65, 0.85, "stored_params()", "thresholds the labels used", { fill: FOG, fnSize: 12 });
  arrow(s, 9.4, 2.175, 9.8, 2.175);
  node(s, 9.8, 1.6, 2.9, 1.15, "season_cases()", "the case table", { fill: NIGHT, color: WHITE, subColor: AMBER, fnSize: 14, subSize: 14 });
  // details of season_cases
  const det = ["check_labels_match()", "season_bounds() + scenes_between()", "on_grid_stations()", "station_cases()", "log_stations_without_cases()"];
  box(s, 9.8, 2.95, 2.9, 1.95, det.map((t, i) => ({ text: t, options: { fontFace: MONO, fontSize: 11, breakLine: i < det.length - 1 } })), { fill: WHITE, line: PALE, lw: 1.5, align: "left", margin: 8, valign: "middle", name: "season-detail" });
  arrow(s, 11.25, 4.9, 11.25, 5.35);
  node(s, 9.8, 5.35, 2.9, 0.85, "score_table()", "dedup → views → scores", { fill: NIGHT, color: WHITE, subColor: PALE, fnSize: 13 });
  arrow(s, 9.8, 5.775, 9.4, 5.775);
  box(s, 6.15, 5.35, 3.25, 0.85, "write cases_<mode>_<season>.csv\nand scores_<mode>_<season>.csv", { fill: WHITE, line: PALE, lw: 1.5, mono: true, size: 11 });
  arrow(s, 6.15, 5.775, 5.75, 5.775);
  box(s, 3.0, 5.35, 2.75, 0.85, "log the main row:\nPOD, FAR, CSI, bias", { fill: FOG, size: 12 });
  bullets(s, 0.6, 4.25, 5.1, 0.9, ["Noon-to-noon season bounds keep each night's 00:00–05:00 scenes in the right season"], { size: 13 });
}

// 9. station_cases
{
  const s = content("station_cases(): reports meet pixels and scenes", "Read across: each step takes what is on its left and adds what is on its right. After step B one row is one case.");
  const rows = [
    ["observations + selected", "A", "merge on station, source", "every report + its pixel iy, ix"],
    ["labels[\"time\"]", "B", "match_observations()", "one report per station × scene"],
    ["labels[\"time\"]", "C", "scene_index + at_cases", "(time, y, x) index of each case"],
    ["data[\"bt\"]", "D", "isel(at_cases)", "IR_108, IR_120 at each case"],
    ["labels[\"label\"], params", "E", "assign(...)", "night, slot, truth_fls, truth_fog, label, hidden, threshold_source"],
    ["", "F", "cases[CASE_COLUMNS]", "the case table, 16 columns"],
  ];
  label(s, 0.6, 1.3, 3.2, 0.3, "reads", { size: 12, bold: true });
  label(s, 4.75, 1.3, 3.6, 0.3, "step", { size: 12, bold: true });
  label(s, 9.05, 1.3, 3.6, 0.3, "adds", { size: 12, bold: true });
  rows.forEach(([inp, letter, fn, out], i) => {
    const y = 1.62 + i * 0.8;
    if (inp) {
      box(s, 0.6, y, 3.4, 0.62, inp, { fill: WHITE, line: PALE, lw: 1.5, mono: true, size: 12 });
      arrow(s, 4.0, y + 0.31, 4.6, y + 0.31);
    }
    badge(s, 4.65, y + 0.08, letter, { d: 0.46, size: 14 });
    node(s, 5.2, y, 3.3, 0.62, fn, "", { fill: i === 5 ? NIGHT : FOG, color: i === 5 ? WHITE : TEXT, fnSize: 13 });
    arrow(s, 8.5, y + 0.31, 8.95, y + 0.31, { color: AMBER, width: 2 });
    box(s, 9.0, y, 3.7, 0.62, out, { fill: FOG, size: i === 4 ? 11 : 13, align: "left", margin: 8 });
    if (i < rows.length - 1) arrow(s, 6.85, y + 0.62, 6.85, y + 0.8);
  });
  box(s, 0.6, 6.45, 12.1, 0.45, "One row = one case:  when (scene_time) · where (iy, ix) · what the station saw (visibility, ceiling) · what the labels say (label)", { fill: NIGHT, color: WHITE, size: 13, name: "case-def" });
}

// 10. timeline
{
  const s = content("Matching in time: the report closest to the scan", "SEVIRI scans from south to north, so Central Europe is seen about 12 minutes after the nominal slot. The report closest to that moment wins, if it is within 30 minutes.");
  const x0 = 1.0, x1 = 12.3, t0 = 0, t1 = 80; // 20:30 .. 21:50
  const X = (m) => x0 + (x1 - x0) * (m - t0) / (t1 - t0);
  const ay = 3.75;
  s.addShape(pres.shapes.RECTANGLE, { x: X(12), y: ay - 0.75, w: X(72) - X(12), h: 1.5, fill: { color: AMBER, transparency: 85 }, line: { color: AMBER, width: 1, dashType: "dash" }, objectName: nm("window") });
  label(s, X(12) + 0.12, ay - 0.68, 2.3, 0.3, "±30 min around the scan", { size: 12, color: TEXT, bold: true });
  line(s, x0, ay, x1, ay, { head: false, color: SLATE, width: 2 });
  for (let m = 0; m <= 80; m += 10) {
    line(s, X(m), ay, X(m), ay + 0.12, { head: false, color: SLATE, width: 1.25 });
    const hh = 20 + Math.floor((30 + m) / 60), mm = (30 + m) % 60;
    label(s, X(m) - 0.4, ay + 0.15, 0.8, 0.3, `${hh}:${String(mm).padStart(2, "0")}`, { size: 12, align: "center" });
  }
  line(s, X(30), ay - 1.55, X(30), ay, { head: false, color: SLATE, width: 2, dash: "dash" });
  label(s, X(30) - 1.2, ay - 1.85, 2.4, 0.3, "scene 21:00 (nominal)", { size: 12, align: "center", color: TEXT });
  line(s, X(42), ay - 1.55, X(42), ay, { head: false, color: AMBER, width: 3 });
  label(s, X(42) - 1.0, ay - 1.85, 2.0, 0.3, "scan ≈ 21:12", { size: 12, align: "center", bold: true, color: TEXT });
  const reps = [[20, "20:50 · 22 min away", false], [50, "21:20 · 8 min away", true]];
  for (const [m, t, win] of reps) {
    s.addShape(pres.shapes.OVAL, { x: X(m) - 0.16, y: ay - 0.16, w: 0.32, h: 0.32, fill: { color: win ? TEAL : MIST }, line: { color: WHITE, width: 1.5 }, objectName: nm("report") });
    box(s, X(m) - 1.15, ay + 0.6, 2.3, 0.6, win ? `${t}  ✓ used` : t, { fill: win ? TEAL : FOG, color: win ? WHITE : TEXT, size: 12, name: "report-label" });
  }
  box(s, 0.7, 5.25, 7.4, 1.45, [
    { text: "pd.merge_asof(wanted, observations,", options: { breakLine: true } },
    { text: "    left_on=\"scan_time\", right_on=\"time\",", options: { breakLine: true } },
    { text: "    by=[\"station\", \"source\"], direction=\"nearest\",", options: { breakLine: true } },
    { text: "    tolerance=pd.Timedelta(minutes=30))", options: {} },
  ], { fill: FOG, mono: true, size: 12, align: "left", margin: 10, valign: "middle", name: "code" });
  bullets(s, 8.5, 5.3, 4.2, 1.4, ["wanted = every station × every scene", "no report in the window → no case", "stations only match their own reports"], { size: 13 });
}

// 11. truth rule
{
  const s = content("The truth rule: did the station see fog?", "Any comparison with NaN is False, so a missing value can never produce a 'no'. It can only fail to produce a 'yes'.");
  diamond(s, 0.7, 1.5, 3.8, 1.5, "visibility < 1000 m\nor ceiling < 1000 m?", { size: 13 });
  arrow(s, 4.5, 2.25, 5.1, 2.25); label(s, 4.6, 1.95, 0.4, 0.25, "yes", { size: 11 });
  box(s, 5.1, 1.85, 2.5, 0.8, [{ text: "1", options: { bold: true, fontSize: 20, breakLine: true } }, { text: "fog or low stratus", options: { fontSize: 12 } }], { fill: TEAL, color: WHITE });
  arrow(s, 2.6, 3.0, 2.6, 3.45); label(s, 2.7, 3.08, 0.4, 0.25, "no", { size: 11 });
  diamond(s, 0.7, 3.45, 3.8, 1.5, "both known\nand ≥ 1000 m?", { size: 13 });
  arrow(s, 4.5, 4.2, 5.1, 4.2); label(s, 4.6, 3.9, 0.4, 0.25, "yes", { size: 11 });
  box(s, 5.1, 3.8, 2.5, 0.8, [{ text: "0", options: { bold: true, fontSize: 20, breakLine: true } }, { text: "no fog", options: { fontSize: 12 } }], { fill: SLATE, color: WHITE });
  arrow(s, 2.6, 4.95, 2.6, 5.4); label(s, 2.7, 5.03, 0.4, 0.25, "no", { size: 11 });
  box(s, 1.35, 5.4, 2.5, 0.8, [{ text: "−1", options: { bold: true, fontSize: 20, breakLine: true } }, { text: "undefined, not scored", options: { fontSize: 12 } }], { fill: MIST, color: WHITE });
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: SLATE }, align: "center" } });
  const c = (t, o = {}) => ({ text: t, options: { align: "center", fontFace: MONO, ...o } });
  const rows = [[hdr("visibility (m)"), hdr("ceiling (m)"), hdr("truth_fls")],
    [c("300"), c("inf"), c("1", { bold: true })], [c("8000"), c("180"), c("1", { bold: true })], [c("8000"), c("inf"), c("0", { bold: true })],
    [c("8000"), c("NaN"), c("−1", { bold: true })], [c("NaN"), c("180"), c("1", { bold: true })]];
  s.addTable(rows, { x: 8.3, y: 1.6, w: 4.4, colW: [1.6, 1.4, 1.4], fontSize: 14, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, rowH: 0.48, objectName: nm("table") });
  bullets(s, 8.3, 4.75, 4.4, 1.6, ["truth_fog uses visibility only: the stricter \"ground fog\" view", "limit is stations.truth_limit_m = 1000"], { size: 13 });
}

// 12. hidden + dedup
{
  const s = content("Hidden pixels, and one case per pixel", "Hidden cases stay in the case table but leave the main score: the satellite cannot see fog under a high cloud. The dedup order was fixed in this review.");
  label(s, 0.7, 1.4, 5.6, 0.4, "hidden: a high cloud over the pixel", { size: 18, bold: true, color: TEXT });
  box(s, 0.7, 2.0, 2.6, 0.75, "IR_108 < 255 K\n(cold top)", { fill: FOG, size: 13, mono: false });
  box(s, 0.7, 3.15, 2.6, 0.75, "IR_108 − IR_120 ≥ 1.5 K\n(cirrus signal)", { fill: FOG, size: 13 });
  path(s, [[3.3, 2.375], [3.65, 2.375], [3.65, 2.95]], { head: false });
  path(s, [[3.3, 3.525], [3.65, 3.525], [3.65, 2.95], [4.0, 2.95]]);
  label(s, 3.7, 2.62, 0.4, 0.3, "or", { size: 12, bold: true });
  box(s, 4.0, 2.55, 2.0, 0.8, "hidden = True", { fill: NIGHT, color: WHITE, mono: true, size: 13 });
  bullets(s, 0.7, 4.3, 5.4, 2.2, ["left out of the main score and of tuning", "scored in its own view, with_hidden", "about half the training-season fog reports are hidden"], { size: 14 });
  // dedup
  label(s, 6.9, 1.4, 5.8, 0.4, "deduplicate(): which case a shared pixel keeps", { size: 18, bold: true, color: TEXT });
  const ladder = [["a case with a known truth", "new: an incomplete airport report no longer wins"], ["METAR before SYNOP", "its cloud base is exact; order taken from SOURCES"], ["then the station id", "so the choice is repeatable"]];
  ladder.forEach(([t, sub], i) => {
    const y = 2.0 + i * 0.95;
    badge(s, 6.9, y + 0.12, i + 1);
    box(s, 7.55, y, 5.15, 0.75, [{ text: t, options: { bold: true, fontSize: 14, breakLine: true } }, { text: sub, options: { fontSize: 12 } }], { fill: i === 0 ? WHITE : FOG, line: i === 0 ? AMBER : FOG, lw: 2.25, align: "left", margin: 10 });
  });
  box(s, 6.9, 5.0, 5.8, 1.35, [
    { text: "Example, same pixel and scene", options: { bold: true, breakLine: true } },
    { text: "EHAM (METAR, truth −1) and 06240 (SYNOP, truth 1)", options: { breakLine: true } },
    { text: "before: keeps EHAM, then cannot score it → pixel lost", options: { breakLine: true } },
    { text: "now: keeps 06240 → pixel scored", options: { bold: true } },
  ], { fill: NIGHT, color: WHITE, size: 13, align: "left", margin: 12, name: "dedup-example" });
}

// 13. scoring
{
  const s = content("Scoring: four counts, six scores", "contingency() skips label 255 and truth −1. HSS and PSS count correct 'no' answers too; both were added to the code in this review because the 4.0 K decision rests on them.");
  const gx = 0.7, gy = 1.9, cw = 2.35, ch = 1.25;
  label(s, gx + 1.6, 1.4, 2 * cw, 0.35, "station says", { size: 14, bold: true, align: "center", color: TEXT });
  box(s, gx + 1.6, gy - 0.05, cw - 0.05, 0.45, "fog (1)", { fill: SLATE, color: WHITE, size: 13 });
  box(s, gx + 1.6 + cw, gy - 0.05, cw - 0.05, 0.45, "no fog (0)", { fill: SLATE, color: WHITE, size: 13 });
  box(s, gx, gy + 0.5, 1.5, ch - 0.05, "label:\nfog (1)", { fill: SLATE, color: WHITE, size: 13 });
  box(s, gx, gy + 0.5 + ch, 1.5, ch - 0.05, "label:\nnot fog (0)", { fill: SLATE, color: WHITE, size: 13 });
  const cells = [["hits", TEAL, WHITE], ["false alarms", RUST, WHITE], ["misses", RUST, WHITE], ["correct negatives", TEAL, WHITE]];
  cells.forEach(([t, f, col], i) => {
    const r = Math.floor(i / 2), cI = i % 2;
    box(s, gx + 1.6 + cI * cw, gy + 0.5 + r * ch, cw - 0.05, ch - 0.05, t, { fill: f, color: col, size: 16, bold: true, name: "cell" });
  });
  label(s, gx, 5.15, 6.5, 0.6, "Cases with label 255 (undecided) or truth −1 (unknown) are not counted.", { size: 13, italic: true });
  const f = [["POD", "hits / (hits + misses)", "share of real fog that was labelled"], ["FAR", "FA / (hits + FA)", "share of labelled fog that was wrong"],
    ["CSI", "hits / (hits + misses + FA)", "ignores correct 'no' answers"], ["bias", "(hits + FA) / (hits + misses)", "< 1: too little fog labelled"],
    ["HSS", "2(hits·CN − FA·misses) / …", "correct answers above chance"], ["PSS", "POD − FA / (FA + CN)", "hit rate minus false-alarm rate"]];
  f.forEach(([n, formula, what], i) => {
    const y = 1.45 + i * 0.83;
    box(s, 7.4, y, 0.95, 0.68, n, { fill: i >= 4 ? AMBER : NIGHT, color: i >= 4 ? NIGHT : WHITE, bold: true, size: 15 });
    box(s, 8.45, y, 4.25, 0.68, [{ text: formula, options: { fontFace: MONO, fontSize: 12, bold: true, breakLine: true } }, { text: what, options: { fontSize: 12 } }], { fill: FOG, align: "left", margin: 8 });
  });
}

// 14. bootstrap
{
  const s = content("Intervals: resample whole nights", "Cases on one night share the weather, so they are not independent. Resampling whole nights gives honest intervals. Illustration: 40 synthetic nights, 1000 draws with seed 0, computed with the PR's own functions.");
  const nights = ["night 1", "night 2", "night 3", "…", "night n"];
  label(s, 0.7, 1.35, 5, 0.3, "counts_by_night(): one row of 4 counts per night", { size: 13, bold: true, color: TEXT });
  nights.forEach((n, i) => box(s, 0.7, 1.75 + i * 0.62, 2.7, 0.5, n === "…" ? "…" : `${n}   [h m f c]`, { fill: n === "…" ? WHITE : FOG, mono: n !== "…", size: 12 }));
  arrow(s, 3.5, 3.05, 4.3, 3.05, { color: AMBER, width: 2.25 });
  label(s, 3.45, 2.45, 0.95, 0.55, "× 1000", { size: 14, bold: true, align: "center", color: TEXT });
  const draws = [["draw 1:", "n2 n2 n7 n1 …"], ["draw 2:", "n5 n1 n1 n9 …"], ["draw 3:", "n3 n8 n4 n4 …"]];
  draws.forEach(([a, b], i) => box(s, 4.4, 1.9 + i * 0.75, 2.85, 0.58, [{ text: a + " ", options: { bold: true } }, { text: b, options: { fontFace: MONO } }], { fill: WHITE, line: PALE, lw: 1.5, size: 12, align: "left", margin: 8 }));
  label(s, 4.4, 4.2, 2.85, 0.7, "sum each draw's counts → one CSI per draw", { size: 12, align: "center" });
  arrow(s, 7.3, 3.05, 7.75, 3.05, { color: AMBER, width: 2.25 });
  const hist = [7, 16, 27, 63, 103, 150, 158, 158, 142, 106, 34, 21, 10, 5];
  const mids = [0.413, 0.42, 0.428, 0.435, 0.442, 0.449, 0.456, 0.463, 0.47, 0.478, 0.485, 0.492, 0.499, 0.506];
  s.addChart(pres.charts.BAR, [{ name: "draws", labels: mids.map((m) => m.toFixed(2)), values: hist }], {
    x: 7.75, y: 1.35, w: 4.95, h: 3.6, barDir: "col", barGapWidthPct: 15, chartColors: [HEX.accent4],
    showTitle: true, title: "CSI of 1000 draws", titleFontSize: 13, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showLegend: false, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFrequency: 3,
    valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" }, objectName: nm("histogram"),
  });
  box(s, 0.7, 5.3, 12.0, 1.3, [
    { text: "interval() = 2.5th and 97.5th percentile of the 1000 values", options: { bold: true, breakLine: true } },
    { text: "Here: CSI 0.459, 95 % interval 0.426–0.491.  On the real training season: CSI 0.420 (0.369–0.472) at 4.0 K.", options: {} },
  ], { fill: NIGHT, color: WHITE, size: 14, align: "left", margin: 14, name: "interval" });
}

// 15. views
{
  const s = content("Seven views of the same cases", "Every view is a filter of the deduplicated table, except source, which deduplicates each source on its own so a SYNOP station is not hidden by an airport.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: SLATE } } });
  const m = (t) => ({ text: t, options: { fontFace: MONO, bold: true } });
  const rows = [[hdr("view"), hdr("groups"), hdr("what it shows")],
    [m("main"), "all", "the headline score: visible cases, FLS truth"],
    [m("source"), "metar, synop", "do airports and SYNOP stations agree?"],
    [m("threshold_source"), "fixed / dynamic / fallback", "how the scene's threshold was chosen"],
    [m("lat_band"), "43–47, 47–51, 51–56", "the north–south drift of false fog"],
    [m("slot"), "21:00, 00:00, 03:00, 05:00", "time of night"],
    [m("fog_only"), "all", "against visibility only: ground fog"],
    [m("with_hidden"), "all", "hidden cases counted in, as misses"]];
  s.addTable(rows, { x: 0.7, y: 1.5, w: 8.3, colW: [2.2, 2.6, 3.5], fontSize: 13, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, rowH: 0.52, margin: 0.06, objectName: nm("table") });
  box(s, 9.4, 1.5, 3.3, 4.2, [
    { text: "Each row also counts what the score could not see", options: { bold: true, breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "n_fls_ignored", options: { fontFace: MONO, bold: true, breakLine: true } },
    { text: "fog reports where the label is 255", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "n_no_fls_ignored", options: { fontFace: MONO, bold: true, breakLine: true } },
    { text: "no-fog reports where the label is 255", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "n_fls_hidden", options: { fontFace: MONO, bold: true, breakLine: true } },
    { text: "fog reports under high cloud", options: {} },
  ], { fill: FOG, size: 13, align: "left", valign: "top", margin: 14, name: "extra-counts" });
}

// ---------- Part 3 ----------
section("Part 3", "fogseg tune", "Throws away the stored labels, relabels every case at each threshold on a grid, scores each value and decides whether the best one is clearly better.",
  "Part 3 reuses Part 2's case table and scoring, with one new step: relabelling.");

// 16. tune flow
{
  const s = content("fogseg tune: three grids", "There is no --season option: the season is stations.tune_season, so tune can never touch the test season. The label column of the store is never used; every case is relabelled.");
  box(s, 0.7, 1.45, 12.0, 0.7, [{ text: "cases = ", options: { fontFace: MONO } }, { text: "season_cases()", options: { fontFace: MONO, bold: true } }, { text: " → ", options: {} }, { text: "deduplicate()", options: { fontFace: MONO, bold: true } }, { text: " → keep truth 0 or 1 and not hidden", options: {} }], { fill: NIGHT, color: WHITE, size: 14 });
  const grids = [["1", "offset", "dynamic mode · current 1.5 K", "tuning_offset.csv", "kept or changed"],
    ["2", "btd_fixed as fallback", "dynamic mode · uses the offset from 1", "tuning_btd_fixed.csv", "kept or changed"],
    ["3", "btd_fixed", "fixed mode · for comparison", "tuning_btd_fixed_fixed_mode.csv", "report only"]];
  grids.forEach(([n, p, sub, file, verdict], i) => {
    const x = 0.7 + i * 4.1;
    arrow(s, x + 1.9, 2.15, x + 1.9, 2.55);
    box(s, x, 2.55, 3.8, 1.55, [
      { text: p, options: { fontFace: MONO, bold: true, fontSize: 14, breakLine: true } },
      { text: sub, options: { fontSize: 12, breakLine: true } },
      { text: file, options: { fontFace: MONO, fontSize: 11, breakLine: true } },
      { text: verdict, options: { fontSize: 12, bold: true, color: i === 2 ? SLATE : TEAL } },
    ], { fill: FOG, margin: 10 });
    badge(s, x - 0.15, 2.4, n, { d: 0.42, size: 13 });
  });
  arrow(s, 4.5, 3.3, 4.8, 3.3, { color: AMBER, width: 2 });
  label(s, 0.7, 4.35, 8, 0.35, "each grid runs tune_parameter():", { size: 14, bold: true, color: TEXT });
  const chain = [["grid_values()", "[start, stop, step] → list of values"], ["grid_labels()", "relabel the case pixels at every value"], ["decide()", "score each value, pick, compare"]];
  chain.forEach(([fn, sub], i) => {
    const x = 0.7 + i * 4.1;
    node(s, x, 4.8, 3.8, 0.95, fn, sub, { fill: i === 1 ? NIGHT : WHITE, color: i === 1 ? WHITE : TEXT, subColor: i === 1 ? PALE : TEXT, line: PALE, lw: 1.5 });
    if (i < 2) arrow(s, x + 3.8, 5.275, x + 4.1, 5.275);
  });
  label(s, 0.7, 6.0, 12, 0.5, "btd_fixed grid: 3.0 → 7.0 K in 0.25 K steps (17 values) · offset grid: 0.0 → 3.0 K (13 values)", { size: 13 });
}

// 17. grid labels
{
  const s = content("grid_labels(): one threshold per scene and value", "The clear-ground peak is computed once per scene by usable_peak(), then reused for every grid value. Scenes are read 8 at a time, matching the Zarr chunks on disk.");
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: SLATE } } });
  const m = (t, o = {}) => ({ text: t, options: { fontFace: MONO, ...o } });
  const rows = [[hdr("grid, mode"), hdr("threshold of a scene"), hdr("peak 1.2 K"), hdr("no peak")],
    [m("offset, dynamic", { bold: true }), "clip(peak + value, 3.5, 6.0), else fallback 4.0", m("0 → 3.5 · 1.5 → 3.5 · 3 → 4.2"), m("4.0 always")],
    [m("btd_fixed, dynamic", { bold: true }), "clip(peak + 1.5, 3.5, 6.0), else value", m("3.5 (value unused)"), m("value")],
    [m("btd_fixed, fixed", { bold: true }), "value, on every scene", m("value"), m("value")]];
  s.addTable(rows, { x: 0.7, y: 1.45, w: 12.0, colW: [2.4, 3.7, 4.0, 1.9], fontSize: 13, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, rowH: 0.5, margin: 0.06, objectName: nm("table") });
  // number line: label_scene rule
  label(s, 0.7, 3.55, 12, 0.35, "label_scene(): what one pixel gets at threshold 4.0 K (if the other fog tests pass)", { size: 15, bold: true, color: TEXT });
  const x0 = 1.0, x1 = 12.3, b0 = 2.5, b1 = 5.5;
  const X = (b) => x0 + (x1 - x0) * (b - b0) / (b1 - b0);
  const ly = 5.3;
  box(s, X(2.5), ly - 0.9, X(3.5) - X(2.5), 0.8, "0 · not fog", { fill: SLATE, color: WHITE, size: 14, r: 0.04 });
  box(s, X(3.5), ly - 0.9, X(4.5) - X(3.5), 0.8, "255 · too close to call", { fill: MIST, color: WHITE, size: 14, r: 0.04 });
  box(s, X(4.5), ly - 0.9, X(5.5) - X(4.5), 0.8, "1 · fog", { fill: TEAL, color: WHITE, size: 14, r: 0.04 });
  line(s, x0, ly, x1, ly, { head: false, color: SLATE, width: 2 });
  for (let b = 2.5; b <= 5.5001; b += 0.5) {
    line(s, X(b), ly, X(b), ly + 0.12, { head: false, color: SLATE, width: 1.25 });
    label(s, X(b) - 0.4, ly + 0.15, 0.8, 0.3, `${b.toFixed(1)} K`, { size: 12, align: "center" });
  }
  line(s, X(4.0), ly - 1.35, X(4.0), ly - 0.92, { color: AMBER, width: 3 });
  line(s, X(4.0), ly - 0.1, X(4.0), ly + 0.12, { head: false, color: AMBER, width: 3 });
  label(s, X(4.0) + 0.12, ly - 1.4, 2.6, 0.3, "threshold 4.0 K", { size: 12, bold: true, color: TEXT });
  label(s, X(4.0) + 0.1, ly + 0.5, 4.5, 0.3, "BTD = IR_108 − IR_039 · band = ±0.5 K", { size: 12 });
  label(s, 0.7, 6.25, 12, 0.4, "Pushing hard cases into the 255 band can raise CSI, so every grid row reports n_ignored.", { size: 13, italic: true });
}

// 18. decide
{
  const s = content("decide(): the best value, and is it really better?", "Paired means both values are scored on the same 1000 random night draws, so a foggy night lifts both and cancels in the difference. A value changes only when the lower end of that difference is above 0.");
  const steps = [["score every value", "counts per night → CSI, HSS, PSS, n_ignored"], ["best = highest CSI", "tie → the value closest to current"], ["paired_csi_difference()", "CSI(best) − CSI(current) on the same draws"]];
  steps.forEach(([t, sub], i) => {
    const x = 0.7 + i * 3.05;
    box(s, x, 1.6, 2.75, 1.1, [{ text: t, options: { bold: true, fontSize: i === 2 ? 12 : 14, fontFace: i === 2 ? MONO : undefined, breakLine: true } }, { text: sub, options: { fontSize: 12 } }], { fill: FOG, margin: 8 });
    arrow(s, x + 2.75, 2.15, x + 3.05, 2.15);
  });
  diamond(s, 9.85, 1.5, 2.85, 1.3, "lower end of the\ndifference > 0?", { size: 12 });
  arrow(s, 11.275, 2.8, 11.275, 3.2); label(s, 11.38, 2.85, 0.4, 0.25, "yes", { size: 11 });
  box(s, 10.2, 3.2, 2.15, 0.55, "changed", { fill: TEAL, color: WHITE, bold: true, size: 14 });
  path(s, [[9.85, 2.15], [9.6, 2.15], [9.6, 3.475], [9.3, 3.475]]); label(s, 9.25, 2.75, 0.35, 0.25, "no", { size: 11 });
  box(s, 7.15, 3.2, 2.15, 0.55, "kept", { fill: SLATE, color: WHITE, bold: true, size: 14 });
  // toy example table
  label(s, 0.7, 4.05, 8, 0.35, "Toy example from the walkthrough (real function output)", { size: 14, bold: true, color: TEXT });
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: SLATE }, align: "center" } });
  const c = (t, o = {}) => ({ text: t, options: { align: "center", ...o } });
  const rows = [[hdr("value"), hdr("hits"), hdr("misses"), hdr("FA"), hdr("CN"), hdr("CSI"), hdr("CSI interval"), hdr("ignored")],
    [c("3.5", { bold: true }), c("2"), c("0"), c("1"), c("1"), c("0.67", { bold: true }), c("0.50–1.00"), c("2")],
    [c("4.0 (current)"), c("1"), c("0"), c("1"), c("1"), c("0.50"), c("0.00–1.00"), c("3")],
    [c("4.5"), c("1"), c("1"), c("0"), c("2"), c("0.50"), c("0.50–0.50"), c("2")]];
  s.addTable(rows, { x: 0.7, y: 4.45, w: 8.2, colW: [1.5, 0.75, 0.9, 0.7, 0.7, 0.85, 1.6, 1.2], fontSize: 13, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, rowH: 0.42, objectName: nm("table") });
  box(s, 9.3, 4.45, 3.4, 1.7, [
    { text: "best 3.5 · difference 0.00 to +0.50", options: { bold: true, breakLine: true } },
    { text: "the interval touches 0, so the gain may be chance → ", options: {} },
    { text: "kept", options: { bold: true, color: AMBER } },
  ], { fill: NIGHT, color: WHITE, size: 13, align: "left", margin: 12 });
}

// ---------- Part 4 ----------
section("Part 4", "The decision and this review", "Why the labels now use a fixed 4.0 K threshold, what it scores, and what was fixed while reviewing the PR.",
  "The last part is about outcomes.");

// 19. Why 4.0 K
{
  const s = content("Why 4.0 K and not 3.5 K", "CSI ignores correct 'no' answers, so it rewards labelling too much fog when fog is common. HSS and PSS count them, and both peak at 4.0 K. This change of objective after seeing the data is stated openly in the notes.");
  const xs = ["3.25", "3.50", "3.75", "4.00", "4.25", "4.50"];
  s.addChart(pres.charts.LINE, [
    { name: "CSI", labels: xs, values: [0.434, 0.441, 0.438, 0.420, 0.392, 0.352] },
    { name: "HSS", labels: xs, values: [0.265, 0.316, 0.353, 0.371, 0.368, 0.347] },
    { name: "PSS", labels: xs, values: [0.266, 0.314, 0.346, 0.356, 0.345, 0.317] },
  ], {
    x: 0.6, y: 1.4, w: 7.6, h: 5.3, chartColors: [HEX.accent4, HEX.accent1, HEX.accent2], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
    showTitle: true, title: "Training season 2023-24, fixed mode", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "BTD threshold (K)", catAxisTitleFontSize: 12, catAxisTitleColor: HEX.dk2, catAxisTitleFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 12, valAxisLabelFontSize: 12,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valAxisMinVal: 0.25, valAxisMaxVal: 0.45, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" }, objectName: nm("chart"),
  });
  const cards = [["3.5 K", "CSI peaks here (0.441)", "but 27 % of 'no' reports get labelled fog, and 48 % of the cloud-free north"],
    ["4.0 K", "HSS and PSS peak here", "false fog on 'no' reports halves to 15 %, for 0.02 less CSI"],
    ["5.0 K", "the old threshold", "CSI 0.26: it caught only 27 % of the reported fog"]];
  cards.forEach(([big, head, sub], i) => {
    const y = 1.45 + i * 1.75;
    box(s, 8.6, y, 4.1, 1.55, [
      { text: big, options: { fontSize: 26, bold: true, color: i === 1 ? AMBER : WHITE, breakLine: true } },
      { text: head, options: { fontSize: 14, bold: true, breakLine: true } },
      { text: sub, options: { fontSize: 12 } },
    ], { fill: i === 1 ? NIGHT : SLATE, color: WHITE, align: "left", valign: "middle", margin: 12 });
  });
}

// 20. Results
{
  const s = content("What the new labels score", "The test season scores higher with the same rule: wider low-stratus decks and less fog under high cloud (30 % of fog reports hidden, against 51 % in training). Numbers are from before the dedup fix and move slightly on a re-run.");
  const stats = [["0.26 → 0.42", "training CSI", "old dynamic labels → fixed 4.0 K"], ["0.27 → 0.50", "training POD", "share of reported fog that is labelled"], ["0.60", "test CSI, scored once", "95 % interval 0.55–0.64"]];
  stats.forEach(([big, head, sub], i) => {
    const x = 0.7 + i * 4.05;
    box(s, x, 1.5, 3.8, 2.2, [
      { text: big, options: { fontSize: 36, bold: true, color: AMBER, breakLine: true } },
      { text: head, options: { fontSize: 16, bold: true, breakLine: true } },
      { text: sub, options: { fontSize: 13 } },
    ], { fill: NIGHT, color: WHITE, align: "left", margin: 16 });
  });
  const hdr = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: SLATE } } });
  const rows = [[hdr("labels"), hdr("season"), hdr("POD"), hdr("FAR"), hdr("CSI (95 % interval)")],
    ["dynamic, fallback 5.0 K (before)", "training", "0.27", "0.19", "0.26 (0.20–0.32)"],
    [{ text: "fixed 4.0 K", options: { bold: true } }, "training", "0.50", "0.28", "0.42 (0.37–0.47)"],
    [{ text: "fixed 4.0 K", options: { bold: true } }, "test, scored once", "0.70", "0.19", "0.60 (0.55–0.64)"],
    ["dynamic, fallback 4.0 K", "test, scored once", "0.53", "0.13", "0.49 (0.42–0.54)"]];
  s.addTable(rows, { x: 0.7, y: 4.05, w: 12.0, colW: [3.9, 2.4, 1.3, 1.3, 3.1], fontSize: 13, color: TEXT, border: { type: "solid", color: HEX.accent6, pt: 1 }, rowH: 0.44, objectName: nm("table") });
}

// 21. Fixes
{
  const s = content("Fixed in this review", "Each fix is its own commit on feature/station-validation, with a test that failed before the fix. The full suite passes (304 tests), and ruff check and ruff format --check are clean.");
  const fixes = [
    ["Dedup kept unscorable airport reports", "deduplicate() now puts a known truth first, then METAR, then station id. Used by validate and tune.", "validation.py"],
    ["HSS and PSS were not in the code", "skill() returns hss and pss; every scores_*.csv and tuning_*.csv has them, so the 4.0 K choice can be checked.", "scores.py"],
    ["tune always opened labels_dynamic.zarr", "It now opens the store of labels.mode; a test shows both stores give the same tables.", "cli.py"],
    ["Notes: pixel size and test-season wording", "Pixels are about 3.3 × 5–7 km here; the test season ran once per label mode.", "notes, README"],
  ];
  fixes.forEach(([t, sub, file], i) => {
    const r = Math.floor(i / 2), c = i % 2;
    const x = 0.7 + c * 6.05, y = 1.45 + r * 1.75;
    box(s, x, y, 5.85, 1.55, [
      { text: t, options: { fontSize: 15, bold: true, breakLine: true } },
      { text: sub, options: { fontSize: 13, breakLine: true } },
      { text: file, options: { fontSize: 11, fontFace: MONO, color: SLATE } },
    ], { fill: FOG, align: "left", valign: "top", margin: 12 });
  });
  box(s, 0.7, 5.0, 12.0, 1.85, [
    { text: "Next, on the machine with the data", options: { bold: true, fontSize: 15, color: AMBER, breakLine: true } },
    { text: "fogseg validate --season 2023-24", options: { fontFace: MONO, fontSize: 13, breakLine: true } },
    { text: "fogseg tune", options: { fontFace: MONO, fontSize: 13, breakLine: true } },
    { text: "then refresh the tables in docs/notes/station_validation.md.", options: { fontSize: 13, breakLine: true } },
    { text: "The dedup fix only adds cases where an airport and a SYNOP station share a pixel, so the numbers should move only slightly.", options: { fontSize: 13 } },
  ], { fill: NIGHT, color: WHITE, align: "left", margin: 14 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
