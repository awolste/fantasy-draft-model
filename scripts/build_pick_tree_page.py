"""Render `data/pick_tree.json` as a single self-contained HTML explorer.

Click a pick to select it: the path from round 1 highlights, that pick's
own options appear in the next column, and the panel on the right builds a
copyable plan for the whole path.

The page is deliberately one file with the tree inlined -- it gets opened
from a laptop mid-draft, where a missing CDN or a dead relative path is a
failure at the worst possible moment.

Run `scripts/build_pick_tree.py` first; this only formats what that
produced, and never recomputes anything, so the numbers on the page are
exactly the ones that were measured.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draft pick tree &mdash; slot __SLOT__, rounds 1&ndash;6</title>
<style>
  :root {
    --bg: #0e1116; --panel: #161b22; --panel2: #1c2128; --line: #2d333b;
    --ink: #e6edf3; --dim: #8b949e; --faint: #5a6069;
    --rb: #f0883e; --wr: #58a6ff; --accent: #3fb950; --warn: #d29922;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  }
  header {
    padding: 14px 20px; border-bottom: 1px solid var(--line);
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  }
  h1 { font-size: 16px; margin: 0; font-weight: 600; }
  .meta { color: var(--dim); font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  main { display: grid; grid-template-columns: 184px 1fr 300px; gap: 0; align-items: start; }
  @media (max-width: 1100px) { main { grid-template-columns: 1fr; } }

  aside { padding: 16px; border-right: 1px solid var(--line); }
  aside h2, .panel h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--dim); margin: 0 0 10px; font-weight: 600;
  }
  .preset {
    display: block; width: 100%; text-align: left; margin-bottom: 6px;
    background: var(--panel); color: var(--ink); border: 1px solid var(--line);
    border-radius: 6px; padding: 7px 10px; cursor: pointer; font-size: 13px;
  }
  .preset:hover { border-color: var(--accent); }
  .ctl { margin: 16px 0 0; font-size: 12px; color: var(--dim); }
  .ctl input[type=range] { width: 100%; }

  .board { padding: 16px; overflow-x: auto; }
  .cols { display: flex; gap: 12px; min-width: max-content; align-items: flex-start; }
  .col { min-width: 182px; }
  .col > h3 {
    font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
    color: var(--dim); margin: 0 0 8px; font-weight: 600;
  }
  .node {
    background: var(--panel); border: 1px solid var(--line); border-left-width: 3px;
    border-radius: 6px; padding: 8px 10px; margin-bottom: 7px; cursor: pointer;
    transition: border-color .12s, opacity .12s;
  }
  .node:hover { border-color: var(--dim); }
  .node.rb { border-left-color: var(--rb); }
  .node.wr { border-left-color: var(--wr); }
  .node.sel { border-color: var(--accent); background: var(--panel2); }
  .node.faded { opacity: .34; }
  .node .nm { font-weight: 600; display: flex; justify-content: space-between; gap: 8px; }
  .node .pos { font-size: 11px; color: var(--dim); font-family: ui-monospace, Menlo, monospace; }
  .node .row {
    display: flex; justify-content: space-between; font-size: 11.5px;
    color: var(--dim); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 3px;
  }
  .bar { height: 3px; background: var(--line); border-radius: 2px; margin-top: 6px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--warn); }
  .tie { color: var(--warn); }
  .empty { color: var(--faint); font-size: 12px; padding: 6px 2px; }

  .panel { padding: 16px; border-left: 1px solid var(--line); }
  .plan { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .plan div { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid var(--line); }
  .shape { margin: 12px 0; font-size: 13px; }
  .pill {
    display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
    font-family: ui-monospace, Menlo, monospace; border: 1px solid var(--line);
  }
  .pill.rb { color: var(--rb); } .pill.wr { color: var(--wr); }
  textarea {
    width: 100%; height: 150px; background: var(--bg); color: var(--ink);
    border: 1px solid var(--line); border-radius: 6px; padding: 9px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; resize: vertical;
  }
  button.copy {
    margin-top: 8px; width: 100%; background: var(--accent); color: #06120a; border: 0;
    border-radius: 6px; padding: 8px; font-weight: 600; cursor: pointer;
  }
  .caveat {
    margin: 18px 16px; padding: 12px 14px; border: 1px solid var(--line);
    border-left: 3px solid var(--warn); border-radius: 6px;
    color: var(--dim); font-size: 12px; max-width: 900px;
  }
  .caveat b { color: var(--ink); }
  code { font-family: ui-monospace, Menlo, monospace; color: var(--ink); }
</style>
</head>
<body>
<header>
  <h1>Draft pick tree &mdash; slot __SLOT__ of __NTEAMS__, rounds 1&ndash;6</h1>
  <span class="meta">picks __PICKS__ &middot; built __GEN__ &middot; budget __BUDGET__ &middot; __CALLS__ simulations</span>
</header>

<main>
  <aside>
    <h2>Jump to</h2>
    <button class="preset" data-preset="likely">Most likely to be there</button>
    <button class="preset" data-preset="best">Best title odds</button>
    <button class="preset" data-preset="balanced">Balanced 3 RB / 3 WR</button>
    <button class="preset" data-preset="rb">RB-heavy 4 / 2</button>
    <button class="preset" data-preset="wr">WR-heavy 2 / 4</button>
    <button class="preset" data-preset="clear">Clear</button>

    <div class="ctl">
      <label>Fade branches below <b id="thrv">0</b>% availability</label>
      <input type="range" id="thr" min="0" max="90" step="5" value="0">
    </div>
    <div class="ctl">
      <span class="pill rb">RB</span> <span class="pill wr">WR</span>
      <p style="margin:8px 0 0">Amber bar = chance he is still on the board when we pick.</p>
    </div>
  </aside>

  <div class="board"><div class="cols" id="cols"></div></div>

  <div class="panel">
    <h2>Your path</h2>
    <div class="plan" id="plan"></div>
    <div class="shape" id="shape"></div>
    <h2>Copy as a plan</h2>
    <textarea id="prompt" readonly></textarea>
    <button class="copy" id="copy">Copy plan</button>
  </div>
</main>

<div class="caveat">
  <b>Read the availability bar before the percentage.</b> Championship
  probability is only a fair comparison <i>between siblings</i> &mdash; options in
  the same column under the same parent were simulated from an identical board
  with common random numbers. Two nodes under different parents came from
  different boards and are not directly comparable.
  <br><br>
  <b>Most gaps here are statistical ties.</b> Anything marked <span class="tie">tie</span>
  could not be separated from the best option in that column; treat those as
  equally good and pick on availability or roster fit.
  <br><br>
  <b>Opponents are simulated, not known.</b> Each column was built on one
  representative draft; the availability figure is measured over
  __SURVSIMS__ separate simulations and is the honest guide to whether a branch
  will really be open to you.
</div>

<script>
const TREE = __DATA__;
let path = [];
let threshold = 0;

const childrenAt = d => d === 0 ? TREE.roots : (path[d-1] ? path[d-1].children : []);
const pct = x => x == null ? "&mdash;" : (x * 100).toFixed(1) + "%";

function select(depth, node) { path = path.slice(0, depth); path.push(node); render(); }

function render() {
  const cols = document.getElementById("cols");
  cols.innerHTML = "";
  for (let d = 0; d < 6; d++) {
    const kids = childrenAt(d);
    const col = document.createElement("div");
    col.className = "col";
    col.innerHTML = `<h3>Round ${d+1} &middot; pick ${TREE.our_picks[d]}</h3>`;
    if (!kids || !kids.length) {
      col.innerHTML += `<div class="empty">pick round ${d} first</div>`;
      cols.appendChild(col); continue;
    }
    kids.forEach(n => {
      const el = document.createElement("div");
      const sel = path[d] && path[d].player_id === n.player_id;
      const faded = (n.p_available != null && n.p_available * 100 < threshold) && !sel;
      el.className = "node " + n.position.toLowerCase() + (sel ? " sel" : "") + (faded ? " faded" : "");
      el.innerHTML =
        `<div class="nm"><span>${n.name}</span><span class="pos">${n.position}</span></div>
         <div class="row"><span>title ${pct(n.championship_probability)}</span>
           <span>${n.indistinguishable_from_leader ? '<span class="tie">tie</span>' : ""}</span></div>
         <div class="row"><span>there ${pct(n.p_available)}</span><span>adp ${n.adp == null ? "&mdash;" : n.adp.toFixed(1)}</span></div>
         <div class="bar"><i style="width:${(n.p_available||0)*100}%"></i></div>`;
      el.onclick = () => select(d, n);
      col.appendChild(el);
    });
    cols.appendChild(col);
  }
  renderPlan();
  // A newly revealed round sits off-screen to the right on most laptops;
  // bring it into view rather than leaving the user to discover the scroll.
  const deepest = cols.children[Math.min(path.length, 5)];
  if (deepest && path.length) deepest.scrollIntoView({behavior: "smooth", block: "nearest", inline: "end"});
}

function renderPlan() {
  const plan = document.getElementById("plan");
  const shape = document.getElementById("shape");
  if (!path.length) {
    plan.innerHTML = `<div style="border:0;color:var(--faint)">Click a round&nbsp;1 pick to start.</div>`;
    shape.innerHTML = ""; document.getElementById("prompt").value = ""; return;
  }
  plan.innerHTML = path.map((n, i) =>
    `<div><span>R${i+1} &middot; ${n.name}</span><span style="color:var(--${n.position.toLowerCase()})">${n.position}</span></div>`
  ).join("");

  const last = path[path.length-1];
  const rb = last.counts.RB || 0, wr = last.counts.WR || 0;
  shape.innerHTML = `after ${path.length} pick${path.length>1?"s":""}: `
    + `<span class="pill rb">${rb} RB</span> <span class="pill wr">${wr} WR</span>`
    + (path.length === 6 ? "" : ` &middot; <span style="color:var(--dim)">${6-path.length} to go</span>`);

  const lines = path.map((n, i) =>
    `  R${i+1} (pick ${n.overall_pick})  ${n.name} (${n.position})`
    + `  - title ${pct(n.championship_probability)}, available ${pct(n.p_available)}`
    + (n.indistinguishable_from_leader ? " [statistical tie]" : ""));
  document.getElementById("prompt").value =
    `Draft plan, slot ${TREE.draft_slot} of ${TREE.n_teams}, rounds 1-${path.length}:\n`
    + lines.join("\n").replace(/&mdash;/g, "-")
    + `\n\nEnds ${rb} RB / ${wr} WR.`
    + `\nFrom the precomputed pick tree built ${TREE.generated} at ${TREE.budget};`
    + ` "available" is measured over ${TREE.survival_sims} opponent simulations.`;
}

// --- presets: greedy walks and constrained searches over the same tree
function walk(score) {
  path = [];
  let kids = TREE.roots;
  while (kids && kids.length) {
    const best = kids.reduce((a, b) => score(b) > score(a) ? b : a);
    path.push(best); kids = best.children;
  }
  render();
}
function bestWithShape(wantRb) {
  let bestPath = null, bestScore = -1;
  (function dfs(nodes, acc) {
    for (const n of nodes) {
      acc.push(n);
      if (!n.children || !n.children.length) {
        if ((n.counts.RB || 0) === wantRb) {
          const s = acc.reduce((t, x) => t + (x.championship_probability || 0), 0);
          if (s > bestScore) { bestScore = s; bestPath = acc.slice(); }
        }
      } else dfs(n.children, acc);
      acc.pop();
    }
  })(TREE.roots, []);
  if (bestPath) { path = bestPath; render(); }
  else alert("No path in this tree ends at that shape.");
}
document.querySelectorAll(".preset").forEach(b => b.onclick = () => {
  const p = b.dataset.preset;
  if (p === "likely") walk(n => n.p_available == null ? 0 : n.p_available);
  else if (p === "best") walk(n => n.championship_probability || 0);
  else if (p === "balanced") bestWithShape(3);
  else if (p === "rb") bestWithShape(4);
  else if (p === "wr") bestWithShape(2);
  else { path = []; render(); }
});
document.getElementById("thr").oninput = e => {
  threshold = +e.target.value;
  document.getElementById("thrv").textContent = threshold;
  render();
};
document.getElementById("copy").onclick = async () => {
  await navigator.clipboard.writeText(document.getElementById("prompt").value);
  const b = document.getElementById("copy");
  b.textContent = "Copied!";
  setTimeout(() => b.textContent = "Copy plan", 1200);
};
render();
</script>
</body>
</html>
"""


def _check_script(html: str) -> None:
    """Refuse to write a page whose JavaScript does not parse.

    This exists because it already happened: `PAGE` was a normal Python
    string, so every `\n` written inside the JavaScript became a real
    newline, which split `lines.join("\n")` into an unterminated string
    literal. The whole script failed to parse and the page rendered
    completely blank -- no error visible, nothing to click.

    It survived review because the page was "verified" by pasting an
    equivalent script into a browser rather than by running the generated
    file, and the preview pane strips inline scripts so the real artifact
    was never executed. A syntax check on the actual output is the cheap
    guard against the whole class.

    Skipped with a warning when node is unavailable, rather than failing a
    build for a missing dev tool.
    """
    node = shutil.which("node")
    if node is None:
        print("WARNING: node not found -- generated JavaScript was NOT syntax-checked")
        return

    start = html.index("<script>") + len("<script>")
    script = html[start : html.index("</script>", start)]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        tmp = fh.name
    try:
        result = subprocess.run([node, "--check", tmp], capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if result.returncode != 0:
        raise SystemExit(
            "generated JavaScript does not parse -- refusing to write a blank "
            f"page:\n{result.stderr.strip()}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", type=Path, default=Path("data/pick_tree.json"))
    ap.add_argument("--out", type=Path, default=Path("pick_tree.html"))
    args = ap.parse_args()

    tree = json.loads(args.tree.read_text())
    html = (
        PAGE.replace("__DATA__", json.dumps(tree))
        .replace("__SLOT__", str(tree["draft_slot"]))
        .replace("__NTEAMS__", str(tree["n_teams"]))
        .replace("__PICKS__", ", ".join(str(p) for p in tree["our_picks"]))
        .replace("__GEN__", tree["generated"])
        .replace("__BUDGET__", tree["budget"])
        .replace("__CALLS__", str(tree["n_recommend_calls"]))
        .replace("__SURVSIMS__", str(tree["survival_sims"]))
    )
    _check_script(html)
    args.out.write_text(html)
    size = args.out.stat().st_size / 1024
    print(f"wrote {args.out} ({size:.0f} KB)")


if __name__ == "__main__":
    main()
