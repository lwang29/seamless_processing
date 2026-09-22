"""HTML for the accepted-clip gallery. Self-contained: no CDN, no build step."""

from __future__ import annotations

import json
from typing import Any, Mapping

STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#101112; color:#e9e9e9;
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
code,.mono { font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
a { color:#7fb2ff; }
header { position:sticky; top:0; z-index:20; background:#17181a; border-bottom:1px solid #2a2c2f;
         padding:10px 16px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0; font-weight:650; letter-spacing:.01em; }
.muted { color:#8b8f94; }
.spacer { flex:1 }
select,button { font:inherit; background:#212326; color:#e9e9e9; border:1px solid #34373b;
                border-radius:6px; padding:5px 10px; cursor:pointer; }
button:hover,select:hover { background:#2a2d31; }
main { padding:16px; max-width:1800px; margin:0 auto; }
details.about { background:#16181a; border:1px solid #26292c; border-radius:8px;
                padding:12px 16px; margin-bottom:16px; }
details.about summary { cursor:pointer; font-weight:600; color:#fff; }
details.about h3 { font-size:13px; margin:18px 0 6px; color:#fff; letter-spacing:.02em; }
details.about table { border-collapse:collapse; margin:10px 0; font-size:13px; }
details.about td, details.about th { border:1px solid #2c2f33; padding:5px 12px; text-align:left; }
.grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(186px,1fr)); }
.card { background:#17191b; border:1px solid #26292c; border-radius:8px; overflow:hidden;
        position:relative; }
.card.rated-good { border-color:#2d6a3f; } .card.rated-bad { border-color:#7a2b2b; }
.card.rated-unsure { border-color:#7a6320; }
.frame { position:relative; width:100%; aspect-ratio:3/4; background:#000; cursor:pointer; }
.frame img, .frame video { width:100%; height:100%; object-fit:cover; display:block; }
.badge { position:absolute; top:6px; left:6px; background:rgba(0,0,0,.72); border-radius:5px;
         padding:1px 7px; font-size:11px; font-weight:650; }
.vend { position:absolute; top:6px; right:6px; background:rgba(0,0,0,.72); border-radius:5px;
        padding:1px 7px; font-size:11px; color:#b9bec4; }
.play { position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        font-size:34px; color:rgba(255,255,255,.82); text-shadow:0 2px 10px rgba(0,0,0,.8);
        pointer-events:none; }
.meta { padding:7px 9px; font-size:11px; color:#9aa0a6; }
.meta b { color:#d7dade; font-weight:600; }
.bars { display:flex; gap:3px; margin-top:5px; }
.bar { flex:1; height:4px; background:#26292c; border-radius:2px; overflow:hidden; }
.bar i { display:block; height:100%; background:#4a7fc6; }
.rate { display:flex; gap:4px; padding:0 9px 9px; }
.rate button { flex:1; padding:3px 0; font-size:11px; }
.rate button.on { outline:2px solid #7fb2ff; }
footer { padding:26px 16px 60px; color:#777; max-width:1000px; margin:0 auto; font-size:13px; }
.warn { background:#241f18; border-left:3px solid #c08a3e; padding:10px 14px; margin:14px 0;
        border-radius:0 6px 6px 0; }
"""

SCRIPT = r"""
const DATA = __DATA__;
const ITEMS = DATA.items;
const KEY = "vibes-gallery-ratings-v1";
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ratings = {}; }
let order = ITEMS.slice();

const $ = id => document.getElementById(id);
const save = () => localStorage.setItem(KEY, JSON.stringify(ratings));

function tally() {
  const counts = {good: 0, bad: 0, unsure: 0};
  for (const v of Object.values(ratings)) if (counts[v] !== undefined) counts[v]++;
  const n = counts.good + counts.bad + counts.unsure;
  $("tally").textContent = n
    ? `${n} rated — ${counts.good} good, ${counts.bad} not good, ${counts.unsure} unsure`
    : `${ITEMS.length} clips`;
}

function card(item) {
  const el = document.createElement("div");
  el.className = "card";
  if (ratings[item.slug]) el.classList.add("rated-" + ratings[item.slug]);
  el.dataset.slug = item.slug;
  const pct = v => Math.round(v * 100);
  el.innerHTML = `
    <div class="frame" data-slug="${item.slug}">
      <img loading="lazy" src="posters/${item.slug}.jpg" alt="">
      <span class="badge">${item.quality.toFixed(2)}</span>
      <span class="vend">${item.vendor}</span>
      <span class="play">&#9654;</span>
    </div>
    <div class="meta">
      <b>${item.gesture_frac.toFixed(2)}</b> of speech active &middot; ${item.speech_s}s speech
      <div class="bars" title="posture / persistence / vigour / integrity">
        <span class="bar"><i style="width:${pct(item.posture)}%"></i></span>
        <span class="bar"><i style="width:${pct(item.persistence)}%"></i></span>
        <span class="bar"><i style="width:${pct(item.vigour)}%"></i></span>
        <span class="bar"><i style="width:${pct(item.integrity)}%"></i></span>
      </div>
    </div>
    <div class="rate">
      <button data-r="good">good</button>
      <button data-r="bad">not good</button>
      <button data-r="unsure">?</button>
    </div>`;
  for (const b of el.querySelectorAll(".rate button")) {
    if (ratings[item.slug] === b.dataset.r) b.classList.add("on");
    b.onclick = ev => {
      ev.stopPropagation();
      ratings[item.slug] = ratings[item.slug] === b.dataset.r ? undefined : b.dataset.r;
      if (!ratings[item.slug]) delete ratings[item.slug];
      save(); tally();
      el.className = "card" + (ratings[item.slug] ? " rated-" + ratings[item.slug] : "");
      for (const o of el.querySelectorAll(".rate button")) o.classList.toggle("on", ratings[item.slug] === o.dataset.r);
    };
  }
  el.querySelector(".frame").onclick = () => play(el, item);
  return el;
}

function play(el, item) {
  const frame = el.querySelector(".frame");
  if (frame.querySelector("video")) { const v = frame.querySelector("video"); v.paused ? v.play() : v.pause(); return; }
  // One <video> per opened card, created on demand: 400 preloading players
  // would saturate the connection and make the grid unusable.
  const v = document.createElement("video");
  v.src = "clips/" + item.slug + ".mp4";
  v.controls = true; v.autoplay = true; v.loop = true; v.playsInline = true;
  frame.innerHTML = "";
  frame.appendChild(v);
}

function draw() {
  const grid = $("grid");
  grid.innerHTML = "";
  const vendor = $("vendor").value;
  const band = $("band").value;
  let rows = order.filter(i => (vendor === "all" || i.vendor === vendor));
  if (band === "low") rows = rows.filter(i => i.quality < 0.5);
  else if (band === "mid") rows = rows.filter(i => i.quality >= 0.5 && i.quality < 0.7);
  else if (band === "high") rows = rows.filter(i => i.quality >= 0.7);
  if ($("unrated").checked) rows = rows.filter(i => !ratings[i.slug]);
  for (const item of rows) grid.appendChild(card(item));
  $("shown").textContent = rows.length + " shown";
}

function resort() {
  const how = $("sort").value;
  if (how === "quality-desc") order = ITEMS.slice().sort((a, b) => b.quality - a.quality);
  else if (how === "quality-asc") order = ITEMS.slice().sort((a, b) => a.quality - b.quality);
  else if (how === "random") {
    // Deterministic shuffle from the slug, so a reload does not reorder under
    // someone mid-assessment.
    order = ITEMS.slice().sort((a, b) =>
      (a.slug.split("").reduce((h, c) => h * 31 + c.charCodeAt(0), 7) % 9973) -
      (b.slug.split("").reduce((h, c) => h * 31 + c.charCodeAt(0), 7) % 9973));
  } else order = ITEMS.slice();
  draw();
}

$("export").onclick = () => {
  const rows = ITEMS.filter(i => ratings[i.slug]).map(i => ({
    slug: i.slug, file_id: i.file_id, vendor: i.vendor, start_s: i.start_s,
    gesture_quality: i.quality, rating: ratings[i.slug],
  }));
  if (!rows.length) { alert("No ratings yet. Click good / not good / ? under any clip."); return; }
  const blob = new Blob([JSON.stringify(rows, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `gallery_ratings_${rows.length}of${ITEMS.length}.json`;
  a.click(); URL.revokeObjectURL(a.href);
};

for (const id of ["sort", "vendor", "band"]) $(id).onchange = id === "sort" ? resort : draw;
$("unrated").onchange = draw;
tally(); resort();
"""


def render_page(context: Mapping[str, Any]) -> str:
    sample, population = context["sample"], context["population"]
    vendors = sorted({item["vendor"] for item in context["items"]})

    def vendor_row(dist: Mapping[str, Any]) -> str:
        return "".join(f"<td>{dist['vendors'].get(v, 0):.1%}</td>" for v in vendors)

    options = "".join(f'<option value="{v}">{v}</option>' for v in vendors)
    data = json.dumps(context, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow,noarchive,noimageindex">
<title>Accepted clips — random sample</title>
<style>{STYLE}</style></head><body>

<header>
  <h1>Co-speech subset — random sample of accepted clips</h1>
  <span class="muted" id="shown"></span>
  <span class="muted" id="tally"></span>
  <span class="spacer"></span>
  <label class="muted"><input type="checkbox" id="unrated"> unrated only</label>
  <select id="sort">
    <option value="quality-desc">sort: score high &rarr; low</option>
    <option value="quality-asc">sort: score low &rarr; high</option>
    <option value="random">sort: shuffled</option>
  </select>
  <select id="vendor"><option value="all">all vendors</option>{options}</select>
  <select id="band">
    <option value="all">all scores</option>
    <option value="high">score &ge; 0.70</option>
    <option value="mid">0.50 &ndash; 0.70</option>
    <option value="low">below 0.50</option>
  </select>
  <button id="export">Download ratings</button>
</header>

<main>
<details class="about" open>
<summary>What this is, and what to look at</summary>

<p><b>{sample['clips']} clips drawn uniformly at random</b> from the
{population['clips']:,} clips the automated pipeline accepted
({population['hours']:,.0f} hours). Seed <code>{context['seed']}</code>. The sample is
<i>not</i> curated, sorted by quality before sampling, or hand-picked — that would
flatter the output and answer a question nobody asked. Each clip is 30 seconds,
cropped to the upper body, with that participant's own audio.</p>

<div class="warn"><b>The question worth asking of each clip:</b> while this person is
<i>talking</i>, do their hands and arms move naturally and visibly? Technical
cleanliness is not the criterion — a clip with perfect tracking and a participant
whose hands rest in their lap is a failure of this pipeline, and seeing any here
is the most useful thing you could tell us.</div>

<h3>Is the sample representative?</h3>
<table>
<tr><th></th><th>clips</th><th>score p10</th><th>p50</th><th>p90</th>{''.join(f'<th>{v}</th>' for v in vendors)}</tr>
<tr><td><b>this sample</b></td><td>{sample['clips']}</td>
<td>{sample['quality'][0]}</td><td>{sample['quality'][1]}</td><td>{sample['quality'][2]}</td>{vendor_row(sample)}</tr>
<tr><td>all accepted</td><td>{population['clips']:,}</td>
<td>{population['quality'][0]}</td><td>{population['quality'][1]}</td><td>{population['quality'][2]}</td>{vendor_row(population)}</tr>
</table>

<h3>Reading a card</h3>
<p>The number top-left is the clip's <b>gesture-quality score</b> (0&ndash;1); the
pipeline accepts at 0.34. The four bars underneath are the score's components, in
order: <b>posture</b> (how high the hands are carried and how much space they
use), <b>persistence</b> (how much of the speaking time is active, and how long
each gesture lasts), <b>vigour</b> (arm speed), <b>integrity</b> (how clean and
directionally coherent the motion is). The line above them is the share of
speaking time the filter scored as active, and how many seconds the participant
speaks in the clip.</p>

<p>Sorting by <i>score low &rarr; high</i> is the fastest way to find the worst
material the filter admits — that is where the accept threshold should be judged.</p>

<h3>If you want to record what you see</h3>
<p>The <b>good / not good / ?</b> buttons under each clip save in this browser
only. <b>Download ratings</b> writes them to a JSON file you can send back; they
import straight into the pipeline's calibration set. Nothing is uploaded from
this page.</p>

<p class="muted">Full documentation of every filtering step, its thresholds and
its known failure modes is in <code>pipeline.md</code>; the non-technical summary
is <code>pipeline_in_plain_language.md</code>. Both are in the directory this page
was served from.</p>
</details>

<div class="grid" id="grid"></div>
</main>

<footer>
Unmodified recordings of identifiable research participants from Meta's Seamless
Interaction Dataset, shown for project review under the dataset's research terms.
Please do not redistribute.
</footer>

<script>{SCRIPT.replace("__DATA__", data)}</script>
</body></html>
"""
