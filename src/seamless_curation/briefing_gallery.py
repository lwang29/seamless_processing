"""A read-only briefing gallery: what the three detectors do, and what they did.

This is deliberately not the reviewer gallery. That one exists to *collect*
judgements, so it carries a free-text box per clip and says as little as possible
about what the detectors think, to avoid leading the reviewer. This one exists to
*explain* finished work to someone seeing it for the first time, so it leads with
prose and measured rates, drops the notes entirely, and shows only the handful of
numbers that make a given verdict legible.

Everything shown is passed in. This module computes no statistics of its own and
owns no thresholds; it renders what the scan and the analysis measured.
"""

from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


def _human_bytes(count: Any) -> str:
    try:
        size = float(count)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    number = _number(value)
    if not math.isfinite(number):
        return "—"
    if digits == 0:
        return f"{number:,.0f}{suffix}"
    return f"{number:.{digits}f}{suffix}"


def _percent(value: Any, digits: int = 2) -> str:
    number = _number(value)
    return "—" if not math.isfinite(number) else f"{100 * number:.{digits}f}%"


def _duration(seconds: Any) -> str:
    number = _number(seconds)
    if not math.isfinite(number):
        return "—"
    minutes, rest = divmod(int(round(number)), 60)
    return f"{minutes:d}:{rest:02d}"


# The only per-clip numbers the briefing shows. Each is here because it makes one
# verdict legible; anything that needed a paragraph of explanation to interpret
# was cut, since the point of this page is that it can be read cold.
SIGNAL_ROWS: tuple[tuple[str, str, str], ...] = (
    ("fm1_knee_between_torso_p50", "FM1 · knee position along the torso", "ratio"),
    ("unit_flagged_frac", "FM1 · share of this session's files reading seated", "fraction"),
    ("fm2_smplh_valid_frac", "FM2 · frames with a trusted SMPL-H fit", "percent"),
    ("fm2_longest_invalid_run_s", "FM2 · longest untrusted stretch", "seconds"),
    ("fm3_static_frac", "FM3 · share of the recording with both hands parked", "percent"),
    ("observed_duration_s", "recording length", "clock"),
)

_RENDERED = {"rendered", "reused"}

SignalRows = Sequence[tuple[str, str, str]]


def _signal_table(record: Mapping[str, Any], signal_rows: SignalRows = SIGNAL_ROWS) -> str:
    raw = record.get("signals_json", "")
    signals: dict[str, Any] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                signals = parsed
        except json.JSONDecodeError:
            signals = {}
    rows: list[str] = []
    for key, label, kind in signal_rows:
        if key not in signals or signals[key] is None:
            continue
        value = signals[key]
        if kind == "percent":
            shown = _percent(value)
        elif kind == "seconds":
            number = _number(value)
            if not math.isfinite(number) or number <= 0:
                # A file with no untrusted frames at all. "0.0 s" reads like a
                # measurement that happened to round to zero; "none" is the fact.
                shown = "none"
            else:
                shown = f"{number:.2f} s" if number < 1 else f"{number:.1f} s"
        elif kind == "clock":
            shown = _duration(value)
        elif kind == "fraction":
            shown = _fmt(value, 2)
        elif kind == "text":
            shown = str(value)
        elif kind == "degrees":
            shown = _fmt(value, 1, "°")
        elif kind == "decibels":
            shown = _fmt(value, 1, " dB")
        else:
            shown = _fmt(value)
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(shown)}</td></tr>"
        )
    if not rows:
        return ""
    return f'<table class="signals">{"".join(rows)}</table>'


def _card(record: Mapping[str, Any], signal_rows: SignalRows = SIGNAL_ROWS) -> str:
    media = record.get("media_file")
    if media and record.get("status") in _RENDERED:
        source = html.escape(quote(str(media)))
        buttons = ""
        full = record.get("source_media_file")
        if full:
            name = html.escape(
                str(record.get("source_media_name") or Path(str(full)).name), quote=True
            )
            size = _human_bytes(record.get("source_media_bytes"))
            buttons = (
                f'<a class="download" href="{html.escape(quote(str(full)))}" download="{name}"'
                f' title="The whole recording, not just this excerpt">'
                f'&#x2b07; Full recording <span class="size">{size}</span></a>'
            )
        # Both: `preload="metadata"` on the plain source so the card shows its
        # first frame straight away and the page reads as a contact sheet, and
        # `data-clip` so the script below can swap in an in-memory blob, which is
        # what makes the clip seekable. See the script for why streaming was not
        # enough on its own.
        media_html = (
            f'<video controls preload="metadata" src="{source}" data-clip="{source}"></video>'
            f'<div class="clipbar">{buttons}</div>'
        )
    else:
        reason = record.get("error") or record.get("status") or "media unavailable"
        media_html = f'<div class="missing">No clip: {html.escape(str(reason))}</div>'

    label = str(record.get("label") or "")
    activity = str(record.get("activity_type") or "")
    note = str(record.get("clip_note") or "")
    badge = str(record.get("clip_badge") or "")
    badge_kind = str(record.get("clip_badge_kind") or "neutral")
    return f"""
<article class="card">
  <header>
    <span class="title"><strong>{html.escape(str(record.get('file_id', '')))}</strong>
      <button class="copy" type="button" data-copy="{html.escape(str(record.get('file_id', '')), quote=True)}"
              title="Copy this file ID">&#x29c9;</button></span>
    {f'<span class="badge {html.escape(badge_kind, quote=True)}">{html.escape(badge)}</span>' if badge else ''}
  </header>
  <div class="meta">{html.escape(label)}{' · ' + html.escape(activity) if activity else ''}</div>
  {media_html}
  {f'<p class="note">{html.escape(note)}</p>' if note else ''}
  {_signal_table(record, signal_rows)}
</article>"""


def build_briefing_html(
    sections: Sequence[Mapping[str, Any]],
    *,
    title: str,
    intro: str,
    detectors: Sequence[Mapping[str, str]],
    stats: Mapping[str, Any],
    extra_panels: str = "",
    signal_rows: SignalRows = SIGNAL_ROWS,
    detectors_heading: str = "The checks",
    contents_heading: str = "Contents",
) -> str:
    """Render the briefing page.

    ``sections`` is an ordered list of ``{id, heading, blurb, records}``, each
    optionally carrying ``ask`` (a question put to the reviewer, rendered as a
    callout above the grid) and ``kind`` (``"ask"`` tints the section).
    ``detectors`` is an ordered list of ``{id, name, aim, how, rate}``.
    ``stats`` carries the sample-level numbers rendered above the contents.
    """

    detector_blocks = "".join(
        f"""
<section class="detector" id="about-{html.escape(str(d['id']), quote=True)}">
  <h3>{html.escape(str(d['name']))}<span class="rate">{html.escape(str(d.get('rate', '')))}</span></h3>
  <p class="aim">{d['aim']}</p>
  <p class="how">{d['how']}</p>
</section>"""
        for d in detectors
    )

    # (number, label), which is what every builder has always emitted. This was
    # unpacked the other way round, so the big blue slot showed the caption and
    # the figure appeared in the small grey one.
    stat_cells = "".join(
        f'<div class="stat"><span class="value">{html.escape(str(number))}</span>'
        f'<span class="key">{html.escape(str(label))}</span></div>'
        for number, label in stats["headline"]
    )

    def _entry(section: Mapping[str, Any]) -> str:
        label = str(section.get("short") or section["heading"])
        return (
            f'<li class="{"ask" if section.get("kind") == "ask" else ""}">'
            f'<a href="#{html.escape(str(section["id"]), quote=True)}">'
            f'{html.escape(label)}</a>'
            f'<span class="count">{len(section["records"])} clips</span></li>'
        )

    # Two levels when the sections carry a `group`, one when they do not. The
    # nesting is what makes a 28-section page navigable.
    if any(section.get("group") for section in sections):
        order: list[str] = []
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for section in sections:
            key = str(section.get("group") or "")
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(section)
        blocks = []
        for key in order:
            members = grouped[key]
            total = sum(len(m["records"]) for m in members)
            anchor = members[0]["id"]
            head = (
                f'<li class="toc-group"><a href="#{html.escape(str(anchor), quote=True)}">'
                f'{html.escape(key)}</a>'
                f'<span class="count">{total} clips in {len(members)} sections</span>'
                if key else '<li class="toc-group">'
            )
            blocks.append(head + f'<ol>{"".join(_entry(m) for m in members)}</ol></li>')
        contents = "".join(blocks)
    else:
        contents = "".join(_entry(section) for section in sections)

    body_sections = "".join(
        f"""
<hr class="divider">
<section class="group{' ask' if s.get('kind') == 'ask' else ''}"
         id="{html.escape(str(s['id']), quote=True)}">
  <div class="group-head">
    <h2>{html.escape(str(s['heading']))}</h2>
    <p>{s['blurb']}</p>
    <a class="totop" href="#contents">&#x2191; contents</a>
  </div>
  {f'<div class="askbox"><strong>What I need from you:</strong> {s["ask"]}</div>' if s.get('ask') else ''}
  <div class="grid">{''.join(_card(r, signal_rows) for r in s['records'])}</div>
</section>"""
        for s in sections
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: dark; font-family: system-ui,-apple-system,sans-serif;
  background:#0e1116; color:#e8edf2; }}
body {{ margin:0 auto; padding:1.5rem 1.5rem 5rem; max-width:1600px; line-height:1.55; }}
h1 {{ margin:0 0 .3rem; font-size:1.6rem; }}
h2 {{ margin:0 0 .3rem; font-size:1.25rem; }}
h3 {{ margin:0 0 .35rem; font-size:1rem; display:flex; justify-content:space-between;
  align-items:baseline; gap:1rem; }}
a {{ color:#9ecbff; }}
.lede {{ color:#aeb8c6; max-width:78ch; margin:.2rem 0 1.4rem; }}
.panel {{ background:#161b23; border:1px solid #2c3542; border-radius:10px;
  padding:1rem 1.2rem; margin:0 0 1.2rem; }}
.panel > h2 {{ margin-bottom:.7rem; }}
.detectors {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:1rem; }}
.detector {{ background:#11161d; border:1px solid #2c3542; border-left:3px solid #4d7fb8;
  border-radius:6px; padding:.75rem .9rem; }}
.detector .rate {{ color:#ffd79a; font-size:.85rem; font-weight:600; white-space:nowrap; }}
.detector .aim {{ margin:.2rem 0 .5rem; }}
.detector .how {{ margin:0; color:#9aa6b6; font-size:.87rem; }}
.detector code {{ background:#0b0e13; padding:.05rem .3rem; border-radius:3px; font-size:.85em; }}
.stats {{ display:flex; flex-wrap:wrap; gap:.7rem; }}
.stat {{ background:#11161d; border:1px solid #2c3542; border-radius:6px;
  padding:.55rem .85rem; min-width:8.5rem; }}
.stat .value {{ display:block; font-size:1.35rem; font-weight:600; color:#9ecbff; }}
.stat .key {{ display:block; font-size:.78rem; color:#9aa6b6; }}
table.rates {{ border-collapse:collapse; width:100%; margin-top:.9rem; font-size:.9rem; }}
table.rates th, table.rates td {{ border-bottom:1px solid #2c3542; padding:.35rem .6rem;
  text-align:left; white-space:nowrap; }}
table.rates th {{ color:#9aa6b6; font-weight:500; }}
/* Numeric columns are right-aligned, so their headers must be too, or the
   labels float left of the digits they belong to. */
table.rates th.num, table.rates td.num {{ text-align:right;
  font-variant-numeric:tabular-nums; }}
table.rates tbody th {{ color:#e8edf2; }}
table.rates tr.total td, table.rates tr.total th {{ border-top:2px solid #3a4757;
  font-weight:600; }}
#contents > ol {{ margin:.3rem 0 0; padding-left:1.2rem; }}
#contents ol ol {{ margin:.2rem 0 .6rem; padding-left:1.1rem; }}
#contents li {{ margin:.2rem 0; }}
#contents li.toc-group {{ margin:.6rem 0 .2rem; }}
#contents li.toc-group > a {{ font-weight:600; font-size:1.02rem; }}
#contents .count {{ color:#8f9aab; font-size:.83rem; margin-left:.5rem; }}
.divider {{ border:0; border-top:2px solid #2c3542; margin:2.6rem 0 1.4rem; }}
.group-head {{ position:sticky; top:0; z-index:3; background:#0e1116;
  border-bottom:1px solid #2c3542; padding:.7rem 0 .6rem; margin-bottom:1rem; }}
.group-head p {{ margin:.15rem 0 0; color:#aeb8c6; max-width:82ch; font-size:.92rem; }}
.totop {{ position:absolute; right:0; top:.8rem; font-size:.82rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:1rem; }}
.card {{ background:#161b23; border:1px solid #2c3542; border-radius:8px; padding:.7rem; }}
.card header {{ font-size:.88rem; word-break:break-all; display:flex; gap:.5rem;
  align-items:baseline; justify-content:space-between; }}
.badge {{ flex:0 0 auto; font-size:.72rem; font-weight:600; padding:.1rem .45rem;
  border-radius:4px; white-space:nowrap; background:#243040; color:#b9c8db; }}
.badge.pass {{ background:#16321f; color:#8fe0a6; }}
.badge.flag {{ background:#37211f; color:#ffb0a0; }}
.badge.repair {{ background:#2a2740; color:#c9bcff; }}
#contents li.ask > a {{ color:#ffd79a; font-weight:600; }}
.group.ask .group-head {{ border-bottom-color:#6b5a2f; }}
.askbox {{ background:#231d10; border:1px solid #6b5a2f; border-left:3px solid #d8a94b;
  border-radius:6px; padding:.65rem .9rem; margin:0 0 1rem; color:#f0e2c6; }}
.askbox strong {{ color:#ffd79a; }}
.card .meta {{ color:#8f9aab; font-size:.8rem; margin-bottom:.35rem; }}
.card .note {{ margin:.4rem 0 .2rem; font-size:.85rem; color:#ffd79a; }}
video {{ width:100%; background:#000; border-radius:4px; }}
.missing {{ min-height:8rem; display:grid; place-items:center; background:#281d20;
  color:#ffb8b8; border-radius:4px; }}
.clipbar {{ margin:.4rem 0 .2rem; display:flex; gap:.5rem; align-items:center;
  flex-wrap:wrap; }}
.card header .title {{ display:inline-flex; gap:.35rem; align-items:baseline; }}
.copy {{ background:#243040; color:#b9c8db; border:1px solid #3a475a; border-radius:4px;
  cursor:pointer; font-size:.78rem; line-height:1; padding:.15rem .35rem; }}
.copy:hover {{ background:#2e3c4e; color:#e8edf2; }}
.copy.copied {{ background:#16321f; color:#8fe0a6; border-color:#2c5c3c; }}
.transport {{ border-radius:8px; padding:.7rem 1rem; margin:0 0 1.2rem; font-size:.9rem; }}
.transport.ok {{ background:#12261a; border:1px solid #2c5c3c; color:#a9e8bd; }}
.transport.warn {{ background:#2a2110; border:1px solid #6b5a2f; color:#f0dcb4; }}
.transport code {{ background:#0b0e13; padding:.05rem .3rem; border-radius:3px; }}
.download {{ display:inline-block; padding:.28rem .6rem; border:1px solid #4c586a;
  border-radius:5px; background:#1d2430; color:#cfe3ff; text-decoration:none; font-size:.8rem; }}
.download:hover {{ background:#26303e; border-color:#6d7d96; }}
.download .size {{ color:#8f9aab; }}
table.signals {{ border-collapse:collapse; width:100%; font-size:.82rem; margin-top:.3rem; }}
table.signals th {{ text-align:left; font-weight:400; color:#9aa6b6; padding:.18rem .3rem;
  border-bottom:1px solid #232b36; }}
table.signals td {{ text-align:right; font-variant-numeric:tabular-nums; padding:.18rem .3rem;
  border-bottom:1px solid #232b36; }}
</style></head><body>

<h1>{html.escape(title)}</h1>
<p class="lede">{intro}</p>

<div id="transport" class="transport">Checking whether this server supports range
requests…</div>

<section class="panel">
  <h2>{html.escape(detectors_heading)}</h2>
  <div class="detectors">{detector_blocks}</div>
</section>

<section class="panel">
  <h2>What they did to this sample</h2>
  <div class="stats">{stat_cells}</div>
  {stats['table']}
  <p class="how" style="color:#9aa6b6;font-size:.85rem;margin-top:.8rem;">{stats['footnote']}</p>
</section>

{extra_panels}

<section class="panel" id="contents">
  <h2>{html.escape(contents_heading)}</h2>
  <ol>{contents}</ol>
</section>

{body_sections}

<script>
// Why this exists: dragging the progress bar snapped back to where it was, and
// two rounds of fixes did not clear it.
//
// Round 7 fixed the clip (libx264's default GOP put three keyframes in a
// thirty-second render, so a seek could only land on 0, 10 or 20 s -- now one a
// second) and shipped a range-capable server, because `http.server` answers
// every seek request with 200 and the whole file. Round 8 added full buffering,
// since a clip entirely inside `video.buffered` needs no request to seek at all.
//
// It still failed, with the transport reporting 206 and the clip fully buffered.
// That combination rules out the network and points at the media element's
// resource loader, which is what an embedded webview -- an editor preview pane --
// replaces with its own. Those loaders commonly refuse seeks whatever the server
// does.
//
// So this stops using the loader. Each clip is fetched into memory and handed to
// the element as a blob URL: no ranges, no streaming, no loader, just bytes the
// page already owns. Only clips near the viewport are fetched, and the oldest
// are revoked once past a cap, so a 200-clip page holds a bounded amount.
(function () {{
  var LIVE_CLIPS = 14;                 // about 60 MB at the usual 4 MB a clip
  var loaded = [];

  var release = function () {{
    while (loaded.length > LIVE_CLIPS) {{
      var old = loaded.shift();
      if (old.video.paused) {{
        URL.revokeObjectURL(old.url);
        old.video.src = old.video.dataset.clip;
        old.video.dataset.state = 'idle';
      }} else {{
        loaded.push(old);            // never evict something that is playing
        break;
      }}
    }}
  }};

  var load = function (video) {{
    if (video.dataset.state === 'loading' || video.dataset.state === 'ready') return;
    video.dataset.state = 'loading';
    fetch(video.dataset.clip)
      .then(function (response) {{
        if (!response.ok) throw new Error(response.status);
        return response.blob();
      }})
      .then(function (blob) {{
        var url = URL.createObjectURL(blob);
        var swap = function () {{
          var at = video.currentTime;
          video.src = url;
          // Swapping the source rewinds the element, so put the playhead back.
          if (at > 0) {{
            video.addEventListener('loadedmetadata', function once() {{
              video.removeEventListener('loadedmetadata', once);
              video.currentTime = at;
            }});
          }}
          video.dataset.state = 'ready';
          loaded.push({{ video: video, url: url }});
          release();
        }};
        // Never yank the source out from under something that is playing.
        if (video.paused) {{ swap(); }}
        else {{ video.addEventListener('pause', function once() {{
          video.removeEventListener('pause', once); swap();
        }}); }}
      }})
      .catch(function (error) {{
        // A `file://` page cannot fetch. The plain source is already set, so the
        // clip still plays; seeking then depends on the browser, which is the
        // situation this whole script exists to avoid. Say so rather than fail
        // silently.
        video.dataset.state = 'idle';
      }});
  }};

  var videos = Array.prototype.slice.call(document.querySelectorAll('video[data-clip]'));
  videos.forEach(function (video) {{
    video.dataset.state = 'idle';
    video.addEventListener('play', function () {{ load(video); }});
  }});
  if ('IntersectionObserver' in window) {{
    var watcher = new IntersectionObserver(function (entries) {{
      entries.forEach(function (entry) {{ if (entry.isIntersecting) load(entry.target); }});
    }}, {{ rootMargin: '300px 0px' }});
    videos.forEach(function (video) {{ watcher.observe(video); }});
  }} else {{
    videos.slice(0, LIVE_CLIPS).forEach(load);
  }}

  // Copy a file ID with one click, since every answer comes back as a list of them.
  document.addEventListener('click', function (event) {{
    var button = event.target.closest ? event.target.closest('.copy') : null;
    if (!button) return;
    var text = button.dataset.copy;
    var done = function () {{
      button.classList.add('copied');
      button.textContent = '\u2713';
      setTimeout(function () {{ button.classList.remove('copied'); button.textContent = '\u29c9'; }}, 1200);
    }};
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(done, function () {{ fallback(text, done); }});
    }} else {{
      fallback(text, done);
    }}
  }});
  function fallback(text, done) {{
    var box = document.createElement('textarea');
    box.value = text;
    box.style.position = 'fixed';
    box.style.opacity = '0';
    document.body.appendChild(box);
    box.select();
    try {{ document.execCommand('copy'); done(); }} catch (error) {{ /* nothing else to try */ }}
    document.body.removeChild(box);
  }}

  // Report what the transport did, so this never has to be diagnosed by guesswork.
  var banner = document.getElementById('transport');
  var first = videos[0];
  if (banner && first) {{
    fetch(first.dataset.clip, {{ headers: {{ Range: 'bytes=0-99' }} }})
      .then(function (response) {{
        banner.className = 'transport ok';
        banner.innerHTML = '\u2713 Clips are fetched into memory and played from a '
          + 'blob, so scrubbing does not depend on the transport at all. '
          + '(This one answered a range request with <strong>' + response.status
          + '</strong>.)';
      }})
      .catch(function () {{
        banner.className = 'transport warn';
        banner.innerHTML = '\u26a0 Opened from a <code>file://</code> path, so clips '
          + 'cannot be fetched into memory and scrubbing depends on the browser. '
          + 'Serve the gallery with <code>scripts/serve_gallery.py</code> and open '
          + 'it over <code>http://localhost</code> instead.';
      }});
  }}
}})();
</script>
</body></html>"""


def write_briefing(path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(build_briefing_html(**kwargs), encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
