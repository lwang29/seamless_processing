"""The review app: a keyboard-driven local page that writes verdicts to disk.

``seamless-curation review`` starts it on ``127.0.0.1`` and nothing else can
reach it; participant media stays on the cluster and is reached over an SSH port
forward. It exists because the v0 galleries could not collect a decision: the
note-taking one wrote free text into browser ``localStorage`` keyed by a hash
that changed whenever anything was re-rendered, and the briefing pages that
replaced it collect nothing at all by design.

What makes it fast enough to review thousands of items:

* **One item on screen, keys not clicks.** ``A`` accept, ``R`` reject, ``U``
  unsure, digits pick a reject reason, arrows move, ``V`` plays the clip. A
  verdict auto-advances.
* **The card first, the video on demand.** The card answers most items in a
  couple of seconds. The 30-second clip with audio is one keystroke away for the
  ones it does not, and the app records ``saw_video`` so the manifest can say
  which verdicts were taken with sound.
* **Server-side persistence, immediately.** Every verdict is POSTed and appended
  to the JSONL log before the UI advances. Closing the tab loses nothing, and a
  second reviewer on another port appends to the same log.
* **Resumable.** The queue opens at the first unreviewed item.
* **Range requests.** ``http.server`` answers a seek with ``200`` and the whole
  file, which browsers read as "not seekable"; this server answers ``206``, so
  the scrubber works.
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

import pandas as pd

from .config import RunConfig
from .review_store import REJECT_REASONS, VERDICTS, Verdict, VerdictStore

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")

ITEM_COLUMNS = [
    "review_item_id", "file_id", "vendor", "label", "split", "interaction_type",
    "participant_id", "clips", "clip_seconds", "item_score",
    "gesture_frac_speech", "speech_segments_covered", "wrist_excursion_p90_mm",
    "elbow_excursion_p90_mm", "gesture_speech_ratio", "sync_r", "consistency_r",
    "smplh_valid_frac", "hand_frozen_frac", "posture_spread_mm",
    "wrist_height_p75_mm", "hands_together_frac", "arm_abduction_p75_deg",
]


def build_queue(config: RunConfig) -> list[dict[str, Any]]:
    """Review items in review order, annotated with which artefacts exist."""

    items = pd.read_csv(config.review_manifest_path)
    columns = [column for column in ITEM_COLUMNS if column in items.columns]
    queue: list[dict[str, Any]] = []
    for record in items[columns].to_dict("records"):
        stem = str(record["review_item_id"])
        card = config.media_root / f"{stem}.card.png"
        clip = config.media_root / f"{stem}.clip.mp4"
        if not card.exists():
            continue
        record["card"] = card.name
        record["clip"] = clip.name if clip.exists() else None
        sidecar = config.media_root / f"{stem}.json"
        record["card_fingerprint"] = ""
        if sidecar.exists():
            try:
                record["card_fingerprint"] = str(
                    json.loads(sidecar.read_text(encoding="utf-8")).get("card_fingerprint") or ""
                )
            except (OSError, ValueError):
                pass
        queue.append(record)
    return queue


class _Handler(BaseHTTPRequestHandler):
    server_version = "seamless-review/1.0"
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, state: "_State", **kwargs: Any) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover - noise
        return

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self._send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/queue":
            self._send_json({
                "items": self.state.queue,
                "verdicts": self.state.verdict_map(),
                "reasons": list(REJECT_REASONS),
                "run_id": self.state.config.run_id,
                "reviewer": self.state.reviewer,
            })
        elif path == "/api/stats":
            self._send_json(self.state.store.counts())
        elif path.startswith("/media/"):
            self._send_media(path[len("/media/") :])
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    # ----------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if unquote(urlparse(self.path).path) != "/api/verdict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "malformed JSON")
            return
        try:
            verdict = Verdict(
                review_item_id=str(payload["review_item_id"]),
                file_id=str(payload.get("file_id") or ""),
                verdict=str(payload["verdict"]),
                reviewer=str(payload.get("reviewer") or self.state.reviewer),
                verdict_source="human",
                reasons=tuple(payload.get("reasons") or ()),
                note=str(payload.get("note") or ""),
                card_fingerprint=str(payload.get("card_fingerprint") or ""),
                saw_video=bool(payload.get("saw_video")),
            )
        except (KeyError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self.state.store.append(verdict)
        self.state.remember(verdict)
        self._send_json({"ok": True, "counts": self.state.store.counts()})

    # -------------------------------------------------------------- helpers
    def _send_json(self, payload: Any) -> None:
        self._send_bytes(json.dumps(payload, default=str).encode("utf-8"), "application/json")

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, name: str) -> None:
        root = self.state.config.media_root.resolve()
        target = (root / name).resolve()
        # is_relative_to, not a string prefix: "<root>/clips" is a prefix of
        # "<root>/clips_private", so a prefix test serves any sibling directory
        # whose name starts with the media directory's name.
        if not target.is_relative_to(root) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = target.stat().st_size
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        header = self.headers.get("Range", "")
        match = _RANGE.fullmatch(header.strip()) if header else None
        if match and size == 0:
            # A zero-byte file has no satisfiable range. Answering 206 with a
            # promised byte that never arrives desynchronises a keep-alive
            # connection permanently.
            match = None
        if not match:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(target.read_bytes())
            return
        start_text, stop_text = match.groups()
        if start_text:
            start = int(start_text)
            stop = int(stop_text) if stop_text else size - 1
        else:  # suffix range: the last N bytes
            start = max(0, size - int(stop_text or 0))
            stop = size - 1
        if start >= size or stop < start:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        stop = min(stop, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.send_header("Content-Length", str(stop - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with target.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(stop - start + 1))


class _State:
    def __init__(self, config: RunConfig, reviewer: str) -> None:
        self.config = config
        self.reviewer = reviewer
        self.store = VerdictStore(config.verdict_log)
        self.queue = build_queue(config)
        self._verdicts: dict[str, dict[str, Any]] = {}
        resolved = self.store.resolve()
        for record in resolved.to_dict("records"):
            self._verdicts[str(record["review_item_id"])] = {
                "verdict": record["verdict"],
                "reviewer": record["reviewer"],
                "reasons": record.get("reasons") or [],
            }
        self._lock = threading.Lock()

    def verdict_map(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._verdicts)

    def remember(self, verdict: Verdict) -> None:
        with self._lock:
            self._verdicts[verdict.review_item_id] = {
                "verdict": verdict.verdict,
                "reviewer": verdict.reviewer,
                "reasons": list(verdict.reasons),
            }


def serve(
    config: RunConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    reviewer: str = "",
    open_browser: bool = False,
) -> None:
    import getpass

    state = _State(config, reviewer or getpass.getuser())
    if not state.queue:
        raise SystemExit(
            f"no rendered cards under {config.media_root}; run `seamless-curation render` first"
        )
    server = ThreadingHTTPServer((host, port), partial(_Handler, state=state))
    reviewed = len(state.verdict_map())
    print(
        f"review: {len(state.queue)} items rendered, {reviewed} already judged\n"
        f"        http://{host}:{port}/   (ssh -N -L {port}:localhost:{port} <cluster-host>)\n"
        f"        verdicts append to {config.verdict_log}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; verdicts are on disk")
    finally:
        server.server_close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Seamless co-speech review</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111; color:#eee; font:13px/1.45 ui-monospace,Menlo,Consolas,monospace; }
 header { display:flex; gap:18px; align-items:center; padding:6px 12px; background:#191919;
          border-bottom:1px solid #2c2c2c; position:sticky; top:0; z-index:5; flex-wrap:wrap; }
 header b { color:#fff; } .muted { color:#8d8d8d; }
 #card { display:block; max-width:100%; height:auto; }
 #stage { padding:8px 12px 60px; }
 #videoWrap { display:none; margin-top:10px; }
 #videoWrap.on { display:block; }
 video { width:860px; max-width:100%; background:#000; }
 .pill { padding:1px 7px; border-radius:9px; background:#262626; }
 .accept { background:#1d4a22; color:#b8f0bd; } .reject { background:#4d1e1e; color:#f3b9b9; }
 .unsure { background:#4a411c; color:#f0e3b8; }
 kbd { background:#2a2a2a; border:1px solid #3a3a3a; border-radius:4px; padding:0 5px; }
 #help { position:fixed; bottom:0; left:0; right:0; background:#181818; border-top:1px solid #2c2c2c;
         padding:5px 12px; display:flex; gap:14px; flex-wrap:wrap; }
 #reasons { display:none; gap:8px; } #reasons.on { display:flex; }
 #note { width:340px; background:#222; color:#eee; border:1px solid #3a3a3a; padding:2px 6px; }
</style></head><body>
<header>
  <b id="pos">-</b><span id="who" class="muted"></span>
  <span id="meta"></span>
  <span id="verdict" class="pill">unjudged</span>
  <span id="counts" class="muted"></span>
  <input id="note" placeholder="note (optional)">
</header>
<div id="stage"><img id="card" alt="review card"><div id="videoWrap"><video id="clip" controls preload="none"></video></div></div>
<div id="help">
  <span><kbd>A</kbd> accept</span><span><kbd>R</kbd> reject</span><span><kbd>U</kbd> unsure</span>
  <span><kbd>V</kbd> video</span><span><kbd>&larr;</kbd><kbd>&rarr;</kbd> move</span>
  <span><kbd>N</kbd> next unjudged</span><span><kbd>G</kbd> go to #</span>
  <span id="reasons"></span>
</div>
<script>
let items=[], verdicts={}, reasons=[], reviewer="", index=0, pendingReject=false, sawVideo=false;

async function boot(){
  const data = await (await fetch('/api/queue')).json();
  items = data.items; verdicts = data.verdicts; reasons = data.reasons; reviewer = data.reviewer;
  document.getElementById('who').textContent = reviewer + ' @ ' + data.run_id;
  document.getElementById('reasons').innerHTML =
    reasons.map((r,i)=>`<span><kbd>${i+1}</kbd> ${r}</span>`).join(' ');
  index = Math.max(0, items.findIndex(it => !verdicts[it.review_item_id]));
  if (index < 0) index = 0;
  show();
}

function pct(x){ return x==null||isNaN(x) ? '-' : Math.round(x*100)+'%'; }
function mm(x){ return x==null||isNaN(x) ? '-' : Math.round(x)+'mm'; }

function show(){
  const it = items[index]; if(!it) return;
  sawVideo = false; pendingReject = false;
  document.getElementById('reasons').classList.remove('on');
  document.getElementById('pos').textContent = `${index+1}/${items.length}`;
  document.getElementById('meta').innerHTML =
    `<span class="muted">${it.review_item_id}</span> ${it.file_id} ` +
    `<span class="pill">${it.vendor} ${it.label}</span> ` +
    `<span class="pill">${it.clips} clips / ${Math.round(it.clip_seconds)}s</span> ` +
    `<span class="muted">gest-in-speech ${pct(it.gesture_frac_speech)} · covered ${pct(it.speech_segments_covered)} · ` +
    `spread ${mm(it.posture_spread_mm)} · wrist-h ${mm(it.wrist_height_p75_mm)} · ` +
    `clasped ${pct(it.hands_together_frac)} · abduction ${Math.round(it.arm_abduction_p75_deg||0)}&deg;</span>`;
  document.getElementById('card').src = '/media/' + it.card + '?v=' + index;
  const wrap = document.getElementById('videoWrap'), clip = document.getElementById('clip');
  wrap.classList.remove('on'); clip.pause(); clip.removeAttribute('src'); clip.load();
  document.getElementById('note').value = '';
  paintVerdict();
  window.scrollTo(0,0);
}

function paintVerdict(){
  const it = items[index], v = verdicts[it.review_item_id], el = document.getElementById('verdict');
  el.className = 'pill ' + (v ? v.verdict : '');
  el.textContent = v ? `${v.verdict}${(v.reasons&&v.reasons.length)?' · '+v.reasons.join(','):''} (${v.reviewer})` : 'unjudged';
}

let busy = false;
async function send(verdict, reasonList){
  // Browser key auto-repeat fires keydown every ~30 ms while POST latency is
  // ~100 ms. Without this guard a held key writes several verdicts against one
  // item and scrolls the next few past the reviewer with none at all.
  if (busy) return;
  busy = true;
  const it = items[index];
  const body = { review_item_id: it.review_item_id, file_id: it.file_id, verdict,
                 reasons: reasonList||[], card_fingerprint: it.card_fingerprint || '',
                 note: document.getElementById('note').value, reviewer, saw_video: sawVideo };
  let res;
  try {
    res = await (await fetch('/api/verdict', {method:'POST', headers:{'Content-Type':'application/json'},
                                              body: JSON.stringify(body)})).json();
  } catch (err) {
    busy = false; alert('verdict not saved: ' + err); return;
  }
  verdicts[it.review_item_id] = { verdict, reviewer, reasons: reasonList||[] };
  const c = res.counts; document.getElementById('counts').textContent =
    `accept ${c.accept} · reject ${c.reject} · unsure ${c.unsure}`;
  paintVerdict();
  if (index < items.length - 1) { index++; show(); }
  busy = false;
}

function toggleVideo(){
  const it = items[index]; if(!it.clip){ return; }
  const wrap = document.getElementById('videoWrap'), clip = document.getElementById('clip');
  if (wrap.classList.contains('on')) { clip.pause(); wrap.classList.remove('on'); return; }
  clip.src = '/media/' + it.clip; wrap.classList.add('on'); sawVideo = true; clip.play();
}

document.addEventListener('keydown', ev => {
  if (ev.repeat) return;
  if (ev.target.id === 'note' && ev.key !== 'Escape') return;
  const key = ev.key.toLowerCase();
  if (pendingReject && /^[1-9]$/.test(key)) {
    const pick = reasons[parseInt(key,10)-1]; if (pick) { send('reject', [pick]); } return;
  }
  if (key === 'a') { send('accept'); }
  else if (key === 'r') { pendingReject = true; document.getElementById('reasons').classList.add('on'); }
  else if (key === 'u') { send('unsure'); }
  else if (key === 'v') { toggleVideo(); }
  else if (key === 'arrowleft') { if (index>0){ index--; show(); } }
  else if (key === 'arrowright') { if (index<items.length-1){ index++; show(); } }
  else if (key === 'n') { const at = items.findIndex((it,i)=> i>index && !verdicts[it.review_item_id]);
                          if (at>=0){ index=at; show(); } }
  else if (key === 'g') { const to = prompt('go to item number'); const n=parseInt(to,10);
                          if(n>=1 && n<=items.length){ index=n-1; show(); } }
  else if (key === 'escape') { pendingReject=false; document.getElementById('reasons').classList.remove('on'); }
});
boot();
</script></body></html>
"""
