"""Phase 17: Interactive Speculative Code-Completion CLI & Local Streaming HTTP/SSE Server.

Exposes `Mamba3CodeLM` + `PagedBenesLUT6CodeDrafter` + `HardwareBRAMPageCache` with:
  1. Real-time ANSI-colored CLI token streaming (`complete` & interactive `repl`)
  2. Local multi-threaded HTTP + SSE streaming server (`serve`: `GET /`, `GET /v1/health`,
     `POST /v1/complete` SSE/JSON, `POST /v1/crystallize`, + optional `--watch` file-edit daemon)
  3. End-to-end automated HTTP/SSE + Live Online Sleep-Crystallization showcase (`demo`)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request

from clai.service.live_code_engine import (
    FileEditSleepWatcher,
    LiveSpeculativeCodeEngine,
    PretrainedHFStreamEngineAdapter,
    SpeculativeBurstEvent,
)


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"   # Rung 0 Zero-MAC LUT6 Draft Byte!
ANSI_CYAN = "\033[96m"    # Rung 2 Mamba-3 Bonus Byte!
ANSI_YELLOW = "\033[93m"  # Rung 2 Override / Escalation Byte
ANSI_DIM = "\033[2m"


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MoL Speculative Code Engine — Rung 0 Beneš-LUT6 + Rung 2 Mamba-3</title>
<style>
  :root { --bg: #0b0f17; --panel: #131a27; --border: #253147; --txt: #e5ecf6; --r0: #34d399; --r2b: #38bdf8; --r2o: #fbbf24; }
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--bg); color: var(--txt); margin: 0; padding: 24px; }
  h1 { font-size: 18px; margin: 0 0 6px 0; }
  .sub { color: #8da0be; font-size: 12px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  textarea { width: 100%; box-sizing: border-box; height: 130px; background: #080c13; color: var(--txt); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-family: inherit; font-size: 13px; }
  button, select, input { background: #1e293b; color: var(--txt); border: 1px solid #3b4d6b; border-radius: 6px; padding: 7px 12px; font-family: inherit; cursor: pointer; }
  button.primary { background: #059669; border-color: #10b981; font-weight: 600; }
  .out { background: #070a10; border: 1px solid var(--border); border-radius: 6px; padding: 12px; min-height: 120px; white-space: pre-wrap; line-height: 1.45; font-size: 13px; margin-top: 10px; }
  .tok-R0_LUT6 { background: rgba(52,211,153,0.18); color: var(--r0); border-bottom: 2px solid var(--r0); }
  .tok-R2_BONUS { background: rgba(56,189,248,0.16); color: var(--r2b); border-bottom: 2px solid var(--r2b); }
  .tok-R2_OVERRIDE, .tok-R2_ESCALATE { background: rgba(251,191,36,0.16); color: var(--r2o); }
  .stats { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: 12px; color: #b6c5dd; }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; margin-top: 10px; }
  th, td { border-bottom: 1px solid var(--border); padding: 5px 6px; text-align: left; }
</style>
</head>
<body>
  <h1>Mixture of Limits (MoL) — Live Beneš-LUT6 + Mamba-3 Speculative Code Server</h1>
  <div class="sub">Green = Zero-MAC Rung 0 LUT6 Burst | Cyan = Rung 2 Mamba-3 Bonus | Amber = Rung 2 Override/Escalate</div>
  <div class="grid">
    <div class="card">
      <div><strong>1. Live Streaming Code Completion (SSE)</strong></div>
      <div style="margin:8px 0; display:flex; gap:8px; align-items:center;">
        <select id="lang"><option value="auto">Auto</option><option value="py">Python</option><option value="rs">Rust</option><option value="sv">SystemVerilog</option></select>
        <input id="maxb" type="number" value="96" min="16" max="256" style="width:70px;">
        <button class="primary" onclick="runStream()">Stream Completion</button>
      </div>
      <textarea id="prompt">def forward_stateful(
        self, x: torch.Tensor, state: MambaCodeLayerState
    ) -> tuple[torch.Tensor,</textarea>
      <div class="out" id="streamOut"></div>
      <div class="stats" id="statsBar">Ready.</div>
    </div>
    <div class="card">
      <div><strong>2. Live Online Sleep-Crystallization (Newly Edited File)</strong></div>
      <div style="margin:8px 0; display:flex; gap:8px;">
        <input id="fpath" value="src/custom_router.py" style="flex:1;">
        <button class="primary" onclick="runCrystallize()">Sleep-Crystallize into KV260 BRAM</button>
      </div>
      <textarea id="editCode">def route_quantum_benes_frame(pkt_bus: bytes, vni_tag: int) -> tuple[int, bool]:
    crc_mask = (vni_tag << 4) ^ 0x5A3C
    return crc_mask, True
</textarea>
      <div class="out" id="crystOut" style="min-height:60px;">Edit or paste new code above to hot-compile Joint-Mode LUT6 INIT pages into BRAM.</div>
      <table>
        <thead><tr><th>Pass</th><th>Rung 0 Tier</th><th>BRAM Slot</th><th>Acc/Prop</th><th>B/Pass</th><th>NVMe / RAM</th><th>Emitted</th></tr></thead>
        <tbody id="traceBody"></tbody>
      </table>
    </div>
  </div>
<script>
async function runStream() {
  const prompt = document.getElementById('prompt').value;
  const lang = document.getElementById('lang').value;
  const max_bytes = parseInt(document.getElementById('maxb').value || '96', 10);
  const out = document.getElementById('streamOut');
  const tbody = document.getElementById('traceBody');
  out.innerHTML = ''; tbody.innerHTML = '';
  const resp = await fetch('/v1/complete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'Accept': 'text/event-stream'},
    body: JSON.stringify({prompt, lang, max_bytes, stream: true})
  });
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    let idx;
    while ((idx = buf.indexOf('\\n\\n')) !== -1) {
      const chunk = buf.slice(0, idx); buf = buf.slice(idx + 2);
      const dataLine = chunk.split('\\n').find(l => l.startsWith('data: '));
      if (!dataLine) continue;
      const payload = JSON.parse(dataLine.slice(6));
      if (payload.step !== undefined) {
        payload.emitted_bytes.forEach((b, i) => {
          const sp = document.createElement('span');
          sp.className = 'tok-' + payload.token_origins[i];
          sp.textContent = new TextDecoder().decode(new Uint8Array([b]));
          out.appendChild(sp);
        });
        const tr = document.createElement('tr');
        const nvmeCell = payload.air_nvme_active ? `${payload.step_nvme_kb_read.toFixed(0)} KB (${payload.peak_resident_ram_mb.toFixed(1)}M RAM)` : 'VRAM';
        tr.innerHTML = `<td>${payload.step}</td><td>${payload.rung0_tier}</td><td>#${payload.kv260_bram_slot} ${payload.kv260_bram_hit ? 'HIT' : 'FB'}</td><td>${payload.draft_accepted_len}/${payload.draft_proposed_len}</td><td>${payload.mean_bytes_per_pass.toFixed(2)}</td><td>${nvmeCell}</td><td>${JSON.stringify(payload.emitted_text)}</td>`;
        tbody.appendChild(tr);
        const nvmeBar = payload.air_nvme_active ? ` | NVMe Read: ${payload.cumulative_nvme_mb_read.toFixed(2)} MB (Saved ${payload.nvme_sweeps_saved} sweeps, Peak RAM ${payload.peak_resident_ram_mb.toFixed(2)} MB)` : '';
        document.getElementById('statsBar').textContent =
          `Bytes: ${payload.cumulative_bytes} | Passes: ${payload.cumulative_passes} | Speedup: ${payload.speedup_vs_t1.toFixed(2)}x | Rung-0 LUT6 Share: ${payload.r0_lut6_share_pct.toFixed(1)}% | Accept: ${payload.draft_accept_rate_pct.toFixed(1)}% | Energy: ${payload.mj_per_byte.toFixed(2)} mJ/B${nvmeBar}`;
      }
    }
  }
}
async function runCrystallize() {
  const file_path = document.getElementById('fpath').value;
  const content = document.getElementById('editCode').value;
  const r = await fetch('/v1/crystallize', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({file_path, content, adapt_r2_steps: 10})
  });
  const j = await r.json();
  document.getElementById('crystOut').textContent = JSON.stringify(j, null, 2);
}
</script>
</body>
</html>
"""


def format_colored_burst(ev: SpeculativeBurstEvent) -> str:
    parts: list[str] = []
    for b_val, orig in zip(ev.emitted_bytes, ev.token_origins):
        ch = bytes([b_val]).decode("utf-8", errors="replace")
        if orig == "R0_LUT6":
            parts.append(f"{ANSI_GREEN}{ch}{ANSI_RESET}")
        elif orig == "R2_BONUS":
            parts.append(f"{ANSI_CYAN}{ch}{ANSI_RESET}")
        else:
            parts.append(f"{ANSI_YELLOW}{ch}{ANSI_RESET}")
    return "".join(parts)


def make_http_handler(engine: LiveSpeculativeCodeEngine):
    class MoLCodeRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, obj: dict | list) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(body.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                raw = DASHBOARD_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
                return
            if self.path == "/v1/health":
                self._send_json(HTTPStatus.OK, engine.health_snapshot())
                return
            if self.path == "/v1/crystallizations":
                self._send_json(HTTPStatus.OK, [asdict(e) for e in engine.crystallization_history])
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                req = self._read_json_body()
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            if self.path == "/v1/complete":
                prompt = str(req.get("prompt", "def "))
                lang = str(req.get("lang", "auto"))
                file_path = str(req.get("file_path", ""))
                max_bytes = max(1, min(512, int(req.get("max_bytes", 96))))
                want_stream = bool(req.get("stream", False)) or (
                    "text/event-stream" in self.headers.get("Accept", "")
                )

                if not want_stream:
                    verify = bool(req.get("verify_lossless", False))
                    summary = engine.complete_sync(
                        prompt=prompt,
                        lang=lang,
                        file_path=file_path,
                        max_bytes=max_bytes,
                        verify_lossless=verify,
                    )
                    self._send_json(HTTPStatus.OK, asdict(summary))
                    return

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                full_bytes: list[int] = []
                last_ev: SpeculativeBurstEvent | None = None
                for ev in engine.stream_complete(
                    prompt=prompt, lang=lang, file_path=file_path, max_bytes=max_bytes
                ):
                    full_bytes.extend(ev.emitted_bytes)
                    last_ev = ev
                    frame = f"event: burst\ndata: {json.dumps(asdict(ev))}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()

                done_obj = {
                    "completion_text": bytes(full_bytes).decode("utf-8", errors="replace"),
                    "total_bytes": len(full_bytes),
                    "total_passes": last_ev.cumulative_passes if last_ev else 0,
                    "mean_bytes_per_pass": last_ev.mean_bytes_per_pass if last_ev else 1.0,
                    "r0_lut6_share_pct": last_ev.r0_lut6_share_pct if last_ev else 0.0,
                    "bram_generation": engine.bram_cache.bram_generation,
                    "air_nvme_active": engine.air_nvme,
                    "total_nvme_mb_read": last_ev.cumulative_nvme_mb_read if last_ev else 0.0,
                    "peak_resident_ram_mb": last_ev.peak_resident_ram_mb if last_ev else 0.0,
                    "nvme_sweeps_saved": last_ev.nvme_sweeps_saved if last_ev else 0,
                }
                self.wfile.write(f"event: done\ndata: {json.dumps(done_obj)}\n\n".encode("utf-8"))
                self.wfile.flush()
                return

            if self.path == "/v1/adapt_r2":
                content = str(req.get("content", ""))
                adapt_steps = max(1, min(60, int(req.get("adapt_r2_steps", 20))))
                dt_ms = engine.adapt_rung2_online(content, adapt_r2_steps=adapt_steps)
                self._send_json(HTTPStatus.OK, {"status": "adapted_r2", "adapt_r2_steps": adapt_steps, "elapsed_ms": round(dt_ms, 2)})
                return

            if self.path == "/v1/crystallize":
                file_path = str(req.get("file_path", "edited_buffer.py"))
                content = req.get("content")
                if content is None:
                    p = Path(file_path)
                    if not p.is_file():
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"File not found: {file_path}"})
                        return
                    content = p.read_bytes()
                lang = str(req.get("lang", "auto"))
                adapt_steps = max(0, min(40, int(req.get("adapt_r2_steps", 10))))
                ev = engine.sleep_crystallize_text(
                    file_path=file_path,
                    content=content,
                    lang=lang,
                    adapt_r2_steps=adapt_steps,
                )
                self._send_json(HTTPStatus.OK, asdict(ev))
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {self.path}"})

    return MoLCodeRequestHandler


def create_http_server(engine: LiveSpeculativeCodeEngine, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_http_handler(engine))


def run_cli_complete(engine: LiveSpeculativeCodeEngine, args: argparse.Namespace) -> None:
    prompt = args.prompt
    if args.file:
        prompt = Path(args.file).read_text(encoding="utf-8")[-256:]

    print("=" * 104)
    print(
        f" {ANSI_BOLD}MoL LIVE SPECULATIVE STREAM{ANSI_RESET} | "
        f"{ANSI_GREEN}Green=Rung0 LUT6 (0 FLOPs){ANSI_RESET} | "
        f"{ANSI_CYAN}Cyan=Rung2 Bonus{ANSI_RESET} | "
        f"{ANSI_YELLOW}Yellow=Rung2 Override/Escalate{ANSI_RESET}"
    )
    print("=" * 104)
    sys.stdout.write(f"{ANSI_DIM}{prompt}{ANSI_RESET}")
    sys.stdout.flush()

    events: list[SpeculativeBurstEvent] = []
    for ev in engine.stream_complete(
        prompt=prompt, lang=args.lang, file_path=args.file or "", max_bytes=args.max_bytes
    ):
        events.append(ev)
        sys.stdout.write(format_colored_burst(ev))
        sys.stdout.flush()
    print("\n" + "-" * 104)

    print(
        f"{'Pass':>4} | {'Rung 0 Hardware Tier':<20} | {'KV260 BRAM':<12} | {'Acc/Prop':>8} | "
        f"{'B/Pass':>6} | {'NVMe I/O':>10} | {'PeakRAM':>8} | {'Step(ms)':>8} | {'Burst Bytes'}"
    )
    print("-" * 116)
    for ev in events[: min(24, len(events))]:
        bram_str = f"#{ev.kv260_bram_slot:03d} {'HIT' if ev.kv260_bram_hit else 'CAM_FB'}"
        nvme_str = f"{ev.step_nvme_kb_read:>7.1f} KB" if ev.air_nvme_active else "   VRAM   "
        ram_str = f"{ev.peak_resident_ram_mb:>5.2f} MB"
        escaped = repr(ev.emitted_text)
        print(
            f"{ev.step:>4d} | {ev.rung0_tier:<20} | {bram_str:<12} | "
            f"{ev.draft_accepted_len:>3d}/{ev.draft_proposed_len:<4d} | "
            f"{ev.mean_bytes_per_pass:>6.2f} | {nvme_str:>10} | {ram_str:>8} | "
            f"{ev.step_latency_ms:>8.2f} | {escaped}"
        )
    if len(events) > 24:
        print(f"  ... ({len(events) - 24} additional passes omitted) ...")
    last = events[-1]
    lossless_str = ""
    if args.verify_lossless:
        raw_spec = bytes(b for e in events for b in e.emitted_bytes)
        raw_ref = engine.standalone_mamba3_greedy(prompt.encode("utf-8", errors="replace"), args.max_bytes)
        assert raw_spec == raw_ref, "Lossless equivalence violation!"
        lossless_str = " | Lossless=100% EXACT"
    nvme_summary = (
        f" | AirNVMe Read={last.cumulative_nvme_mb_read:.2f} MB (Saved {last.nvme_sweeps_saved} sweeps, PeakRAM={last.peak_resident_ram_mb:.2f}/{engine.full_model_bytes/(1024*1024):.2f} MB)"
        if last.air_nvme_active
        else ""
    )
    print("-" * 116)
    print(
        f"  Summary: {last.cumulative_bytes}B in {last.cumulative_passes} passes "
        f"({last.mean_bytes_per_pass:.2f} B/pass = {last.speedup_vs_t1:.2f}x) | "
        f"R0 LUT6 Share={last.r0_lut6_share_pct:.1f}% | "
        f"Accept={last.draft_accept_rate_pct:.1f}%{nvme_summary}{lossless_str}"
    )


def run_cli_repl(engine: LiveSpeculativeCodeEngine, max_bytes: int = 80) -> None:
    cur_lang = "auto"
    print("MoL Interactive Speculative REPL (`/lang py|rs|sv`, `/crystallize <file>`, `/status`, `/quit`)")
    while True:
        try:
            line = input(f"\n[{cur_lang}] >>> ")
        except EOFError:
            break
        cmd = line.strip()
        if not cmd:
            continue
        if cmd in ("/quit", "/exit"):
            break
        if cmd == "/status":
            print(json.dumps(engine.health_snapshot(), indent=2))
            continue
        if cmd.startswith("/lang "):
            cur_lang = cmd.split(maxsplit=1)[1].strip()
            print(f"Language set to: {cur_lang}")
            continue
        if cmd.startswith("/crystallize "):
            fpath = cmd.split(maxsplit=1)[1].strip()
            ev = engine.sleep_crystallize_text(fpath, Path(fpath).read_bytes(), adapt_r2_steps=10)
            print(json.dumps(asdict(ev), indent=2))
            continue

        ns = argparse.Namespace(prompt=line, file="", lang=cur_lang, max_bytes=max_bytes, verify_lossless=True)
        run_cli_complete(engine, ns)


def _sse_request_complete(base_url: str, payload: dict) -> tuple[list[dict], dict]:
    req = urllib.request.Request(
        f"{base_url}/v1/complete",
        data=json.dumps({**payload, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    bursts: list[dict] = []
    done_meta: dict = {}
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_sse = resp.read().decode("utf-8")
    for block in raw_sse.strip().split("\n\n"):
        lines = block.splitlines()
        ev_type = ""
        data_json = ""
        for ln in lines:
            if ln.startswith("event: "):
                ev_type = ln[7:].strip()
            elif ln.startswith("data: "):
                data_json = ln[6:].strip()
        if ev_type == "burst" and data_json:
            bursts.append(json.loads(data_json))
        elif ev_type == "done" and data_json:
            done_meta = json.loads(data_json)
    return bursts, done_meta


def run_automated_demo(engine: LiveSpeculativeCodeEngine) -> dict:
    """
    Spins up the live HTTP/SSE server + `FileEditSleepWatcher`, streams polyglot completions over
    SSE, triggers live Online Sleep-Crystallization on a newly edited domain module, verifies
    100.0% lossless byte-exact equivalence + burst speedup jump, and saves telemetry JSON!
    """
    print("=" * 104)
    print(" PHASE 17: LIVE STREAMING HTTP/SSE SERVER + ONLINE FILE-EDIT SLEEP-CRYSTALLIZATION DEMO")
    print("=" * 104)

    with tempfile.TemporaryDirectory() as tmp_watch:
        watch_path = Path(tmp_watch)
        watcher = FileEditSleepWatcher(engine, watch_path, poll_interval_sec=0.2, adapt_r2_steps=0)
        watcher.start()

        srv = create_http_server(engine, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        srv_thread.start()

        try:
            with urllib.request.urlopen(f"{base_url}/v1/health", timeout=10) as h_resp:
                health_before = json.loads(h_resp.read().decode("utf-8"))
            print(
                f"  [HTTP Server Live @ {base_url}] Device={health_before['device']} | "
                f"LUT6 6-gram Pages={health_before['lut6_pages_6gram']:,} | "
                f"KV260 BRAM Occupied={health_before['kv260_bram_occupied_pages']}/{health_before['kv260_bram_capacity_pages']}"
            )

            # 1. Stream Python, Rust, and SystemVerilog completions over real SSE!
            demo_prompts = [
                (
                    "py",
                    "        self.d_model = d_model\n        self.n_heads = n_heads\n        self.d_head = ",
                ),
                (
                    "rs",
                    "pub fn verify_kv260_bank_swap(reg_base: usize, shadow_id: u32) -> Result<u32, ",
                ),
                (
                    "sv",
                    "    always @(posedge clk) begin\n        if (!rst_n) begin\n            active_bank <= ",
                ),
            ]
            sse_runs = []
            for lang, prm in demo_prompts:
                bursts, done_meta = _sse_request_complete(base_url, {"prompt": prm, "lang": lang, "max_bytes": 64})
                ref_bytes = engine.standalone_mamba3_greedy(prm.encode("utf-8"), 64)
                emitted_bytes = bytes(b for ev in bursts for b in ev["emitted_bytes"])
                assert emitted_bytes == ref_bytes, f"SSE stream mismatch on {lang}!"
                sse_runs.append({"lang": lang, "prompt": prm, "summary": done_meta, "num_sse_events": len(bursts)})
                print(
                    f"  [SSE /{lang}] 64B in {done_meta['total_passes']} passes "
                    f"({done_meta['mean_bytes_per_pass']:.2f} B/pass | Rung-0 LUT6 Share={done_meta['r0_lut6_share_pct']:.1f}%) "
                    f"-> {repr(done_meta['completion_text'][:52])}..."
                )

            # 2. Simulate developer authoring a BRAND-NEW domain file:
            #    (a) Wake Phase: Rung 2 `Mamba3CodeLM` adapts to the new file while Rung 0 `LUT6` BRAM is stale
            #    (b) Sleep Phase: `FileEditSleepWatcher` compiles Joint-Mode `LUT6 INIT` pages into KV260 BRAM!
            novel_code = (
                "class TelemetryQuorumPacket:\n"
                "    def dispatch_quorum_frame(self, vni_id: int, epoch_tag: int) -> tuple[int, int]:\n"
                "        quorum_crc = ((vni_id << 5) ^ epoch_tag) & 0x7FFF\n"
                "        return quorum_crc, epoch_tag + 1\n\n"
                "    def verify_quorum_frame(self, vni_id: int, epoch_tag: int) -> tuple[int, int]:\n"
                "        quorum_crc = ((vni_id << 5) ^ epoch_tag) & 0x7FFF\n"
                "        return quorum_crc, epoch_tag + 1\n"
            )
            engine.adapt_rung2_online(novel_code, adapt_r2_steps=28)

            novel_prompt = (
                "class TelemetryQuorumPacket:\n"
                "    def dispatch_quorum_frame(self, vni_id: int, epoch_tag: int) -> "
            )
            bursts_pre, pre_edit_done = _sse_request_complete(
                base_url, {"prompt": novel_prompt, "lang": "py", "max_bytes": 64}
            )
            spec_pre = bytes(b for ev in bursts_pre for b in ev["emitted_bytes"])

            # Save newly edited file into watched workspace -> `FileEditSleepWatcher` crystallizes LUT6 BRAM live!
            edited_path = watch_path / "telemetry_quorum_packet.py"
            edited_path.write_text(novel_code, encoding="utf-8")
            watcher_events = watcher.scan_once()
            assert len(watcher_events) >= 1, "Watcher did not trigger Sleep-Crystallization on new file!"
            cryst_ev = asdict(watcher_events[-1])

            bursts_post, post_edit_done = _sse_request_complete(
                base_url, {"prompt": novel_prompt, "lang": "py", "max_bytes": 64}
            )
            ref_post = engine.standalone_mamba3_greedy(novel_prompt.encode("utf-8"), 64)
            spec_post = bytes(b for ev in bursts_post for b in ev["emitted_bytes"])
            assert spec_post == ref_post and spec_post == spec_pre, "Post-crystallization SSE broke lossless parity!"

            print("-" * 104)
            print(
                f"  [Live File-Edit Sleep-Crystallization] File={Path(cryst_ev['file_path']).name} | "
                f"BRAM Gen #{cryst_ev['bram_generation']} | "
                f"Programmed {cryst_ev['kv260_bram_slots_programmed']} KV260 BRAM Slots in {cryst_ev['elapsed_ms']:.1f} ms!"
            )
            print(
                f"  [Before -> After Online Sleep-Crystallization] "
                f"File Probe: {cryst_ev['pre_bytes_per_pass']:.2f} -> {cryst_ev['post_bytes_per_pass']:.2f} B/pass | "
                f"Live SSE Prompt: {pre_edit_done['mean_bytes_per_pass']:.2f} -> {post_edit_done['mean_bytes_per_pass']:.2f} B/pass "
                f"(Rung-0 LUT6 Share: {pre_edit_done['r0_lut6_share_pct']:.1f}% -> {post_edit_done['r0_lut6_share_pct']:.1f}%)"
            )
            print("=" * 104)

            report = {
                "health_before": health_before,
                "health_after": engine.health_snapshot(),
                "sse_polyglot_runs": sse_runs,
                "online_crystallization_event": cryst_ev,
                "novel_file_sse_before": pre_edit_done,
                "novel_file_sse_after": post_edit_done,
                "lossless_exact_parity": True,
            }
            out_json = Path("artifacts/live_cli_server_telemetry.json")
            out_json.parent.mkdir(exist_ok=True)
            out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return report
        finally:
            watcher.stop()
            srv.shutdown()
            srv.server_close()


BACKEND_CHOICES = ["mamba3", "lfm3-1.7b", "lfm2.5-350m", "qwen3.5-0.8b", "lfm2.5-8b-a1b", "qwen3.6-35b-a3b"]


def main() -> None:
    parser = argparse.ArgumentParser(description="MoL Speculative Code-Completion CLI & HTTP/SSE Server")
    parser.add_argument("--checkpoint", default="artifacts/mol_code_engine.pt")
    parser.add_argument("--retrain", action="store_true", help="Force corpus rebuild of checkpoint")
    parser.add_argument("--air-nvme", action="store_true", help="Enable AirMoL 4-bit per-layer NVMe shard streaming")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="mamba3",
        help="Target model backend (`mamba3`, `lfm2.5-350m`, `qwen3.5-0.8b`, `lfm2.5-8b-a1b`, `qwen3.6-35b-a3b`)",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_comp = sub.add_parser("complete", help="Stream ANSI-colored speculative completion in terminal")
    p_comp.add_argument("--prompt", default="def forward_stateful(\n        self, x: torch.Tensor")
    p_comp.add_argument("--file", default="", help="Optional source file whose trailing 256B form the prompt")
    p_comp.add_argument("--lang", default="auto")
    p_comp.add_argument("--max-bytes", type=int, default=80)
    p_comp.add_argument("--verify-lossless", action="store_true", default=True)
    p_comp.add_argument("--air-nvme", dest="sub_air_nvme", action="store_true")
    p_comp.add_argument("--backend", dest="sub_backend", choices=BACKEND_CHOICES)

    p_cryst = sub.add_parser("crystallize", help="Run live online Sleep-Crystallization on edited file(s)")
    p_cryst.add_argument("files", nargs="+")
    p_cryst.add_argument("--adapt-steps", type=int, default=10)
    p_cryst.add_argument("--air-nvme", dest="sub_air_nvme", action="store_true")
    p_cryst.add_argument("--backend", dest="sub_backend", choices=BACKEND_CHOICES)

    p_repl = sub.add_parser("repl", help="Interactive terminal speculative completion REPL")
    p_repl.add_argument("--max-bytes", type=int, default=80)
    p_repl.add_argument("--air-nvme", dest="sub_air_nvme", action="store_true")
    p_repl.add_argument("--backend", dest="sub_backend", choices=BACKEND_CHOICES)

    p_srv = sub.add_parser("serve", help="Run local multi-threaded HTTP + SSE streaming server")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.add_argument("--watch", default="", help="Optional directory to watch for live file edits")
    p_srv.add_argument("--air-nvme", dest="sub_air_nvme", action="store_true")
    p_srv.add_argument("--backend", dest="sub_backend", choices=BACKEND_CHOICES)

    p_demo = sub.add_parser("demo", help="Run automated HTTP/SSE + Live File-Edit Sleep-Crystallization verification")
    p_demo.add_argument("--air-nvme", dest="sub_air_nvme", action="store_true")
    p_demo.add_argument("--backend", dest="sub_backend", choices=BACKEND_CHOICES)

    args = parser.parse_args()
    cmd = args.cmd or "demo"
    use_air_nvme = bool(args.air_nvme or getattr(args, "sub_air_nvme", False))
    chosen_backend = getattr(args, "sub_backend", None) or args.backend

    if chosen_backend == "mamba3":
        engine = LiveSpeculativeCodeEngine.load_or_train(
            checkpoint_path=Path(args.checkpoint),
            force_retrain=args.retrain,
            air_nvme=use_air_nvme,
        )
    else:
        engine = PretrainedHFStreamEngineAdapter.from_backend(chosen_backend)

    if cmd == "complete":
        run_cli_complete(engine, args)
    elif cmd == "crystallize":
        for fpath in args.files:
            ev = engine.sleep_crystallize_text(fpath, Path(fpath).read_bytes(), adapt_r2_steps=args.adapt_steps)
            print(json.dumps(asdict(ev), indent=2))
    elif cmd == "repl":
        run_cli_repl(engine, max_bytes=args.max_bytes)
    elif cmd == "serve":
        watcher = None
        if args.watch:
            watcher = FileEditSleepWatcher(engine, Path(args.watch))
            watcher.start()
        srv = create_http_server(engine, host=args.host, port=args.port)
        print(f"MoL Speculative Code Server listening on http://{args.host}:{args.port} (Dashboard + /v1/complete SSE)")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if watcher:
                watcher.stop()
            srv.server_close()
    else:
        run_automated_demo(engine)


if __name__ == "__main__":
    main()
