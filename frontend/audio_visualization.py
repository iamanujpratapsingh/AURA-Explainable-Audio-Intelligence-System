import base64
import json

import streamlit as st
import streamlit.components.v1 as components


def render_audio_visualization(audio_bytes, mime_type, detections=None):
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    event_data = [
        {
            "start": item.get("start", 0),
            "name": item.get("event_name", "Detected phrase"),
            "phrase": item.get("matched_phrase", ""),
            "status": item.get("status", "detected"),
            "is_alert": item.get("is_alert", False),
        }
        for item in (detections or [])
    ]
    encoded_events = base64.b64encode(
        json.dumps(event_data, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; color: #1c2922; background: #f4f7f5; font: 14px/1.45 "Segoe UI", sans-serif; }}
  .panel {{ border: 1px solid #d5dfd8; padding: 16px; background: #ffffff; }}
  .heading {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 12px; }}
  .heading strong {{ font-size: 15px; }}
  .heading span, .hint {{ color: #5b6b61; font-size: 12px; }}
  canvas {{ display: block; width: 100%; height: 156px; background: #f6f9f7; border: 1px solid #d5dfd8; }}
  .readout {{ display: flex; justify-content: space-between; color: #52635a; font-variant-numeric: tabular-nums; margin: 7px 0 12px; }}
  audio {{ width: 100%; height: 38px; }}
  .events {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
  .events button {{ border: 1px solid #cbd7ce; border-left: 3px solid #c3342d; background: #f6f9f7; color: #1c2922; padding: 7px 10px; cursor: pointer; text-align: left; }}
  .events button.review {{ border-left-color: #a66a00; }}
  .events button:hover, .events button:focus-visible {{ outline: 2px solid #16745b; outline-offset: 1px; }}
  .empty {{ color: #5b6b61; padding: 8px 0; }}
  .hint {{ margin-top: 10px; }}
</style>
</head>
<body>
<section class="panel" aria-label="Audio amplitude waveform">
  <div class="heading"><strong>Amplitude waveform</strong><span>Audio level over recording time</span></div>
  <canvas id="waveform" aria-label="Audio amplitude waveform with event markers"></canvas>
  <div class="readout"><span id="position">00:00.0</span><span id="duration">Duration loading</span></div>
  <audio id="player" controls preload="metadata">Audio playback is not supported by this browser.</audio>
  <div class="events" id="events" aria-label="Detected event navigation"></div>
  <div class="hint" id="hint">Event markers link configured phrase matches to their transcript timestamps. They are not sound classifications.</div>
</section>
<script>
const canvas = document.getElementById('waveform');
const context = canvas.getContext('2d');
const player = document.getElementById('player');
const position = document.getElementById('position');
const duration = document.getElementById('duration');
const hint = document.getElementById('hint');
const encodedEvents = '{encoded_events}';
const events = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob(encodedEvents), c => c.charCodeAt(0))));
const encodedAudio = '{encoded_audio}';
player.src = `data:{mime_type};base64,${{encodedAudio}}`;
let peaks = [];
let drawFrame = 0;
function formatTime(seconds) {{
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  return `${{String(Math.floor(value / 60)).padStart(2, '0')}}:${{String(Math.floor(value % 60)).padStart(2, '0')}}.${{Math.floor(value % 1 * 10)}}`;
}}
function draw() {{
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
  canvas.height = Math.max(1, Math.floor(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  const width = bounds.width, height = bounds.height, middle = height / 2;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#cbd7ce'; context.lineWidth = 1;
  context.beginPath(); context.moveTo(0, middle); context.lineTo(width, middle); context.stroke();
  if (peaks.length) {{
    const step = width / peaks.length;
    context.strokeStyle = '#16745b'; context.lineWidth = Math.max(1, step * 0.62);
    context.beginPath();
    peaks.forEach((peak, index) => {{
      const x = index * step + step / 2;
      context.moveTo(x, middle - peak[1] * (middle - 9));
      context.lineTo(x, middle - peak[0] * (middle - 9));
    }});
    context.stroke();
  }}
  const total = player.duration || 0;
  if (total > 0) {{
    events.forEach(event => {{
      const x = Math.max(0, Math.min(width, event.start / total * width));
      context.strokeStyle = event.is_alert ? '#ff5454' : '#e5aa37';
      context.lineWidth = 2; context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }});
    if (player.currentTime > 0) {{
      const x = player.currentTime / total * width;
      context.strokeStyle = '#17231d'; context.lineWidth = 1;
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
    }}
  }}
}}
function tick() {{
  position.textContent = formatTime(player.currentTime);
  draw();
  if (!player.paused && !player.ended) drawFrame = requestAnimationFrame(tick);
}}
const resizeObserver = new ResizeObserver(draw);
resizeObserver.observe(canvas);
player.addEventListener('loadedmetadata', () => {{
  duration.textContent = `Duration ${{formatTime(player.duration)}}`;
  events.forEach(event => {{
    const button = document.createElement('button');
    button.className = event.is_alert ? '' : 'review';
    button.type = 'button';
    button.textContent = `${{formatTime(event.start)}} · ${{event.phrase}} · ${{event.status === 'needs_verification' ? 'Needs verification' : event.name}}`;
    button.setAttribute('aria-label', `Jump to ${{formatTime(event.start)}}: ${{event.phrase}}`);
    button.addEventListener('click', () => {{ player.currentTime = event.start; player.play().catch(() => {{}}); }});
    document.getElementById('events').appendChild(button);
  }});
  if (!events.length) document.getElementById('events').innerHTML = '<span class="empty">No configured phrase matches in this recording.</span>';
  draw();
}});
player.addEventListener('timeupdate', tick);
player.addEventListener('play', () => {{ cancelAnimationFrame(drawFrame); drawFrame = requestAnimationFrame(tick); }});
player.addEventListener('pause', () => {{ cancelAnimationFrame(drawFrame); tick(); }});
player.addEventListener('error', () => {{ hint.textContent = 'This browser could not play or decode the selected audio format. Try WAV, MP3, or M4A in a supported browser.'; }});
async function decodeWaveform() {{
  try {{
    const bytes = Uint8Array.from(atob(encodedAudio), c => c.charCodeAt(0));
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const buffer = await audioContext.decodeAudioData(bytes.buffer);
    const channels = Array.from({{ length: buffer.numberOfChannels }}, (_, index) => buffer.getChannelData(index));
    const bins = Math.min(1200, Math.max(240, Math.floor(canvas.clientWidth * 1.5)));
    const blockSize = Math.max(1, Math.floor(buffer.length / bins));
    peaks = Array.from({{ length: Math.ceil(buffer.length / blockSize) }}, (_, index) => {{
      let minimum = 1, maximum = -1;
      const from = index * blockSize, to = Math.min(buffer.length, from + blockSize);
      for (let sample = from; sample < to; sample += Math.max(1, Math.floor((to - from) / 80))) {{
        let value = 0; channels.forEach(channel => value += channel[sample] || 0); value /= channels.length;
        minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
      }}
      return [minimum, maximum];
    }});
    await audioContext.close(); draw();
  }} catch (error) {{
    hint.textContent = 'Waveform decoding is unavailable for this format in this browser; playback and timestamped results may still work.';
  }}
}}
decodeWaveform();
</script>
</body>
</html>"""
    components.html(html, height=400, scrolling=False)