// Motion mode: play the DHO synth with the phone as a drumstick.
//
// All sensing lives in motion-input.js (instrument-agnostic); this module
// is only the DHO-specific glue and a full-screen overlay:
//
//   pointing position -> which layer-1 node (energy plane row, left to right)
//   downward flick    -> bridge.strikeAt(position, strength)
//   holding the pad   -> energy plane pointer at the pointed-at position,
//                        i.e. exactly what holding a finger on the 2D
//                        energy plane does (sustained drive, plus the
//                        plane's own crossing strikes in impulse/resonator
//                        modes as the stick sweeps across nodes)
//
// Like fullscreen-mode.js/ar-mode.js, it reaches the synth only through
// window.DHOSynth.

import { MotionInput } from './motion-input.js';

const POINTER_ID = 'motion';
const FLASH_MS = 180;

let session = null;

export async function startMotionMode(overlayEl) {
  if (session) return;
  const bridge = window.DHOSynth;
  if (!bridge) {
    console.error('motion-mode: window.DHOSynth is not available yet');
    return;
  }

  const strip = overlayEl.querySelector('#motion-strip');
  const pad = overlayEl.querySelector('#motion-pad');
  const status = overlayEl.querySelector('#motion-status');
  const meter = overlayEl.querySelector('#motion-meter-fill');
  const recenterBtn = overlayEl.querySelector('#motion-recenter');
  const headingSelect = overlayEl.querySelector('#motion-heading');
  const thresholdSlider = overlayEl.querySelector('#motion-threshold');
  const thresholdLabel = overlayEl.querySelector('#motion-threshold-label');
  const arcSlider = overlayEl.querySelector('#motion-arc');
  const arcLabel = overlayEl.querySelector('#motion-arc-label');

  const motion = new MotionInput({
    heading: headingSelect.value,
    strikeThreshold: parseFloat(thresholdSlider.value),
    arcDegrees: parseFloat(arcSlider.value),
  });

  // ---- node strip -------------------------------------------------------
  let cells = [];
  let cursor = null;
  const flashUntil = [];
  function buildStrip() {
    strip.replaceChildren();
    const n = bridge.state.n1;
    cells = [];
    for (let j = 0; j < n; j++) {
      const cell = document.createElement('div');
      cell.className = 'motion-cell';
      strip.appendChild(cell);
      cells.push(cell);
      flashUntil[j] = 0;
    }
    cursor = document.createElement('div');
    cursor.className = 'motion-cursor';
    strip.appendChild(cursor);
  }
  buildStrip();

  const nodeAt = (x) => Math.min(cells.length - 1, Math.floor(x * cells.length));

  // ---- sustain (hold) ---------------------------------------------------
  const holds = new Set();
  let lastFrame = { position: 0.5, heading: 0, elevation: 0 };
  function onPadDown(e) {
    e.preventDefault();
    pad.setPointerCapture(e.pointerId);
    holds.add(e.pointerId);
    pad.classList.add('active');
    bridge.planes.energyPlane.setPointer(POINTER_ID, lastFrame.position);
  }
  function onPadUp(e) {
    holds.delete(e.pointerId);
    if (holds.size) return;
    pad.classList.remove('active');
    bridge.planes.energyPlane.clearPointer(POINTER_ID);
  }
  pad.addEventListener('pointerdown', onPadDown);
  pad.addEventListener('pointerup', onPadUp);
  pad.addEventListener('pointercancel', onPadUp);

  // ---- motion events ----------------------------------------------------
  motion.addEventListener('frame', (e) => {
    const prev = lastFrame.position;
    lastFrame = e.detail;
    if (holds.size && e.detail.position !== prev) {
      bridge.planes.energyPlane.setPointer(POINTER_ID, e.detail.position);
    }
  });
  motion.addEventListener('strike', (e) => {
    const { position, strength } = e.detail;
    bridge.strikeAt(position, strength);
    flashUntil[nodeAt(position)] = performance.now() + FLASH_MS;
    navigator.vibrate?.(Math.round(8 + 20 * strength));
  });

  // ---- controls ---------------------------------------------------------
  const onRecenter = () => motion.recenter();
  recenterBtn.addEventListener('click', onRecenter);

  const onHeading = async () => {
    try {
      await motion.configure({ heading: headingSelect.value });
      motion.recenter();
    } catch (err) {
      status.textContent = String(err.message || err);
    }
  };
  headingSelect.addEventListener('change', onHeading);

  const onThreshold = () => {
    const v = parseFloat(thresholdSlider.value);
    thresholdLabel.textContent = `${v.toFixed(1)} rad/s`;
    motion.configure({ strikeThreshold: v });
  };
  thresholdSlider.addEventListener('input', onThreshold);
  onThreshold();

  const onArc = () => {
    const v = parseFloat(arcSlider.value);
    arcLabel.textContent = `${v}°`;
    motion.configure({ arcDegrees: v });
  };
  arcSlider.addEventListener('input', onArc);
  onArc();

  // ---- render loop ------------------------------------------------------
  let rafHandle = null;
  function frame() {
    rafHandle = requestAnimationFrame(frame);
    if (cells.length !== bridge.state.n1) buildStrip();
    const now = performance.now();
    const pointed = nodeAt(lastFrame.position);
    cells.forEach((cell, j) => {
      cell.classList.toggle('pointed', j === pointed);
      cell.classList.toggle('flash', now < flashUntil[j]);
    });
    cursor.style.left = `${lastFrame.position * 100}%`;
    const fullScale = motion.options.strikeFullScale;
    meter.style.width = `${Math.min(100, (motion.tracker.speed / fullScale) * 100)}%`;
    meter.parentElement.style.setProperty(
      '--threshold',
      `${Math.min(100, (motion.options.strikeThreshold / fullScale) * 100)}%`,
    );
    if (motion.running) {
      status.textContent =
        `${motion.sourceName} | heading ${lastFrame.heading.toFixed(0)}°` +
        ` elev ${lastFrame.elevation.toFixed(0)}° | speed ${motion.tracker.speed.toFixed(1)}`;
    }
  }

  // Keep the screen on while playing; it's re-requested on return to the
  // tab since the browser drops it whenever the page is hidden.
  let wakeLock = null;
  const requestWakeLock = async () => {
    try {
      wakeLock = await navigator.wakeLock?.request('screen');
    } catch {
      wakeLock = null;
    }
  };
  const onVisibility = () => {
    if (document.visibilityState === 'visible') requestWakeLock();
  };
  document.addEventListener('visibilitychange', onVisibility);

  session = {
    motion,
    teardown() {
      cancelAnimationFrame(rafHandle);
      motion.stop();
      pad.removeEventListener('pointerdown', onPadDown);
      pad.removeEventListener('pointerup', onPadUp);
      pad.removeEventListener('pointercancel', onPadUp);
      recenterBtn.removeEventListener('click', onRecenter);
      headingSelect.removeEventListener('change', onHeading);
      thresholdSlider.removeEventListener('input', onThreshold);
      arcSlider.removeEventListener('input', onArc);
      document.removeEventListener('visibilitychange', onVisibility);
      wakeLock?.release().catch(() => {});
      holds.clear();
      pad.classList.remove('active');
      bridge.planes.energyPlane.clearPointer(POINTER_ID);
    },
  };

  frame();
  status.textContent = 'Starting sensors…';
  await bridge.startAudio();
  requestWakeLock();
  try {
    await motion.start();
    motion.recenter();
  } catch (err) {
    status.textContent = String(err.message || err);
  }
}

export function stopMotionMode() {
  if (!session) return;
  session.teardown();
  session = null;
}
