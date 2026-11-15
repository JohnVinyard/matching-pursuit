// Full-screen, mobile-optimized touch control surface.
//
// Renders the same five control "rows" ar-mode.js does (energy, damping x2
// layers, tension-mod x2 layers) as a 2D donut-wedge diagram -- three
// sections arranged around a circle (energy, damping, tension-mod), each
// separated by a visible gap, with damping and tension-mod further split
// into an inner (layer 1) and outer (layer 2) ring. Bigger, thumb-friendly
// targets than the stacked planes in the normal 2D UI, meant for mobile.
//
// Unlike ar-mode.js's circle, which exactly tiles 360 degrees (a tracked
// fingertip has to land *somewhere*), this view deliberately leaves a gap
// between each of the three sections, so there's a dead zone a thumb can
// rest in without accidentally triggering energy, damping, or tension-mod.
//
// Driven directly by Pointer Events (not hand tracking or a webcam), so it
// works with touch, mouse, or pen, and -- like the normal 2D planes --
// multiple simultaneous touches are tracked independently by pointer id.
// Drives the existing plane objects' setPointer/clearPointer entry points
// (via window.DHOSynth, exactly as ar-mode.js does); never posts worklet
// messages or computes weights for anything other than this module's own
// visual feedback.
//
// A loaded ES module (consistent with ar-mode.js/mic-input.js's lazy-import
// pattern), though it only uses plain Canvas 2D + Pointer Events internally.

const DEG = Math.PI / 180;

// Same three-way arrangement as ar-mode.js's ARCS (energy bottom, damping
// right, tension-mod left), but each section's actual angular span is
// narrower than its full 120-degree allotment, leaving GAP_DEG of dead zone
// between each pair of neighbors.
const GAP_DEG = 14;
const SECTION_HALF_SPAN = ((120 - GAP_DEG) * DEG) / 2;

const SECTIONS = {
  energy: { centerDeg: -90 },
  damping: { centerDeg: 30 },
  tensionMod: { centerDeg: 150 },
};

// Inner ring = layer 1, outer ring = layer 2, for the two dual-zone types;
// energy (no layer pair) spans the same combined radial range as one ring,
// normalized so 0 = circle center, 1 = ellipse edge.
const INNER_RING = [0.32, 0.6];
const OUTER_RING = [0.62, 0.9];
const ENERGY_RING = [0.32, 0.9];

const ROW_DEFS = [
  { plane: 'tensionModPlane', zone: 'top', count: (s) => s.n1, section: 'tensionMod', ring: INNER_RING, colorVar: '--mustard' },
  { plane: 'tensionModPlane', zone: 'bottom', count: (s) => s.n2, section: 'tensionMod', ring: OUTER_RING, colorVar: '--mustard' },
  { plane: 'dampingPlane', zone: 'top', count: (s) => s.n1, section: 'damping', ring: INNER_RING, colorVar: '--rust' },
  { plane: 'dampingPlane', zone: 'bottom', count: (s) => s.n2, section: 'damping', ring: OUTER_RING, colorVar: '--rust' },
  { plane: 'energyPlane', zone: null, count: (s) => s.n1, section: 'energy', ring: ENERGY_RING, colorVar: '--accent' },
];

// Same tent-kernel math as main.js's tentWeights/nodePositions (duplicated
// here, not imported, same precedent as ar-mode.js -- used only for this
// module's own visual feedback, never sent anywhere).
function nodePositions(n) {
  const positions = [];
  for (let j = 0; j < n; j++) positions.push((j + 0.5) / n);
  return positions;
}
function tentWeights(x, n) {
  const positions = nodePositions(n);
  const halfWidth = 1 / n;
  const weights = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    const dist = Math.abs(x - positions[j]);
    weights[j] = Math.max(0, 1 - dist / halfWidth);
  }
  return weights;
}

function angleDeg(angleRad) {
  return (((angleRad / DEG) % 360) + 360) % 360;
}
function angularDistanceDeg(a, b) {
  const d = Math.abs(a - b) % 360;
  return Math.min(d, 360 - d);
}
function wrapToPi(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function sectionAt(angleRad) {
  const deg = angleDeg(angleRad);
  for (const [id, def] of Object.entries(SECTIONS)) {
    const centerDeg = ((def.centerDeg % 360) + 360) % 360;
    if (angularDistanceDeg(deg, centerDeg) <= SECTION_HALF_SPAN / DEG) return id;
  }
  return null;
}

function rowAt(sectionId, rNorm) {
  for (const def of ROW_DEFS) {
    if (def.section === sectionId && rNorm >= def.ring[0] && rNorm <= def.ring[1]) return def;
  }
  return null;
}

// Position along a row's own arc, 0 at its clockwise-most edge (matching
// the section's a0 = center - halfSpan) to 1 at its other edge.
function normXFor(sectionId, angleRad) {
  const centerRad = SECTIONS[sectionId].centerDeg * DEG;
  const delta = wrapToPi(angleRad - centerRad);
  return Math.min(1, Math.max(0, (delta + SECTION_HALF_SPAN) / (2 * SECTION_HALF_SPAN)));
}

let session = null;

export async function startFullscreen(overlayEl) {
  if (session) return; // already running

  const bridge = window.DHOSynth;
  if (!bridge) {
    console.error('fullscreen-mode: window.DHOSynth is not available yet');
    return;
  }

  const canvas = overlayEl.querySelector('#fullscreen-canvas');
  const ctx = canvas.getContext('2d');

  let cx = 0;
  let cy = 0;
  let rx = 0;
  let ry = 0;

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = overlayEl.clientWidth;
    const h = overlayEl.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cx = w / 2;
    cy = h / 2;
    // Fills most of the available space either way -- genuinely circular
    // on a roughly-square viewport, increasingly ellipsoid (stretched
    // toward whichever dimension is larger) on a narrow phone screen or a
    // wide window, rather than being letterboxed down to the smaller of the
    // two dimensions.
    rx = w * 0.42;
    ry = h * 0.42;
  }

  function pointOn(angle, rNorm) {
    return { x: cx + rNorm * rx * Math.cos(angle), y: cy + rNorm * ry * Math.sin(angle) };
  }

  function colorOf(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  // pointerId -> { def, normX } -- which row (if any) a pointer currently
  // drives, and where along its arc, so move/up events and the per-frame
  // weight-visualization never have to re-hit-test from scratch.
  const bindings = new Map();

  function getPlane(key) {
    return bridge.planes[key];
  }

  function clearBinding(id) {
    const prev = bindings.get(id);
    if (!prev) return;
    getPlane(prev.def.plane).clearPointer(id);
    bindings.delete(id);
  }

  function hitTest(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const dx = (clientX - rect.left - cx) / rx;
    const dy = (clientY - rect.top - cy) / ry;
    const angleRad = Math.atan2(dy, dx);
    const rNorm = Math.hypot(dx, dy);
    const sectionId = sectionAt(angleRad);
    if (sectionId === null) return null;
    const def = rowAt(sectionId, rNorm);
    if (!def) return null;
    return { def, normX: normXFor(sectionId, angleRad) };
  }

  function updatePointer(e) {
    const hit = hitTest(e.clientX, e.clientY);
    if (!hit) {
      clearBinding(e.pointerId);
      return;
    }
    const prev = bindings.get(e.pointerId);
    if (prev && prev.def !== hit.def) clearBinding(e.pointerId);
    getPlane(hit.def.plane).setPointer(e.pointerId, hit.normX, hit.def.zone ?? undefined);
    bindings.set(e.pointerId, hit);
  }

  function onPointerDown(e) {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    updatePointer(e);
  }
  function onPointerMove(e) {
    if (!bindings.has(e.pointerId)) return;
    updatePointer(e);
  }
  function onPointerUp(e) {
    clearBinding(e.pointerId);
  }

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);

  function drawWedgePath(a0, a1, rInner, rOuter) {
    const steps = Math.max(2, Math.ceil(Math.abs(a1 - a0) / (4 * DEG)));
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const t = a0 + (a1 - a0) * (i / steps);
      const p = pointOn(t, rOuter);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    for (let i = steps; i >= 0; i--) {
      const t = a0 + (a1 - a0) * (i / steps);
      const p = pointOn(t, rInner);
      ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
  }

  // A thin visual gap between adjacent node wedges within a row, so
  // individual nodes read as distinct touch targets.
  const NODE_GAP_FRAC = 0.08;

  function drawRow(def) {
    const centerRad = SECTIONS[def.section].centerDeg * DEG;
    const a0 = centerRad - SECTION_HALF_SPAN;
    const a1 = centerRad + SECTION_HALF_SPAN;
    const n = def.count(bridge.state);
    const color = colorOf(def.colorVar);

    const weights = new Array(n).fill(0);
    for (const b of bindings.values()) {
      if (b.def !== def) continue;
      const tent = tentWeights(b.normX, n);
      for (let j = 0; j < n; j++) weights[j] += tent[j];
    }

    ctx.fillStyle = color;
    for (let j = 0; j < n; j++) {
      const t0 = j / n;
      const t1 = (j + 1) / n;
      const pad = (t1 - t0) * NODE_GAP_FRAC;
      const wedgeA0 = a0 + (a1 - a0) * (t0 + pad);
      const wedgeA1 = a0 + (a1 - a0) * (t1 - pad);
      drawWedgePath(wedgeA0, wedgeA1, def.ring[0], def.ring[1]);
      ctx.globalAlpha = 0.15 + Math.min(1, weights[j]) * 0.65;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  let rafHandle = null;
  function frame() {
    rafHandle = requestAnimationFrame(frame);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const def of ROW_DEFS) drawRow(def);
  }

  window.addEventListener('resize', resize);
  resize();
  frame();

  session = {
    cancel: () => cancelAnimationFrame(rafHandle),
    resize,
    bindings,
    clearBinding,
    teardownListeners: () => {
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointercancel', onPointerUp);
    },
  };
}

export function stopFullscreen() {
  if (!session) return;

  session.cancel();
  window.removeEventListener('resize', session.resize);
  session.teardownListeners();

  // Clear every currently-bound pointer so nothing is left "stuck on".
  for (const id of Array.from(session.bindings.keys())) session.clearBinding(id);

  session = null;
}
