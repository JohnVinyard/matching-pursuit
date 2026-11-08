// AR hand-tracking control surface.
//
// Renders the same five control "rows" the 2D UI's planes represent --
// energy, damping (layer 1 + layer 2), tension-mod (layer 1 + layer 2) --
// as floating 3D spheres in front of the user, and drives them from
// two-hand, five-fingertip-per-hand webcam tracking (MediaPipe Tasks
// Vision's HandLandmarker). A tracked fingertip's position along a row
// feeds the *exact same* tent-kernel weight computation the 2D planes use
// (via each plane object's setPointer/clearPointer, exposed on
// window.DHOSynth by main.js's buildPlanes()) -- so touching a sphere has
// exactly the same effect as a mouse/touch press. This module never posts
// worklet messages or computes weights for real; it only ever calls into
// the existing plane objects.
//
// Loaded as an ES module (for `three` and the MediaPipe package, both only
// available as ESM), lazily imported by main.js (a classic script) only
// when AR mode is actually opened.

import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/+esm';

// Fingertip landmark indices (MediaPipe hand-landmark convention): thumb,
// index, middle, ring, pinky.
const FINGERTIPS = [4, 8, 12, 16, 20];

const DEG = Math.PI / 180;

// The five rows are arranged around a circle, centered in front of the
// user, split into three equal (120 degree) arcs -- one per control type --
// rather than stacked horizontal lines:
//
//               top-center (tension-mod & damping both end here)
//                         ___
//                     .--'   '--.
//   tension-mod   .-"             "-.   damping
//   (left third) /                   \  (right third)
//               |                     |
//                '-.               .-'
//                   '--.       .--'
//                       '--.--'
//                    energy (bottom third)
//
// Each arc is defined by the angle (standard math convention: 0 = +x/right,
// increasing counterclockwise) at node-position 0 and node-position 1 along
// it -- i.e. where a finger at normX=0 vs. normX=1 sits -- rather than a
// plain [min, max], since the tension-mod arc has to sweep *backwards*
// through the +-180 degree seam (down the left side) to reach the top; a1
// is deliberately left "unwrapped" (outside atan2's own (-pi, pi] range) so
// a single linear interpolation/matching scheme (see matchArcAngle) handles
// all three arcs uniformly, seam or no seam.
const ARCS = {
  energy: { a0: -150 * DEG, a1: -30 * DEG },
  damping: { a0: -30 * DEG, a1: 90 * DEG },
  tensionMod: { a0: -150 * DEG, a1: -270 * DEG },
};

// Layer 1 sits on the inner ring, layer 2 on the outer ring, for the two
// control types that have a layer pair; energy (a single layer) sits on a
// ring in between, at the same color as the 2D UI's --accent/--rust/
// --mustard convention.
const INNER_RADIUS = 0.075; // meters
const OUTER_RADIUS = 0.12; // meters
const MID_RADIUS = (INNER_RADIUS + OUTER_RADIUS) / 2;
const RING_GAP = OUTER_RADIUS - INNER_RADIUS;
const RADIAL_BAND_HALF = RING_GAP * 0.45; // radial hit-test half-width for a thin (paired) ring
const RADIAL_BAND_HYSTERESIS = RING_GAP * 0.15; // extra tolerance for the ring a finger is already bound to, so it doesn't flicker near a boundary
// energy has no inner/outer pair to disambiguate against, so its band
// covers the whole span the other two types split between their two rings
// (plus the usual margin), rather than a thin band at exactly MID_RADIUS
const ENERGY_BAND_HALF = RING_GAP / 2 + RADIAL_BAND_HALF;

// Sized so the whole circle sits well inside the camera's view frustum at
// PLANE_Z, with real margin to spare -- including at its widest points,
// where damping's and tension-mod's outer ring cross true horizontal.
const PLANE_Z = -0.6; // depth of the control rows (and the video backdrop), in front of the camera at the origin

const ROW_DEFS = [
  { id: 'tensionModL1', color: 0xd6a53f, plane: 'tensionModPlane', zone: 'top', count: (s) => s.n1, arc: ARCS.tensionMod, radius: INNER_RADIUS, bandHalf: RADIAL_BAND_HALF },
  { id: 'tensionModL2', color: 0xd6a53f, plane: 'tensionModPlane', zone: 'bottom', count: (s) => s.n2, arc: ARCS.tensionMod, radius: OUTER_RADIUS, bandHalf: RADIAL_BAND_HALF },
  { id: 'dampingL1', color: 0xc3693f, plane: 'dampingPlane', zone: 'top', count: (s) => s.n1, arc: ARCS.damping, radius: INNER_RADIUS, bandHalf: RADIAL_BAND_HALF },
  { id: 'dampingL2', color: 0xc3693f, plane: 'dampingPlane', zone: 'bottom', count: (s) => s.n2, arc: ARCS.damping, radius: OUTER_RADIUS, bandHalf: RADIAL_BAND_HALF },
  { id: 'energy', color: 0x3f8a7d, plane: 'energyPlane', zone: null, count: (s) => s.n1, arc: ARCS.energy, radius: MID_RADIUS, bandHalf: ENERGY_BAND_HALF },
];

// Same tent-kernel math as main.js's tentWeights/nodePositions (duplicated
// here, not imported, since main.js is a classic script and this is an ES
// module with no shared build step) -- used only for this module's own
// visual feedback (how brightly a sphere glows), never sent anywhere.
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

// Finds the representation of `angle` (any of angle, angle +- 2*pi) that
// falls within the arc spanned by [a0, a1] (in either order, and possibly
// "unwrapped" outside (-pi, pi], as the tension-mod arc is -- see ARCS),
// or null if none does.
function matchArcAngle(angle, a0, a1) {
  const lo = Math.min(a0, a1);
  const hi = Math.max(a0, a1);
  const eps = 1e-6;
  for (const candidate of [angle, angle + Math.PI * 2, angle - Math.PI * 2]) {
    if (candidate >= lo - eps && candidate <= hi + eps) return candidate;
  }
  return null;
}

let handLandmarkerPromise = null;
function getHandLandmarker() {
  if (!handLandmarkerPromise) {
    handLandmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm',
      );
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        },
        numHands: 2,
        runningMode: 'VIDEO',
      });
    })();
  }
  return handLandmarkerPromise;
}

// Module-level session state -- only one AR session runs at a time.
let session = null;

export async function startAR(overlayEl) {
  if (session) return; // already running

  const bridge = window.DHOSynth;
  if (!bridge) {
    console.error('ar-mode: window.DHOSynth is not available yet');
    return;
  }

  const video = overlayEl.querySelector('#ar-video');
  const canvas = overlayEl.querySelector('#ar-canvas');

  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  await new Promise((resolve) => video.addEventListener('loadeddata', resolve, { once: true }));

  const handLandmarker = await getHandLandmarker();

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 10);
  camera.position.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.8);
  keyLight.position.set(0, 1, 1);
  scene.add(keyLight);

  // Video backdrop: a plane, sized to exactly fill the camera's view
  // frustum at a depth just behind the control rows, textured with the
  // live (mirrored) webcam feed, so tracked fingertip markers visually
  // line up with the user's real hand.
  const videoTexture = new THREE.VideoTexture(video);
  // THREE.VideoTexture samples the raw (un-mirrored) camera frame -- CSS
  // transforms on the <video> element itself have no effect on it, and that
  // <video> element is fully covered by this canvas anyway. Flip the
  // texture's U coordinate here instead, so the rendered backdrop is
  // mirrored to match the fingertips' own mirrored (ndcX) mapping below.
  videoTexture.repeat.x = -1;
  videoTexture.offset.x = 1;
  const videoMaterial = new THREE.MeshBasicMaterial({ map: videoTexture, depthWrite: false });
  const videoPlane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), videoMaterial);
  videoPlane.position.z = PLANE_Z - 0.05;
  scene.add(videoPlane);

  function resize() {
    const w = overlayEl.clientWidth;
    const h = overlayEl.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    const depth = camera.position.z - videoPlane.position.z;
    const viewHeight = 2 * depth * Math.tan((camera.fov * Math.PI) / 360);
    const viewWidth = viewHeight * camera.aspect;
    videoPlane.scale.set(viewWidth, viewHeight, 1);
  }

  // One Object3D per row, holding its current set of sphere meshes (rebuilt
  // whenever that row's node count changes) plus the fingertip markers that
  // land in its band. Every row's group sits at the same point -- the
  // shared circle center -- since what distinguishes a row is now its arc
  // and ring radius (ROW_DEFS), not a per-row world position.
  const rows = ROW_DEFS.map((def) => {
    const group = new THREE.Group();
    group.position.set(0, 0, PLANE_Z);
    scene.add(group);
    return { def, group, spheres: [], lastCount: -1 };
  });

  function rebuildRow(row) {
    for (const mesh of row.spheres) {
      row.group.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    row.spheres = [];

    const n = row.def.count(bridge.state);
    const arcSpan = Math.abs(row.def.arc.a1 - row.def.arc.a0);
    const arcLength = row.def.radius * arcSpan;
    const radius = Math.min(0.012, (arcLength / n) * 0.35);
    const geometry = new THREE.SphereGeometry(radius, 16, 12);
    for (const t of nodePositions(n)) {
      const angle = row.def.arc.a0 + t * (row.def.arc.a1 - row.def.arc.a0);
      const material = new THREE.MeshStandardMaterial({
        color: row.def.color,
        emissive: row.def.color,
        emissiveIntensity: 0,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.x = row.def.radius * Math.cos(angle);
      mesh.position.y = row.def.radius * Math.sin(angle);
      row.group.add(mesh);
      row.spheres.push(mesh);
    }
    row.lastCount = n;
  }

  for (const row of rows) rebuildRow(row);

  function getPlane(key) {
    return bridge.planes[key];
  }

  // Per-fingertip state: id -> { rowIndex }. Lets us (a) clear the correct
  // plane/zone when a finger leaves a row or stops being tracked, and (b)
  // apply RADIAL_BAND_HYSTERESIS so a finger doesn't flicker between two
  // adjacent rings near their shared boundary.
  const bindings = new Map();

  function unbind(id) {
    const prev = bindings.get(id);
    if (!prev) return;
    const row = rows[prev.rowIndex];
    getPlane(row.def.plane).clearPointer(id);
    bindings.delete(id);
  }

  // Which row (if any) a point at the given angle (standard math
  // convention, from atan2) and radius (distance from the circle center)
  // lands in: first narrowed to the rows whose arc contains the angle
  // (exactly one control type's rows, since the three arcs exactly tile
  // the circle), then picked by closest ring radius, same
  // nearest-with-hysteresis scheme the old Y-band version used.
  function findRow(id, angle, radius) {
    const prev = bindings.get(id);
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (matchArcAngle(angle, row.def.arc.a0, row.def.arc.a1) === null) continue;
      const half = row.def.bandHalf + (prev && prev.rowIndex === i ? RADIAL_BAND_HYSTERESIS : 0);
      const dist = Math.abs(radius - row.def.radius);
      if (dist <= half && dist < bestDist) {
        best = i;
        bestDist = dist;
      }
    }
    return best;
  }

  // Standard NDC -> world-space unprojection onto a fixed-depth plane: cast
  // a ray from the camera through the given NDC point, and find where it
  // crosses z === planeZ.
  const unprojectHelper = new THREE.Vector3();
  function unprojectToPlane(ndcX, ndcY, planeZ) {
    unprojectHelper.set(ndcX, ndcY, 0.5).unproject(camera);
    const dir = unprojectHelper.sub(camera.position).normalize();
    const t = (planeZ - camera.position.z) / dir.z;
    return camera.position.clone().addScaledVector(dir, t);
  }

  let lastVideoTime = -1;
  let n1n2Key = `${bridge.state.n1}:${bridge.state.n2}`;
  let rafHandle = null;

  function frame() {
    rafHandle = requestAnimationFrame(frame);

    const key = `${bridge.state.n1}:${bridge.state.n2}`;
    if (key !== n1n2Key) {
      n1n2Key = key;
      for (const row of rows) rebuildRow(row);
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;

      // Reset this (new video frame's) glow accumulator before re-deriving
      // it below -- NOT on every rAF tick, since the webcam's frame rate is
      // typically lower than the display's; resetting unconditionally would
      // make held spheres flicker between render-only frames instead of
      // staying lit.
      for (const row of rows) {
        for (const mesh of row.spheres) mesh.material.emissiveIntensity = 0;
      }

      const detections = handLandmarker.detectForVideo(video, performance.now());
      const seen = new Set();

      for (let h = 0; h < detections.landmarks.length; h++) {
        const landmarks = detections.landmarks[h];
        for (const tip of FINGERTIPS) {
          const lm = landmarks[tip];
          const id = `hand${h}-tip${tip}`;
          seen.add(id);

          // mirrored (x) to match the mirrored video backdrop; y flipped
          // from image space (top=0) to NDC (top=+1)
          const ndcX = 2 * (1 - lm.x) - 1;
          const ndcY = 1 - 2 * lm.y;
          const worldPos = unprojectToPlane(ndcX, ndcY, PLANE_Z);
          const angle = Math.atan2(worldPos.y, worldPos.x);
          const radius = Math.hypot(worldPos.x, worldPos.y);

          const rowIndex = findRow(id, angle, radius);
          if (rowIndex === null) {
            unbind(id);
            continue;
          }

          const row = rows[rowIndex];
          const prev = bindings.get(id);
          if (prev && prev.rowIndex !== rowIndex) unbind(id);

          const n = row.def.count(bridge.state);
          const matched = matchArcAngle(angle, row.def.arc.a0, row.def.arc.a1);
          const normX = Math.min(1, Math.max(0, (matched - row.def.arc.a0) / (row.def.arc.a1 - row.def.arc.a0)));
          getPlane(row.def.plane).setPointer(id, normX, row.def.zone ?? undefined);
          bindings.set(id, { rowIndex });

          const weights = tentWeights(normX, n);
          for (let j = 0; j < n; j++) {
            row.spheres[j].material.emissiveIntensity = Math.max(
              row.spheres[j].material.emissiveIntensity,
              weights[j],
            );
          }
        }
      }

      for (const id of Array.from(bindings.keys())) {
        if (!seen.has(id)) unbind(id);
      }
    }

    renderer.render(scene, camera);
  }

  window.addEventListener('resize', resize);
  resize();
  frame();

  session = {
    stream,
    renderer,
    rows,
    bindings,
    unbind,
    cancel: () => cancelAnimationFrame(rafHandle),
    resize,
  };
}

export function stopAR() {
  if (!session) return;

  session.cancel();
  window.removeEventListener('resize', session.resize);

  // Clear every currently-bound fingertip so nothing is left "stuck on" --
  // unlike this user's own prior hand-tracking demo code elsewhere, which
  // never does this.
  for (const id of Array.from(session.bindings.keys())) session.unbind(id);

  // Actually release the camera (turns off the recording indicator) --
  // also unlike the prior precedent, which never stops its tracks either.
  session.stream.getTracks().forEach((t) => t.stop());

  for (const row of session.rows) {
    for (const mesh of row.spheres) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
  }
  session.renderer.dispose();

  session = null;
}
