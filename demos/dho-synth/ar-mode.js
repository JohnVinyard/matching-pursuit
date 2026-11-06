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

// Row layout, top-to-bottom, mirroring the 2D UI's color convention
// (index.html: --accent cyan = energy, #ff8a5c orange = damping,
// #b892ff purple = tension-mod -- both zones of a type share that type's
// color). `plane`/`zone` say which window.DHOSynth.planes entry and zone
// (null for the single-zone energy plane) a row's touches feed into.
const ROW_DEFS = [
  { id: 'tensionModL1', color: 0xb892ff, plane: 'tensionModPlane', zone: 'top', count: (s) => s.n1 },
  { id: 'tensionModL2', color: 0xb892ff, plane: 'tensionModPlane', zone: 'bottom', count: (s) => s.n2 },
  { id: 'dampingL1', color: 0xff8a5c, plane: 'dampingPlane', zone: 'top', count: (s) => s.n1 },
  { id: 'dampingL2', color: 0xff8a5c, plane: 'dampingPlane', zone: 'bottom', count: (s) => s.n2 },
  { id: 'energy', color: 0x7fd3ff, plane: 'energyPlane', zone: null, count: (s) => s.n1 },
];

// Sized so the whole 5-row layout sits well inside the camera's view
// frustum at PLANE_Z (comfortably under half its height/width, even at a
// narrow/portrait aspect), with real margin to spare rather than touching
// the edges.
const ROW_SPACING = 0.045; // meters between adjacent row centers
const ROW_WIDTH = 0.18; // meters, world-space span of a row's node line
const ROW_BAND_HALF = ROW_SPACING * 0.45; // vertical hit-test half-height
const ROW_BAND_HYSTERESIS = ROW_SPACING * 0.15; // extra tolerance (proportional to spacing) for the row a finger is already bound to, so it doesn't flicker between rows near a boundary
const PLANE_Z = -0.6; // depth of the control rows (and the video backdrop), in front of the camera at the origin

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
  // land in its band.
  const rows = ROW_DEFS.map((def, index) => {
    const group = new THREE.Group();
    group.position.y = (2 - index) * ROW_SPACING; // top-to-bottom stacking
    group.position.z = PLANE_Z;
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
    const radius = Math.min(0.012, (ROW_WIDTH / n) * 0.35);
    const geometry = new THREE.SphereGeometry(radius, 16, 12);
    for (const normX of nodePositions(n)) {
      const material = new THREE.MeshStandardMaterial({
        color: row.def.color,
        emissive: row.def.color,
        emissiveIntensity: 0,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.x = (normX - 0.5) * ROW_WIDTH;
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
  // apply ROW_BAND_HYSTERESIS so a finger doesn't flicker between two
  // adjacent rows near their shared boundary.
  const bindings = new Map();

  function unbind(id) {
    const prev = bindings.get(id);
    if (!prev) return;
    const row = rows[prev.rowIndex];
    getPlane(row.def.plane).clearPointer(id);
    bindings.delete(id);
  }

  function findRow(id, worldY) {
    const prev = bindings.get(id);
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < rows.length; i++) {
      const half = ROW_BAND_HALF + (prev && prev.rowIndex === i ? ROW_BAND_HYSTERESIS : 0);
      const dist = Math.abs(worldY - rows[i].group.position.y);
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

          const rowIndex = findRow(id, worldPos.y);
          if (rowIndex === null) {
            unbind(id);
            continue;
          }

          const row = rows[rowIndex];
          const prev = bindings.get(id);
          if (prev && prev.rowIndex !== rowIndex) unbind(id);

          const n = row.def.count(bridge.state);
          const normX = Math.min(1, Math.max(0, worldPos.x / ROW_WIDTH + 0.5));
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
