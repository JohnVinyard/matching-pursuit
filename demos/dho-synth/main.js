let audioCtx = null;
let workletNode = null;
let analyser = null;
let canvasCtx = null;
let convolverNode = null;
let dryGain = null;
let wetGain = null;

const state = {
  n1: 8,
  n2: 8,
  sparsity: 0.5,
  strategy: 'random',
  excitationMode: 'sustained',
  pulseRate: 100,
  reverb: { file: null, mix: 0, loading: false },
};

// A single-IR convolution reverb (rather than mixing many, as parallel.py's
// NeuralReverb does) -- one impulse response is picked at random per
// randomization, with a random wet/dry mix. Individual objects in this public
// bucket are fetchable (with CORS enabled) even though listing the bucket
// itself is not.
const REVERB_BASE_URL = 'https://matching-pursuit-reverbs.s3.amazonaws.com/';
const REVERB_FILES = [
  'Block Inside.wav', 'Bottle Hall.wav', 'Cement Blocks 1.wav', 'Cement Blocks 2.wav',
  'Chateau de Logne, Outside.wav', 'Conic Long Echo Hall.wav', 'Deep Space.wav',
  'Derlon Sanctuary.wav', 'Direct Cabinet N1.wav', 'Direct Cabinet N2.wav',
  'Direct Cabinet N3.wav', 'Direct Cabinet N4.wav', 'Five Columns Long.wav',
  'Five Columns.wav', 'French 18th Century Salon.wav', 'Going Home.wav',
  'Greek 7 Echo Hall.wav', 'Highly Damped Large Room.wav', 'In The Silo Revised.wav',
  'In The Silo.wav', 'Large Bottle Hall.wav', 'Large Long Echo Hall.wav',
  'Large Wide Echo Hall.wav', 'Masonic Lodge.wav', 'Musikvereinsaal.wav',
  'Narrow Bumpy Space.wav', 'Nice Drum Room.wav', 'On a Star.wav', 'Parking Garage.wav',
  'Rays.wav', 'Right Glass Triangle.wav', 'Ruby Room.wav', 'Scala Milan Opera Hall.wav',
  'Small Drum Room.wav', 'Small Prehistoric Cave.wav', 'St Nicolaes Church.wav',
  'Trig Room.wav', 'Vocal Duo.wav',
];

const reverbBufferCache = new Map(); // filename -> Promise<AudioBuffer>

function loadReverbBuffer(file) {
  if (!reverbBufferCache.has(file)) {
    const url = REVERB_BASE_URL + encodeURIComponent(file);
    reverbBufferCache.set(
      file,
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((buf) => audioCtx.decodeAudioData(buf)),
    );
  }
  return reverbBufferCache.get(file);
}

function applyReverbMix(mix) {
  if (!dryGain || !wetGain) return;
  dryGain.gain.setValueAtTime(1 - mix, audioCtx.currentTime);
  wetGain.gain.setValueAtTime(mix, audioCtx.currentTime);
}

let reverbGeneration = 0;

// Loads a specific IR + mix (used both by randomizeReverb below and by
// restoring an exact reverb choice from a shared link). Buffers are cached
// by filename so re-picking a previously-heard IR is instant; only
// first-time picks incur a network fetch.
async function applyReverbSelection(file, mix) {
  if (!audioCtx) return;
  const myGen = ++reverbGeneration;

  state.reverb = { file, mix, loading: true, error: false };
  applyReverbMix(mix);
  renderReverbInfo();

  try {
    const buffer = await loadReverbBuffer(file);
    if (myGen !== reverbGeneration) return; // a newer selection beat us here
    convolverNode.buffer = buffer;
    state.reverb.loading = false;
    renderReverbInfo();
  } catch (err) {
    console.error('Failed to load reverb IR:', file, err);
    reverbBufferCache.delete(file); // don't cache the failure -- let a later pick retry
    if (myGen !== reverbGeneration) return;
    state.reverb.loading = false;
    state.reverb.error = true;
    renderReverbInfo();
  }
}

// Picks a random impulse response + wet/dry mix.
function randomizeReverb() {
  const file = REVERB_FILES[Math.floor(Math.random() * REVERB_FILES.length)];
  const mix = uniform(0, 0.7);
  return applyReverbSelection(file, mix);
}

function uniform(lo, hi) {
  return lo + Math.random() * (hi - lo);
}

// dampLambda is a continuous-time decay rate (1/sec, see stepNode's
// alpha/beta discretization): low values (~0.3, ~3.3s time constant) give
// long sustains, high values (~80, ~12ms time constant) give sharp,
// percussive hits.
const DAMP_LAMBDA_RANGE = [0.3, 80];

// Skews sampling toward the fast/percussive end of DAMP_LAMBDA_RANGE, with
// long sustains as an increasingly rare "long tail" -- roughly how decay
// times are distributed among real physical objects (most things damp out
// quickly when struck; only a few, like a bell or a taut string, ring for a
// long time). Draws a uniform(0,1) value, raises it to DAMP_SKEW (>1
// concentrates mass near 0), then maps 0 -> the fast end and 1 -> the slow
// end of the log-scaled range, so small (common) values land near fast decay
// and only rare draws near 1 reach all the way to a multi-second sustain.
// At DAMP_SKEW=2, roughly 76% of draws decay within 300ms, but ~11% still
// sustain past a second, and the full multi-second range stays reachable.
const DAMP_SKEW = 2;

function skewedDampLambda([lo, hi], skew) {
  const logLo = Math.log10(lo);
  const logHi = Math.log10(hi);
  const t = Math.pow(Math.random(), skew);
  return Math.pow(10, logHi - t * (logHi - logLo));
}

// `gain` feeds each layer-2 node's own tanh saturation (see the processor's
// per-node `Math.tanh(z2 * node.gain)`), matching parallel.py's
// `instrument.gains`. It's randomized across a range that spans clean
// (< 1, rarely saturating) to hard-driven (> 1, saturates most of the time,
// adding buzzy harmonics) so nodes differ in harmonic character, not just
// pitch/decay. Overall loudness-vs-node-count is normalized separately in the
// processor's final mix, so this doesn't need a 1/n compensation term.
const FILTER_TAPS = 32;

// A random, unit-normalized 32-tap FIR kernel per node (matching
// `unit_norm(instrument.filters)` in parallel.py) -- iid random taps give a
// broadband, noisy coloration rather than a narrow resonant peak, i.e. a
// "wide-band" filter. filterMix is the unconstrained [dry, filtered] blend
// weight (`instrument.filters_mix`), applied before the node's gain/tanh.
function randomFilter() {
  const taps = Array.from({ length: FILTER_TAPS }, () => uniform(-1, 1));
  const norm = Math.sqrt(taps.reduce((acc, v) => acc + v * v, 0)) || 1;
  return {
    filterTaps: taps.map((v) => v / norm),
    filterMix: [uniform(-1, 1), uniform(-1, 1)],
  };
}

function randomNodeParams(n, isModulatorLayer) {
  const params = [];
  for (let i = 0; i < n; i++) {
    params.push({
      mass: uniform(0.3, 3),
      tensionExp: uniform(4, isModulatorLayer ? 7 : 8),
      tensionMod: 0,
      dampLambda: skewedDampLambda(DAMP_LAMBDA_RANGE, DAMP_SKEW),
      dampMod: 0,
      excitationGain: uniform(0.5, 1.5),
      gain: uniform(0.5, 3),
      ...randomFilter(),
    });
  }
  return params;
}

// `sparsity` is the probability that any given routing weight is zeroed out
// rather than kept -- the intuition being that a mostly-sparse routing matrix
// (each layer-2 node driven/modulated by only a couple of layer-1 nodes,
// rather than a diffuse blend of all of them) should produce more distinct,
// less "averaged-together" timbres per node. Each row keeps at least one
// nonzero entry so a node is never silently orphaned by chance.
function randomRouter(n2, n1, lo, hi, sparsity) {
  const router = [];
  for (let k = 0; k < n2; k++) {
    const row = [];
    for (let j = 0; j < n1; j++) {
      row.push(Math.random() < sparsity ? 0 : uniform(lo, hi));
    }
    if (row.every((v) => v === 0)) {
      row[Math.floor(Math.random() * n1)] = uniform(lo, hi);
    }
    router.push(row);
  }
  return router;
}

// tensionRouter's range is the sole control over FM/modulation intensity --
// there's no separate "mod depth" scalar (removed: it was redundant with this
// range, since multiplying an already-routed value by a constant is
// equivalent to scaling every entry of the matrix that produced it).
//
// This was previously +-0.008, shrunk from +-0.4 specifically to tame FM that
// felt oversensitive -- but that oversensitivity turned out to be an artifact
// of a phase-continuity bug in the oscillator (since fixed), which made every
// routed tension change produce a small waveform discontinuity injecting
// broadband noise-like energy on top of the "real" FM sidebands. With that
// bug fixed, +-0.008 is too tame (clean but glassy/bell-like -- all the
// complexity that discontinuity was supplying for free is gone); +-0.1
// recovered comparable spectral richness to the old buggy behavior in a
// quick spectral-flatness check, cleanly this time. Depth-vs-richness isn't
// monotonic (classic FM sideband/Bessel-function behavior), so this is a
// starting point to dial in by ear, not a precisely derived value. Widened
// again to +-0.2 per feedback that more depth was still wanted.
const TENSION_ROUTER_RANGE = [-0.2, 0.2];

function randomRouters(n1, n2, sparsity) {
  return {
    // tensionRouter: how much layer1 node j's raw output shifts layer2 node k's tension
    tensionRouter: randomRouter(n2, n1, ...TENSION_ROUTER_RANGE, sparsity),
    // forceRouter2: how much targeting/pressing layer1 node j drives layer2 node k's excitation
    forceRouter2: randomRouter(n2, n1, 0, 1.4, sparsity),
  };
}

// Layer 1's nodes are architecturally separate, independently excitable
// resonators -- the control plane lets you strike each one individually --
// which is much closer to separate bars on a mallet instrument (or keys, or
// strings) than to the partials of one single struck note. Each node's note
// is drawn independently at random (the same per-layer tensionExp range the
// "random" strategy uses, i.e. uniform in log-frequency), rather than being
// derived from any shared fundamental -- that's what keeps different
// control-plane positions sounding like genuinely different notes instead of
// fusing into partials of one complex tone (harmonically-related pure tones
// perceptually fuse into a single voice, which is what a shared-fundamental
// series was doing here).
const PHYSICAL_TENSION_EXP_RANGE = { modulator: [4, 7], carrier: [4, 8] };

// Damping still increases with frequency (higher notes decay faster), which
// remains true regardless of whether nodes are partials of one voice or
// separate notes -- it's a general acoustic fact (higher frequencies lose
// energy to radiation/friction faster), so it scales off each node's
// frequency ratio to the bottom of its layer's range.
//
// That scaling has a side effect worth calling out: stepNode's leaky
// integrator settles to a steady-state energy of force/(mass*lambda), so
// simply multiplying lambda up for high notes doesn't just shorten them --
// it makes them proportionally quieter too (up to ~1000x quieter at the top
// of the carrier's register, since freq ratios there reach ~100x and the
// exponent is 1.5). Scaling mass *down* by that same factor cancels it out:
// mass*lambda (and so the steady-state loudness) stays roughly constant
// across the whole register, while the post-release decay rate -- which
// depends on lambda alone, not the product -- still correctly runs faster
// for higher notes.
const FREQ_DAMPING_EXPONENT = 1.5;

function physicalLayerParams(n, isModulatorLayer) {
  const [minExp, maxExp] = isModulatorLayer
    ? PHYSICAL_TENSION_EXP_RANGE.modulator
    : PHYSICAL_TENSION_EXP_RANGE.carrier;
  const minFreq = f0Hz(minExp);
  const baseDampLambda = skewedDampLambda(DAMP_LAMBDA_RANGE, DAMP_SKEW);
  const params = [];
  for (let i = 0; i < n; i++) {
    const tensionExp = uniform(minExp, maxExp);
    const freq = f0Hz(tensionExp);
    const dampScale = Math.pow(freq / minFreq, FREQ_DAMPING_EXPONENT);
    params.push({
      mass: uniform(0.3, 3) / dampScale,
      tensionExp,
      tensionMod: 0,
      dampLambda: baseDampLambda * dampScale,
      dampMod: 0,
      excitationGain: uniform(0.5, 1.5),
      gain: uniform(0.5, 3),
      ...randomFilter(),
    });
  }
  return params;
}

// A "mostly, but not entirely, diagonal" routing matrix: each layer-2 node k
// is paired with whichever layer-1 node j sits at the nearest proportional
// position (handling n1 != n2), and that pairing always gets a full-strength
// coupling value -- sparsity never zeroes it out. Every other entry gets a
// much smaller cross-talk value, still subject to sparsity. This echoes
// parallel.py's own tension_router/force_router init (`torch.eye +
// uniform(-0.01, 0.01)`), itself a physically-motivated prior: each mode
// primarily drives its own corresponding partial, with only slight coupling
// to the others -- rather than the fully-random matrix the "random" strategy
// uses, which spreads each node's influence diffusely across every partial.
const DIAGONAL_CROSSTALK_SCALE = 0.15;

function diagonalBiasedRouter(n2, n1, lo, hi, sparsity) {
  const router = [];
  for (let k = 0; k < n2; k++) {
    const t = n2 > 1 ? k / (n2 - 1) : 0;
    const diagJ = Math.round(t * (n1 - 1));
    const row = [];
    for (let j = 0; j < n1; j++) {
      if (j === diagJ) {
        row.push(uniform(lo, hi));
      } else {
        row.push(
          Math.random() < sparsity
            ? 0
            : uniform(lo * DIAGONAL_CROSSTALK_SCALE, hi * DIAGONAL_CROSSTALK_SCALE),
        );
      }
    }
    router.push(row);
  }
  return router;
}

function physicalRouters(n1, n2, sparsity) {
  return {
    tensionRouter: diagonalBiasedRouter(n2, n1, ...TENSION_ROUTER_RANGE, sparsity),
    forceRouter2: diagonalBiasedRouter(n2, n1, 0, 1.4, sparsity),
  };
}

// --- Randomization strategies --------------------------------------------
// A strategy supplies two independent generators:
//   generateLayerParams(n, isModulatorLayer) -> per-node param array for one
//     layer (called once for layer 1, once for layer 2)
//   generateRouters(n1, n2, sparsity) -> { tensionRouter, forceRouter2 }
// Keeping these separate (rather than one monolithic "generate everything"
// call) is what lets the sparsity slider re-roll just the routing matrices
// (regenerateRouters) under whichever strategy is active, without disturbing
// the currently-generated node params.
const STRATEGIES = {
  random: {
    label: 'Fully random',
    generateLayerParams: randomNodeParams,
    generateRouters: randomRouters,
  },
  physical: {
    label: 'Physically plausible',
    generateLayerParams: physicalLayerParams,
    generateRouters: physicalRouters,
  },
};

let lastParams = null;

function randomizeAll() {
  const strategy = STRATEGIES[state.strategy];
  const params = {
    layer1: strategy.generateLayerParams(state.n1, true),
    layer2: strategy.generateLayerParams(state.n2, false),
    ...strategy.generateRouters(state.n1, state.n2, state.sparsity),
  };
  lastParams = params;
  if (workletNode) {
    workletNode.port.postMessage({ type: 'setParams', params });
  }
  renderParamsPanel(params);
  randomizeReverb();
  return params;
}

// Re-rolls only the routing matrices (keeping current node params) so the
// sparsity slider can be dragged live without disturbing masses/tensions.
function regenerateRouters() {
  if (!lastParams) return;
  const strategy = STRATEGIES[state.strategy];
  const routers = strategy.generateRouters(state.n1, state.n2, state.sparsity);
  lastParams.tensionRouter = routers.tensionRouter;
  lastParams.forceRouter2 = routers.forceRouter2;
  if (workletNode) {
    workletNode.port.postMessage({ type: 'setParams', params: routers });
  }
  renderParamsPanel(lastParams);
}

// --- Share links -----------------------------------------------------------
// Packs the entire current instrument (topology, all per-node params, both
// routing matrices, and the surrounding UI state) into a compact binary blob,
// base64url-encoded into a `?p=...` query param. Most of the payload
// (mass/tensionExp/dampLambda/excitationGain/gain) is stored as plain
// float32 since their ranges vary too much across strategies for cheap
// linear quantization to be worth the complexity; the two things that don't
// need much precision -- the 32 random FIR filter taps and 2-value filter
// mix per node, and both n2xn1 routing matrices -- are quantized down to
// int8/int16, which is most of the size savings (filter data alone would
// otherwise be 4x bigger). Doesn't compress across the URL; not needed at
// this size (a couple KB even at 16x16 nodes).
const SHARE_VERSION = 1;
const NODE_BYTE_LENGTH = 4 * 5 + FILTER_TAPS + 2; // 5 float32 scalars + int8 taps + int8 filterMix
const ROUTER_QUANT_RANGE = 32; // generous headroom above any range we've used so far
const EXCITATION_MODE_IDS = ['sustained', 'impulse', 'pulse', 'noise'];

function quantizeU8(v, lo, hi) {
  return Math.max(0, Math.min(255, Math.round(((v - lo) / (hi - lo)) * 255)));
}
function dequantizeU8(b, lo, hi) {
  return lo + (b / 255) * (hi - lo);
}
function quantizeLogU8(v, lo, hi) {
  return quantizeU8(Math.log10(v), Math.log10(lo), Math.log10(hi));
}
function dequantizeLogU8(b, lo, hi) {
  return Math.pow(10, dequantizeU8(b, Math.log10(lo), Math.log10(hi)));
}
function quantizeI8(v) {
  return Math.max(-127, Math.min(127, Math.round(v * 127)));
}
function dequantizeI8(b) {
  return b / 127;
}
function quantizeI16(v, range) {
  return Math.max(-32767, Math.min(32767, Math.round((v / range) * 32767)));
}
function dequantizeI16(v, range) {
  return (v / 32767) * range;
}

function shareByteLength(n1, n2) {
  return 11 + (n1 + n2) * NODE_BYTE_LENGTH + n2 * n1 * 2 * 2;
}

function encodeNode(view, offset, p) {
  view.setFloat32(offset, p.mass, true); offset += 4;
  view.setFloat32(offset, p.tensionExp, true); offset += 4;
  view.setFloat32(offset, p.dampLambda, true); offset += 4;
  view.setFloat32(offset, p.excitationGain, true); offset += 4;
  view.setFloat32(offset, p.gain, true); offset += 4;
  for (let i = 0; i < FILTER_TAPS; i++) {
    view.setInt8(offset, quantizeI8(p.filterTaps[i]));
    offset += 1;
  }
  view.setInt8(offset, quantizeI8(p.filterMix[0])); offset += 1;
  view.setInt8(offset, quantizeI8(p.filterMix[1])); offset += 1;
  return offset;
}

function decodeNode(view, offset) {
  const mass = view.getFloat32(offset, true); offset += 4;
  const tensionExp = view.getFloat32(offset, true); offset += 4;
  const dampLambda = view.getFloat32(offset, true); offset += 4;
  const excitationGain = view.getFloat32(offset, true); offset += 4;
  const gain = view.getFloat32(offset, true); offset += 4;
  const filterTaps = [];
  for (let i = 0; i < FILTER_TAPS; i++) {
    filterTaps.push(dequantizeI8(view.getInt8(offset)));
    offset += 1;
  }
  const filterMix = [dequantizeI8(view.getInt8(offset)), dequantizeI8(view.getInt8(offset + 1))];
  offset += 2;
  const node = { mass, tensionExp, tensionMod: 0, dampLambda, dampMod: 0, excitationGain, gain, filterTaps, filterMix };
  return { node, offset };
}

function bytesToBase64Url(bytes) {
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function base64UrlToBytes(str) {
  let s = str.replace(/-/g, '+').replace(/_/g, '/');
  while (s.length % 4) s += '=';
  const binary = atob(s);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// Encodes the currently-generated instrument (must have randomized/loaded at
// least once) plus the surrounding UI state into a shareable byte blob.
function encodeShareState() {
  if (!lastParams) return null;
  const n1 = state.n1;
  const n2 = state.n2;
  const buffer = new ArrayBuffer(shareByteLength(n1, n2));
  const view = new DataView(buffer);
  let offset = 0;

  view.setUint8(offset, SHARE_VERSION); offset += 1;
  view.setUint8(offset, n1); offset += 1;
  view.setUint8(offset, n2); offset += 1;
  view.setUint8(offset, state.strategy === 'physical' ? 1 : 0); offset += 1;
  view.setUint8(offset, Math.max(0, EXCITATION_MODE_IDS.indexOf(state.excitationMode))); offset += 1;
  const reverbIdx = REVERB_FILES.indexOf(state.reverb.file);
  view.setUint8(offset, reverbIdx >= 0 ? reverbIdx : 255); offset += 1;
  view.setUint8(offset, quantizeU8(state.sparsity, 0, 0.95)); offset += 1;
  view.setUint8(offset, quantizeU8(state.pulseRate, 20, 400)); offset += 1;
  const volume = workletNode ? workletNode.parameters.get('masterGain').value : 0.6;
  const modDepth = workletNode ? workletNode.parameters.get('modDepth').value : 1;
  view.setUint8(offset, quantizeU8(volume, 0, 10)); offset += 1;
  view.setUint8(offset, quantizeLogU8(modDepth, 0.1, 10)); offset += 1;
  view.setUint8(offset, quantizeU8(state.reverb.mix, 0, 1)); offset += 1;

  for (const p of lastParams.layer1) offset = encodeNode(view, offset, p);
  for (const p of lastParams.layer2) offset = encodeNode(view, offset, p);
  for (const row of lastParams.tensionRouter) {
    for (const v of row) { view.setInt16(offset, quantizeI16(v, ROUTER_QUANT_RANGE), true); offset += 2; }
  }
  for (const row of lastParams.forceRouter2) {
    for (const v of row) { view.setInt16(offset, quantizeI16(v, ROUTER_QUANT_RANGE), true); offset += 2; }
  }

  return new Uint8Array(buffer);
}

function decodeShareState(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  const version = view.getUint8(offset); offset += 1;
  if (version !== SHARE_VERSION) throw new Error(`unsupported share link version ${version}`);
  const n1 = view.getUint8(offset); offset += 1;
  const n2 = view.getUint8(offset); offset += 1;
  const strategy = view.getUint8(offset) === 1 ? 'physical' : 'random'; offset += 1;
  const excitationMode = EXCITATION_MODE_IDS[view.getUint8(offset)] || 'sustained'; offset += 1;
  const reverbFileIdx = view.getUint8(offset); offset += 1;
  const sparsity = dequantizeU8(view.getUint8(offset), 0, 0.95); offset += 1;
  const pulseRate = dequantizeU8(view.getUint8(offset), 20, 400); offset += 1;
  const volume = dequantizeU8(view.getUint8(offset), 0, 10); offset += 1;
  const modDepth = dequantizeLogU8(view.getUint8(offset), 0.1, 10); offset += 1;
  const reverbMix = dequantizeU8(view.getUint8(offset), 0, 1); offset += 1;

  const layer1 = [];
  for (let i = 0; i < n1; i++) {
    const r = decodeNode(view, offset);
    layer1.push(r.node);
    offset = r.offset;
  }
  const layer2 = [];
  for (let i = 0; i < n2; i++) {
    const r = decodeNode(view, offset);
    layer2.push(r.node);
    offset = r.offset;
  }

  function decodeRouter(n2_, n1_) {
    const router = [];
    for (let k = 0; k < n2_; k++) {
      const row = [];
      for (let j = 0; j < n1_; j++) {
        row.push(dequantizeI16(view.getInt16(offset, true), ROUTER_QUANT_RANGE));
        offset += 2;
      }
      router.push(row);
    }
    return router;
  }
  const tensionRouter = decodeRouter(n2, n1);
  const forceRouter2 = decodeRouter(n2, n1);

  const reverbFile = reverbFileIdx < REVERB_FILES.length ? REVERB_FILES[reverbFileIdx] : null;

  return {
    n1, n2, strategy, excitationMode, pulseRate, sparsity, volume, modDepth, reverbFile, reverbMix,
    params: { layer1, layer2, tensionRouter, forceRouter2 },
  };
}

function buildShareUrl() {
  const bytes = encodeShareState();
  if (!bytes) return null;
  const encoded = bytesToBase64Url(bytes);
  return `${location.origin}${location.pathname}?p=${encoded}`;
}

// Parsed once at script load (before any UI/audio setup) so the page can
// restore a shared instrument on load. `?p=...` doesn't get rewritten as
// you tweak things afterward -- it's a snapshot, not a live-synced URL.
let loadedShare = null;
try {
  const p = new URLSearchParams(location.search).get('p');
  if (p) loadedShare = decodeShareState(base64UrlToBytes(p));
} catch (err) {
  console.error('Failed to parse shared instrument link:', err);
}

async function ensureAudio() {
  if (audioCtx) return;

  audioCtx = new AudioContext();
  await audioCtx.audioWorklet.addModule('dho-processor.js');

  workletNode = new AudioWorkletNode(audioCtx, 'dho-processor', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [2],
  });

  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;

  // single-IR convolution reverb: workletNode splits into a dry path and a
  // convolved wet path, both summed into analyser -> destination. The
  // scope taps the post-reverb signal, i.e. what's actually audible.
  convolverNode = audioCtx.createConvolver();
  // normalize (default true) auto-scales each IR to a comparable loudness --
  // useful here since random picks range from small drum rooms to huge halls
  // with very different raw impulse energy.
  convolverNode.normalize = true;
  dryGain = audioCtx.createGain();
  wetGain = audioCtx.createGain();
  dryGain.gain.value = 1;
  wetGain.gain.value = 0;

  workletNode.connect(dryGain);
  workletNode.connect(convolverNode);
  convolverNode.connect(wetGain);
  dryGain.connect(analyser);
  wetGain.connect(analyser);
  analyser.connect(audioCtx.destination);

  workletNode.port.postMessage({ type: 'configure', n1: state.n1, n2: state.n2 });
  workletNode.port.postMessage({ type: 'setExcitationMode', mode: state.excitationMode });
  workletNode.port.postMessage({ type: 'setPulseRate', rate: state.pulseRate });

  if (loadedShare) {
    lastParams = loadedShare.params;
    workletNode.port.postMessage({ type: 'setParams', params: lastParams });
    renderParamsPanel(lastParams);
    workletNode.parameters.get('masterGain').setValueAtTime(loadedShare.volume, audioCtx.currentTime);
    workletNode.parameters.get('modDepth').setValueAtTime(loadedShare.modDepth, audioCtx.currentTime);
    if (loadedShare.reverbFile) {
      applyReverbSelection(loadedShare.reverbFile, loadedShare.reverbMix);
    } else {
      applyReverbMix(loadedShare.reverbMix);
    }
  } else {
    randomizeAll();
  }

  drawScope();
}

// Spatial control planes: each plane lays its target nodes left-to-right and
// turns pointer position into a per-node weight vector via a triangular
// "tent" kernel centered on the pointer's X position -- a single click
// targets one node, dragging sweeps/crossfades between neighbors, and
// multiple simultaneous touches (independent pointerIds) can target several
// nodes at once. Three planes exist, grouped by control *type* rather than
// mirroring the network's per-layer structure: energy (into layer 1), and
// one dual-zone plane each for damping and tension-mod, each zoned by layer
// (layer 1 on top, layer 2 on the bottom) -- see buildPlanes() below for how
// they're wired up and ordered on the page.
const STRIKE_PEAK = 1.6;

function nodePositions(n) {
  const positions = [];
  for (let j = 0; j < n; j++) positions.push((j + 0.5) / n);
  return positions;
}

// The "tent kernel": one active point at normalized x contributes a weight to
// every node, falling off linearly to zero at the node's neighbors. Shared by
// createPlane and createDualZonePlane (each sums this over all their active
// points), and reused as-is by AR mode's 3D hand tracking -- a tracked
// fingertip's position along a row is just another point fed into the same
// math, so touching a sphere has exactly the same effect as a mouse/touch
// press.
function tentWeights(x, n) {
  const positions = nodePositions(n);
  const halfWidth = 1 / n;
  const weights = new Array(n).fill(0);
  for (let j = 0; j < n; j++) {
    const dist = Math.abs(x - positions[j]);
    weights[j] = Math.max(0, 1 - dist / halfWidth) * STRIKE_PEAK;
  }
  return weights;
}

function createPlane(elementId, getNodeCount, sendMessage, onCrossing) {
  const el = document.getElementById(elementId);
  const pointers = new Map(); // id -> normalized x in [0, 1]
  const lastCell = new Map(); // id -> nearest node index, for onCrossing

  function computeWeights() {
    const n = getNodeCount();
    const weights = new Array(n).fill(0);
    for (const x of pointers.values()) {
      const tent = tentWeights(x, n);
      for (let j = 0; j < n; j++) weights[j] += tent[j];
    }
    return weights;
  }

  function send() {
    const weights = computeWeights();
    if (workletNode) sendMessage(weights);
    updateMarkerRow(`#${elementId} .node-marker`, weights);
  }

  // Fires whenever a point newly lands, or drags across into a different
  // nearest node -- an edge-triggered "you just struck this node" event, as
  // opposed to `send()`'s continuous "here's the current weight level."
  function checkCrossing(id, x) {
    if (!onCrossing) return;
    const n = getNodeCount();
    const cell = n > 1 ? Math.round(x * (n - 1)) : 0;
    if (lastCell.get(id) !== cell) {
      lastCell.set(id, cell);
      onCrossing(computeWeights());
    }
  }

  // The single entry point for "a point (a real PointerEvent, or a synthetic
  // id for an AR-tracked fingertip) is active at normalized x" -- both input
  // sources funnel through here, so there's exactly one path for weight
  // computation, message-sending and marker updates.
  function setPointer(id, x) {
    pointers.set(id, x);
    checkCrossing(id, x);
    send();
  }

  function clearPointer(id) {
    pointers.delete(id);
    lastCell.delete(id);
    send();
  }

  function normalizedX(e) {
    const rect = el.getBoundingClientRect();
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  }

  el.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    el.classList.add('active');
    setPointer(e.pointerId, normalizedX(e));
  });
  el.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    setPointer(e.pointerId, normalizedX(e));
  });
  const release = (e) => {
    clearPointer(e.pointerId);
    if (pointers.size === 0) el.classList.remove('active');
  };
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);

  return {
    setPointer,
    clearPointer,
    clearPointers: () => {
      pointers.clear();
      lastCell.clear();
    },
    rebuildMarkers() {
      el.querySelectorAll('.node-marker').forEach((m) => m.remove());
      for (const pos of nodePositions(getNodeCount())) {
        const marker = document.createElement('div');
        marker.className = 'node-marker';
        marker.style.left = `${pos * 100}%`;
        el.appendChild(marker);
      }
    },
  };
}

// Same tent-kernel X-axis targeting as createPlane, but splits the plane
// into two independent zones (top/bottom, like the original control plane
// before it was split into single-purpose strips). Grouped by control
// *type* rather than by layer: one plane per type (damping, tension-mod),
// its top zone targeting layer 1's nodes and its bottom zone layer 2's --
// so, unlike the very first version of this split, the two zones can have
// different node counts and send to different targets. A pointer's zone is
// fixed at wherever its gesture started, even if it later drags across the
// divider.
function createDualZonePlane(elementId, getTopNodeCount, sendTop, getBottomNodeCount, sendBottom) {
  const el = document.getElementById(elementId);
  const pointers = new Map(); // id -> { zone: 'top' | 'bottom', x }

  function computeZoneWeights(zone) {
    const n = zone === 'top' ? getTopNodeCount() : getBottomNodeCount();
    const weights = new Array(n).fill(0);
    for (const p of pointers.values()) {
      if (p.zone !== zone) continue;
      const tent = tentWeights(p.x, n);
      for (let j = 0; j < n; j++) weights[j] += tent[j];
    }
    return weights;
  }

  function send() {
    const topWeights = computeZoneWeights('top');
    const bottomWeights = computeZoneWeights('bottom');
    if (workletNode) {
      sendTop(topWeights);
      sendBottom(bottomWeights);
    }
    updateMarkerRow(`#${elementId} .node-marker.top`, topWeights);
    updateMarkerRow(`#${elementId} .node-marker.bottom`, bottomWeights);
  }

  // The single entry point for "a point is active at normalized x within
  // `zone`" -- real PointerEvents (zone fixed at gesture start, see below)
  // and AR hand-tracking (zone re-evaluated every frame from 3D row
  // position) both funnel through here. `zone` is optional so a pointermove-
  // style update (position changed, zone unchanged) can omit it.
  function setPointer(id, x, zone) {
    const existing = pointers.get(id);
    pointers.set(id, { zone: zone ?? existing?.zone ?? 'top', x });
    send();
  }

  function clearPointer(id) {
    pointers.delete(id);
    send();
  }

  function normalizedX(e, rect) {
    return Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  }
  function zoneFor(e, rect) {
    const relY = (e.clientY - rect.top) / rect.height;
    return relY < 0.5 ? 'top' : 'bottom';
  }

  el.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    const rect = el.getBoundingClientRect();
    el.setPointerCapture(e.pointerId);
    el.classList.add('active');
    setPointer(e.pointerId, normalizedX(e, rect), zoneFor(e, rect));
  });
  el.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    // zone omitted -- stays fixed at whatever pointerdown set it to
    setPointer(e.pointerId, normalizedX(e, el.getBoundingClientRect()));
  });
  const release = (e) => {
    clearPointer(e.pointerId);
    if (pointers.size === 0) el.classList.remove('active');
  };
  el.addEventListener('pointerup', release);
  el.addEventListener('pointercancel', release);

  return {
    setPointer,
    clearPointer,
    clearPointers: () => pointers.clear(),
    rebuildMarkers() {
      el.querySelectorAll('.node-marker').forEach((m) => m.remove());
      for (const [zone, getNodeCount] of [['top', getTopNodeCount], ['bottom', getBottomNodeCount]]) {
        for (const pos of nodePositions(getNodeCount())) {
          const marker = document.createElement('div');
          marker.className = `node-marker ${zone}`;
          marker.style.left = `${pos * 100}%`;
          el.appendChild(marker);
        }
      }
    },
  };
}

let planes = [];

function buildPlanes() {
  const energyPlane = createPlane(
    'energyPlane',
    () => state.n1,
    (weights) => {
      workletNode.port.postMessage({ type: 'excite', weights });
    },
    (weights) => {
      // only meaningful in 'impulse' mode -- the continuous 'excite' weights
      // above still update in every mode (used for visuals, and as the
      // amplitude source for 'pulse' mode's periodic ticks in the processor)
      if (state.excitationMode === 'impulse' && workletNode) {
        workletNode.port.postMessage({ type: 'strike', weights });
      }
    },
  );
  const dampingPlane = createDualZonePlane(
    'dampingPlane',
    () => state.n1,
    (weights) => workletNode.port.postMessage({ type: 'dampen', layer: 1, weights }),
    () => state.n2,
    (weights) => workletNode.port.postMessage({ type: 'dampen', layer: 2, weights }),
  );
  const tensionModPlane = createDualZonePlane(
    'tensionModPlane',
    () => state.n1,
    (weights) => workletNode.port.postMessage({ type: 'tensionmod', layer: 1, weights }),
    () => state.n2,
    (weights) => workletNode.port.postMessage({ type: 'tensionmod', layer: 2, weights }),
  );
  planes = [energyPlane, dampingPlane, tensionModPlane];
  for (const plane of planes) plane.rebuildMarkers();

  // Minimal bridge for ar-mode.js (a classic script, main.js, can't export
  // anything an ES module could `import`) to reach the plane objects and
  // shared state/worklet it needs to drive the same setPointer/clearPointer
  // entry points real pointer events use -- intentionally just these three
  // things, not a general API.
  window.DHOSynth = {
    planes: { energyPlane, dampingPlane, tensionModPlane },
    state,
    ensureAudio,
    getWorkletNode: () => workletNode,
    getAudioContext: () => audioCtx,
  };
}

function resetPlanes() {
  for (const plane of planes) {
    plane.clearPointers();
    plane.rebuildMarkers();
  }
}

function f0Hz(tensionExp) {
  // omega0 = sqrt(tension), tension = 10 ** tensionExp; f0 = omega0 / (2*pi)
  return Math.sqrt(10 ** tensionExp) / (2 * Math.PI);
}

// Compact inline sparkline of a node's 32 FIR taps, so a filter's rough shape
// (broadband/noisy vs. lopsided toward dry) is visible at a glance without a
// 32-column table.
function filterSparkline(taps) {
  const w = 64;
  const h = 20;
  const points = taps
    .map((v, i) => {
      const x = (i / (taps.length - 1)) * w;
      const y = h / 2 - v * (h / 2 - 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <line x1="0" y1="${h / 2}" x2="${w}" y2="${h / 2}" stroke="#232838" stroke-width="1" />
    <polyline points="${points}" fill="none" stroke="#7fd3ff" stroke-width="1.2" />
  </svg>`;
}

function renderNodeTable(container, title, nodeParams, showFilter) {
  const filterCols = showFilter ? '<th>filter mix (dry/filt)</th><th>filter taps</th>' : '';
  const rows = nodeParams
    .map((p, i) => {
      const filterCells = showFilter
        ? `
        <td>${p.filterMix[0].toFixed(2)} / ${p.filterMix[1].toFixed(2)}</td>
        <td>${filterSparkline(p.filterTaps)}</td>`
        : '';
      return `
      <tr>
        <td>${i}</td>
        <td>${p.mass.toFixed(2)}</td>
        <td>${p.tensionExp.toFixed(2)}</td>
        <td>${f0Hz(p.tensionExp).toFixed(1)}</td>
        <td>${p.dampLambda.toFixed(2)}</td>
        <td>${p.excitationGain.toFixed(2)}</td>
        <td style="color: ${p.gain > 1 ? '#ffb37f' : 'inherit'}">${p.gain.toFixed(2)}</td>
        ${filterCells}
      </tr>`;
    })
    .join('');
  container.innerHTML = `
    <h3>${title}</h3>
    <table>
      <thead>
        <tr><th>node</th><th>mass</th><th>tensionExp</th><th>f0 (Hz)</th><th>dampLambda</th><th>excGain</th><th>gain (tanh)</th>${filterCols}</tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderMatrix(container, title, matrix, lo, hi, decimals = 2) {
  const cells = matrix
    .map((row) => `
      <tr>${row
        .map((v) => {
          const t = (v - lo) / (hi - lo);
          const hue = t * 200; // red (low) -> cyan (high)
          return `<td style="background:hsl(${hue},60%,${18 + t * 20}%)">${v.toFixed(decimals)}</td>`;
        })
        .join('')}</tr>`)
    .join('');
  container.innerHTML = `
    <h3>${title}</h3>
    <table class="matrix">
      <tbody>${cells}</tbody>
    </table>`;
}

function renderParamsPanel(params) {
  renderNodeTable(document.getElementById('layer1Table'), 'Layer 1 (input / modulator)', params.layer1, true);
  renderNodeTable(document.getElementById('layer2Table'), 'Layer 2 (output / carrier)', params.layer2, true);
  renderMatrix(document.getElementById('tensionRouterGrid'), 'Tension router (row=layer2 node, col=layer1 node)', params.tensionRouter, ...TENSION_ROUTER_RANGE, 4);
  renderMatrix(document.getElementById('forceRouterGrid'), 'Force router (row=layer2 node, col=layer1 node)', params.forceRouter2, 0, 1.4);
}

function renderReverbInfo() {
  const container = document.getElementById('reverbInfo');
  if (!container) return;
  const { file, mix, loading, error } = state.reverb;
  const status = loading
    ? ' <span style="color:var(--muted)">(loading...)</span>'
    : error
      ? ' <span style="color:#ff8a8a">(failed to load -- dry only)</span>'
      : '';
  container.innerHTML = `
    <h3>Reverb</h3>
    <p style="margin:0;font-size:0.85rem">
      IR: <strong>${file || '(none yet)'}</strong>${status} &nbsp;&mdash;&nbsp; mix: ${mix.toFixed(2)}
    </p>`;

  const slider = document.getElementById('reverbMixSlider');
  if (slider) slider.value = mix;
}

function updateMarkerRow(selector, weights) {
  const markers = document.querySelectorAll(selector);
  markers.forEach((marker, j) => {
    const w = weights[j] || 0;
    const level = Math.min(1, w / STRIKE_PEAK);
    marker.style.opacity = 0.3 + level * 0.7;
    marker.style.transform = `translate(-50%, -50%) scale(${1 + level * 0.6})`;
  });
}

// 'waveform' (oscilloscope, time-domain) or 'spectrogram' (scrolling
// frequency-domain). Both read from the same AnalyserNode -- it exposes
// time-domain and frequency-domain views of the same underlying analysis
// independently, so no extra audio graph nodes are needed.
let scopeMode = 'waveform';

function clearScope() {
  const canvas = document.getElementById('scope');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#12141a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function setScopeMode(mode) {
  scopeMode = mode;
  clearScope(); // avoid a stale frame/scroll-trail from the previous mode
}

// Spectrogram frequency axis is log-scaled (20Hz - Nyquist), since the
// synth's audible content sits mostly in the lower part of the spectrum --
// a linear Hz axis would squash almost everything interesting into a thin
// strip at the bottom.
const SPECTROGRAM_MIN_FREQ = 20;

function freqBinColor(value) {
  // 0 -> panel background, 255 -> accent cyan; a single-hue intensity ramp
  const t = value / 255;
  const r = Math.round(0x12 + t * (0x7f - 0x12));
  const g = Math.round(0x14 + t * (0xd3 - 0x14));
  const b = Math.round(0x1a + t * (0xff - 0x1a));
  return `rgb(${r},${g},${b})`;
}

function drawScope() {
  const canvas = document.getElementById('scope');
  canvasCtx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const timeData = new Float32Array(analyser.fftSize);
  const freqData = new Uint8Array(analyser.frequencyBinCount);
  const logMin = Math.log10(SPECTROGRAM_MIN_FREQ);

  function drawWaveformFrame() {
    analyser.getFloatTimeDomainData(timeData);
    canvasCtx.fillStyle = '#12141a';
    canvasCtx.fillRect(0, 0, w, h);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#7fd3ff';
    canvasCtx.beginPath();
    for (let i = 0; i < timeData.length; i++) {
      const x = (i / timeData.length) * w;
      const y = h / 2 + timeData[i] * h * 0.45;
      if (i === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);
    }
    canvasCtx.stroke();
  }

  function drawFreqGridlines(nyquist, logMax) {
    canvasCtx.font = '10px sans-serif';
    for (const f of [100, 1000, 10000]) {
      if (f >= nyquist) continue;
      const t = (Math.log10(f) - logMin) / (logMax - logMin);
      const y = (1 - t) * (h - 1);
      canvasCtx.strokeStyle = 'rgba(232,236,241,0.2)';
      canvasCtx.beginPath();
      canvasCtx.moveTo(0, y);
      canvasCtx.lineTo(w, y);
      canvasCtx.stroke();
      canvasCtx.fillStyle = 'rgba(232,236,241,0.6)';
      canvasCtx.fillText(f >= 1000 ? `${f / 1000}kHz` : `${f}Hz`, 4, y - 2);
    }
  }

  function drawSpectrogramFrame() {
    analyser.getByteFrequencyData(freqData);
    // scroll existing image 1px left, then paint a fresh rightmost column
    canvasCtx.drawImage(canvas, -1, 0);
    const nyquist = audioCtx.sampleRate / 2;
    const logMax = Math.log10(nyquist);
    for (let y = 0; y < h; y++) {
      const t = 1 - y / (h - 1);
      const freq = Math.pow(10, logMin + t * (logMax - logMin));
      const bin = Math.min(freqData.length - 1, Math.round((freq / nyquist) * (freqData.length - 1)));
      canvasCtx.fillStyle = freqBinColor(freqData[bin]);
      canvasCtx.fillRect(w - 1, y, 1, 1);
    }
    drawFreqGridlines(nyquist, logMax);
  }

  function draw() {
    requestAnimationFrame(draw);
    if (scopeMode === 'waveform') drawWaveformFrame();
    else drawSpectrogramFrame();
  }
  draw();
}

function wireTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels = {
    play: document.getElementById('tab-play'),
    params: document.getElementById('tab-params'),
  };
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      buttons.forEach((b) => b.classList.toggle('active', b === btn));
      for (const [key, el] of Object.entries(panels)) el.hidden = key !== target;
    });
  });
}

// AR mode (ar-mode.js) is a separate ES module, loaded lazily via dynamic
// import only when actually opened, so its three.js/MediaPipe cost is never
// paid by default. It drives the existing plane objects' setPointer/
// clearPointer entry points (via window.DHOSynth, set in buildPlanes())
// exactly as real pointer events do -- it never touches worklet messages,
// markers, or the tent kernel directly.
function wireARMode() {
  const enterBtn = document.getElementById('enterArBtn');
  const overlay = document.getElementById('ar-overlay');
  const exitBtn = document.getElementById('ar-exit');
  let arModule = null;

  enterBtn.addEventListener('click', async () => {
    overlay.hidden = false;
    arModule = arModule || await import('./ar-mode.js');
    await arModule.startAR(overlay);
  });
  exitBtn.addEventListener('click', () => {
    arModule?.stopAR();
    overlay.hidden = true;
  });
}

// Mic input (mic-input.js) is a separate module, loaded lazily via dynamic
// import only when actually enabled, following the same pattern as AR mode.
// Unlike AR mode it's not a full-screen takeover: it's a toggle that hands
// the *energy* plane over to a live microphone's spectrogram (random-
// projected down to n1 dimensions, see mic-input.js) while the damping and
// tension-mod planes keep working exactly as before, via the normal 2D UI.
function wireMicInput() {
  const toggleBtn = document.getElementById('micInputBtn');
  let micModule = null;
  let active = false;

  toggleBtn.addEventListener('click', async () => {
    micModule = micModule || await import('./mic-input.js');
    if (active) {
      micModule.stopMicInput();
      toggleBtn.textContent = 'Enable Mic Input';
    } else {
      await micModule.startMicInput();
      toggleBtn.textContent = 'Disable Mic Input';
    }
    active = !active;
  });
}

function wireControls() {
  wireTabs();
  wireARMode();
  wireMicInput();

  const startBtn = document.getElementById('startBtn');
  const randomizeBtn = document.getElementById('randomizeBtn');
  const n1Slider = document.getElementById('n1Slider');
  const n2Slider = document.getElementById('n2Slider');
  const n1Label = document.getElementById('n1Label');
  const n2Label = document.getElementById('n2Label');
  const volumeSlider = document.getElementById('volumeSlider');
  const sparsitySlider = document.getElementById('sparsitySlider');
  const sparsityLabel = document.getElementById('sparsityLabel');
  const modDepthSlider = document.getElementById('modDepthSlider');
  const modDepthLabel = document.getElementById('modDepthLabel');
  const reverbMixSlider = document.getElementById('reverbMixSlider');
  const scopeModeBtn = document.getElementById('scopeModeBtn');
  const strategySelect = document.getElementById('strategySelect');
  const excitationModeSelect = document.getElementById('excitationModeSelect');
  const pulseRateSlider = document.getElementById('pulseRateSlider');
  const pulseRateLabel = document.getElementById('pulseRateLabel');
  const shareBtn = document.getElementById('shareBtn');

  // Restore UI state (not audio params yet -- those need workletNode, which
  // doesn't exist until ensureAudio() runs on the first Start-audio click;
  // see the `loadedShare` branch there) from a shared link, if one was parsed
  // at script load.
  if (loadedShare) {
    state.n1 = loadedShare.n1;
    state.n2 = loadedShare.n2;
    state.strategy = loadedShare.strategy;
    state.excitationMode = loadedShare.excitationMode;
    state.pulseRate = loadedShare.pulseRate;
    state.sparsity = loadedShare.sparsity;

    n1Slider.value = loadedShare.n1; n1Label.textContent = loadedShare.n1;
    n2Slider.value = loadedShare.n2; n2Label.textContent = loadedShare.n2;
    strategySelect.value = loadedShare.strategy;
    excitationModeSelect.value = loadedShare.excitationMode;
    pulseRateSlider.value = loadedShare.pulseRate;
    pulseRateLabel.textContent = `${loadedShare.pulseRate.toFixed(0)} Hz`;
    sparsitySlider.value = loadedShare.sparsity;
    sparsityLabel.textContent = loadedShare.sparsity.toFixed(2);
    volumeSlider.value = loadedShare.volume;
    modDepthSlider.value = loadedShare.modDepth;
    modDepthLabel.textContent = loadedShare.modDepth.toFixed(2);
    reverbMixSlider.value = loadedShare.reverbMix;
  }

  shareBtn.addEventListener('click', async () => {
    const url = buildShareUrl();
    if (!url) {
      shareBtn.textContent = 'Start audio first';
      setTimeout(() => { shareBtn.textContent = 'Share'; }, 1500);
      return;
    }
    try {
      await navigator.clipboard.writeText(url);
      shareBtn.textContent = 'Link copied!';
    } catch (err) {
      window.prompt('Copy this link:', url);
      shareBtn.textContent = 'Share';
      return;
    }
    setTimeout(() => { shareBtn.textContent = 'Share'; }, 1500);
  });

  strategySelect.addEventListener('change', () => {
    state.strategy = strategySelect.value;
    randomizeAll();
  });

  excitationModeSelect.addEventListener('change', () => {
    state.excitationMode = excitationModeSelect.value;
    if (workletNode) {
      workletNode.port.postMessage({ type: 'setExcitationMode', mode: state.excitationMode });
    }
  });

  pulseRateSlider.addEventListener('input', () => {
    state.pulseRate = parseFloat(pulseRateSlider.value);
    pulseRateLabel.textContent = `${state.pulseRate.toFixed(0)} Hz`;
    if (workletNode) {
      workletNode.port.postMessage({ type: 'setPulseRate', rate: state.pulseRate });
    }
  });

  scopeModeBtn.addEventListener('click', () => {
    const next = scopeMode === 'waveform' ? 'spectrogram' : 'waveform';
    setScopeMode(next);
    scopeModeBtn.textContent = next === 'waveform' ? 'Spectrogram' : 'Waveform';
  });

  startBtn.addEventListener('click', async () => {
    await ensureAudio();
    if (audioCtx.state === 'suspended') await audioCtx.resume();
    startBtn.textContent = 'Audio running';
    startBtn.disabled = true;
  });

  randomizeBtn.addEventListener('click', () => {
    randomizeAll();
  });

  function reconfigure() {
    state.n1 = parseInt(n1Slider.value, 10);
    state.n2 = parseInt(n2Slider.value, 10);
    n1Label.textContent = state.n1;
    n2Label.textContent = state.n2;
    resetPlanes();
    if (workletNode) {
      workletNode.port.postMessage({ type: 'configure', n1: state.n1, n2: state.n2 });
      randomizeAll();
    }
  }
  n1Slider.addEventListener('input', reconfigure);
  n2Slider.addEventListener('input', reconfigure);

  volumeSlider.addEventListener('input', () => {
    if (!workletNode) return;
    workletNode.parameters.get('masterGain').setValueAtTime(
      parseFloat(volumeSlider.value), audioCtx.currentTime);
  });

  sparsitySlider.addEventListener('input', () => {
    state.sparsity = parseFloat(sparsitySlider.value);
    sparsityLabel.textContent = state.sparsity.toFixed(2);
    regenerateRouters();
  });

  modDepthSlider.addEventListener('input', () => {
    const depth = parseFloat(modDepthSlider.value);
    modDepthLabel.textContent = depth.toFixed(2);
    if (workletNode) {
      workletNode.parameters.get('modDepth').setValueAtTime(depth, audioCtx.currentTime);
    }
  });

  reverbMixSlider.addEventListener('input', () => {
    const mix = parseFloat(reverbMixSlider.value);
    state.reverb.mix = mix;
    applyReverbMix(mix);
    renderReverbInfo();
  });

  buildPlanes();
}

window.addEventListener('DOMContentLoaded', wireControls);
