// AudioWorkletProcessor implementing a two-layer bank of damped harmonic
// oscillators (DHOs), where layer 1's raw output modulates layer 2's tension
// through a learned-style routing matrix -- a lightweight, pseudo-physics
// stand-in for spring-mass FM synthesis. The raw control-plane weights that
// drive excitation are themselves routed through a per-layer n x n matrix
// before reaching each layer's nodes (forceRouter1 for layer 1, forceRouter2
// for layer 2's cross-layer excitation) -- mirroring parallel.py, where
// every layer (not just the last) routes the shared control signal through
// its own force_router, rather than layer 1 getting a direct, unrouted pass.
//
// Per-node recurrence (mirrors `damped_harmonic_oscillator` /
// `execute_parallel_layer` in parallel.py), adapted for real-time use:
//
//   energy[n]   = (force[n] / mass + energy[n-1]) * decayCoef   // leaky integrator "excitation" envelope
//   shapeAlpha  = 1 / (2 * mass)                                // fixed unit "static damping" term
//   tensionEff  = 10 ** (tensionExp + tensionMod [+ routed tension from layer 1])
//   omega       = sqrt(|tensionEff - shapeAlpha^2|)
//   phi         = atan2(shapeAlpha, omega)
//   phase[n]    = phase[n-1] + omega * dt                       // integrated, not omega * absolute time
//   z           = (1 / cos(phi)) * energy * cos(phase - phi)
//
// `decayCoef` is derived from a continuous-time decay rate (1/sec) so envelope
// timing stays correct regardless of sample rate, unlike the training code's
// synthetic, clip-length-relative time axis. Likewise, phase is integrated
// per-node rather than computed as omega * absolute elapsed time (which is
// what the training code does, fine for a bounded offline clip): with a
// continuously-running real-time instrument, any change in omega -- from
// tensionMod, routed FM, anything -- would otherwise jump the cos() argument
// by roughly Δomega * t, producing an audible click that gets worse the
// longer the instrument has been running.
//
// Each layer-2 node's raw z is then passed through its own random 32-tap FIR
// filter (`instrument.filters` / `fft_convolve` in parallel.py -- done here as
// a plain time-domain convolution since 32 taps is cheap and, unlike an IIR
// design, is unconditionally stable for arbitrary random coefficients), dry/
// filtered-blended via a per-node mix (`instrument.filters_mix`), and *then*
// passed through its own gain + tanh (`torch.tanh(x * instrument.gains[...,
// None])`) before being summed into the output -- filter and nonlinearity are
// both per-node, not single stages applied to the mix.

const FILTER_TAPS = 32;

class Node {
  constructor() {
    this.mass = 1;
    this.tensionExp = 5;
    this.tensionMod = 0;
    this.dampLambda = 2;
    this.dampMod = 0;
    this.excitationGain = 1;
    // per-node gain feeding a tanh saturation, matching `instrument.gains` /
    // `torch.tanh(x * instrument.gains[..., None])` in parallel.py -- applied
    // to each node's own output *before* summing, not as one final limiter.
    this.gain = 1;
    this.energy = 0;
    // running phase accumulator (see stepNode) rather than omega * absolute
    // time -- keeps the waveform continuous across any change in omega.
    this.phase = 0;

    // identity filter (all-zero except tap 0) and dry-only mix by default,
    // so an un-randomized node passes its signal through unchanged.
    this.filterTaps = new Float32Array(FILTER_TAPS);
    this.filterTaps[0] = 1;
    this.filterMix = [1, 0]; // [dry, filtered]
    this.filterHistory = new Float32Array(FILTER_TAPS);
    this.filterPos = 0;

    // 'resonator' engine state (see stepResonator): the complex phasor z,
    // and any strike force waiting to be added on the next sample
    this.zr = 0;
    this.zi = 0;
    this.pendingDrive = 0;
  }
}

function makeNodes(n) {
  const nodes = [];
  for (let i = 0; i < n; i++) nodes.push(new Node());
  return nodes;
}

// Converts a router sent from the main thread (array-of-arrays, row k =
// target node k's weights over source nodes j) into a flat, row-major
// Float32Array -- avoids the extra indirection/bounds-check layer V8 pays on
// nested arrays, and matches the layout process() indexes into.
function flattenRouter(rows, nRows, nCols) {
  const flat = new Float32Array(nRows * nCols);
  if (!rows) return flat;
  for (let k = 0; k < nRows; k++) {
    const row = rows[k];
    if (!row) continue;
    for (let j = 0; j < nCols; j++) flat[k * nCols + j] = row[j] || 0;
  }
  return flat;
}

// Guard against the phi -> +-90deg singularity (1/cos(phi) blowing up) that
// can occur near critical damping -- an existing property of the closed-form
// solution, not something introduced here.
const MIN_COS_PHI = 1e-3;

// Per-node excitation (and, separately, damping-boost) weights are driven
// spatially from the main thread -- clicking/tapping/sweeping the bottom half
// of the control plane targets specific layer-1 nodes' force, the top half
// targets their damping -- rather than either being a single global
// AudioParam. Both are smoothed here with a short one-pole filter to avoid
// zipper clicks when the target vector jumps between messages.
const WEIGHT_SMOOTHING_TAU = 0.003;

// Additional decay rate (1/sec) added at full press-weight when the top
// "increase damping" zone targets a node -- large enough to noticeably
// quicken even a node at the slow end of its natural dampLambda range
// (~0.3, a multi-second sustain) down to a short, percussive decay.
const DAMP_MOD_MAX = 100;

// Additional tensionExp (log10 of tension = omega0^2) added at full
// press-weight when the "tension mod" zone targets a node -- a "deformation"
// gesture: pressing always *raises* tension/pitch, releasing lets it settle
// back to the node's base value. 0.2 in log10(tension) works out to roughly
// a fourth pitch bend at full press (tension is omega0^2, so a pitch ratio r
// needs a tension-exponent delta of 2*log10(r)) -- turned down a bit from
// 0.3 (roughly a fourth-to-fifth) so bends read as less dramatic by default.
const TENSION_MOD_MAX = 0.2;

// Excitation modes shape *how* energy plane weights turn into a driving
// signal for layer 1's leaky-integrator envelope:
//   sustained -- the original behavior: a continuous force proportional to
//     the (smoothed) weight, like bowing/blowing with steady pressure.
//   impulse   -- no continuous force at all; energy is instead bumped
//     directly and instantaneously (a 'strike' message from the main thread,
//     sent on press and on crossing into a new node while dragging), then
//     free-decays per the node's own dampLambda -- a plucked/struck model.
//   pulse     -- also no continuous force; instead the processor itself
//     injects periodic energy bumps at `pulseRate`, amplitude-scaled by
//     whichever nodes are currently held -- a bowed-string stick-slip model.
//   noise     -- the same continuous drive as 'sustained', but multiplied by
//     fresh white noise each sample instead of being a smooth push -- a
//     breath/turbulence model (wind instruments).
const STRIKE_ENERGY_SCALE = 3;
const PULSE_ENERGY_SCALE = 0.4;

// Engines -- which model the processor runs, set per instrument via
// setParams' `engine` (see defaultEngine):
//   legacy    -- the original web-native model above (envelope x cosine,
//                continuous force), used for instruments randomized here
//                and for v2 share links.  Behavior unchanged, apart from
//                the omega floor/ceiling below.
//   envelope  -- an instrument trained in parallel.py with --integrator
//                control_rate: the same envelope x cosine oscillator, but
//                with parallel.py's output stages (layer 1's tanh(gain) *
//                output_gain before it becomes FM, layer 2 summed with its
//                output_gain rather than averaged) and force scaling.
//   resonator -- trained with --integrator sample_rate: each node is a
//                true resonator, z = exp(-alpha + i*omega) * z + drive /
//                mass, output Re(z) + (x / omega) Im(z) (parallel.py's
//                sample_rate_oscillator), driven by the trained mix of
//                strike / noise / bow excitations.
// Trained engines also carry the training run's clock (timeScale, the
// factor between parallel.py's synthetic time and real seconds, and
// trainSampleRate) so the oscillator's static damping term and the
// excitations' energy match training, and `forceScale`, the trained force
// a full press (weight 1) maps to.
function defaultEngine() {
  return {
    type: 'legacy',
    trainSampleRate: 22050,
    controlRate: 512,
    timeScale: 1,
    forceScale: 1,
    excitationMix: [1, 0, 0], // [strike, noise, bow]
    bowHz: 100,
    outputGain1: 1,
    outputGain2: 1,
  };
}

// parallel.py's MIN_OMEGA_HZ: the lowest frequency a node rings at
const MIN_OMEGA_HZ = 20;
// parallel.py caps omega at 0.95 of its own (training) Nyquist
const MAX_OMEGA_FRACTION = 0.95;

// the polyBLEP correction for a unit step at phase 0 (sample_rate_dho.poly_blep)
function polyBlep(t, dt) {
  if (t < dt) {
    const a = t / dt;
    return 2 * a - a * a - 1;
  }
  if (t > 1 - dt) {
    const b = (t - 1) / dt;
    return b * b + 2 * b + 1;
  }
  return 0;
}

// A trained instrument's FIR taps are defined at the training rate; at
// another playback rate the same taps would span a different length of
// time and filter a different band.  Resample them (linearly -- they're
// low-pass by construction, see compute_filters) to the playback rate,
// scaled by 1 / ratio so the passband gain stays the same.
function resampleTaps(taps, ratio) {
  if (Math.abs(ratio - 1) < 1e-6) return Float32Array.from(taps);
  const m = Math.max(1, Math.round(taps.length * ratio));
  const out = new Float32Array(m);
  for (let t = 0; t < m; t++) {
    const pos = t / ratio;
    const i = Math.floor(pos);
    const frac = pos - i;
    const a = i < taps.length ? taps[i] : 0;
    const b = i + 1 < taps.length ? taps[i + 1] : 0;
    out[t] = (a + frac * (b - a)) / ratio;
  }
  return out;
}

// Training never produced anything above its own Nyquist (FM sidebands
// above it alias back down there), so when playing faster than the
// training rate, a trained instrument's output is low-passed there: a
// 4th-order Butterworth, as two biquads.  null when not needed.
const BUTTERWORTH_Q = [0.5412, 1.3066];
function makeOutputFilter(engine) {
  if (engine.type === 'legacy' || sampleRate <= engine.trainSampleRate * 1.01) return null;
  const w0 = (2 * Math.PI * 0.5 * engine.trainSampleRate) / sampleRate;
  const cos = Math.cos(w0);
  return BUTTERWORTH_Q.map((q) => {
    const alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    return {
      b0: (1 - cos) / 2 / a0, b1: (1 - cos) / a0, b2: (1 - cos) / 2 / a0,
      a1: (-2 * cos) / a0, a2: (1 - alpha) / a0,
      s1: 0, s2: 0,
    };
  });
}
function applyOutputFilter(stages, x) {
  if (!stages) return x;
  for (const st of stages) {
    // transposed direct form II
    const y = st.b0 * x + st.s1;
    st.s1 = st.b1 * x - st.a1 * y + st.s2;
    st.s2 = st.b2 * x - st.a2 * y;
    x = y;
  }
  return x;
}

// Linear below LIMIT_KNEE, then a tanh shoulder up to 1: a safety limiter
// for the trained engines' output, which (unlike legacy's per-node-average
// mix) is meant to peak around 1, where a plain tanh would already be
// audibly saturating.
const LIMIT_KNEE = 0.8;
function softLimit(x) {
  const a = Math.abs(x);
  if (a <= LIMIT_KNEE) return x;
  const y = LIMIT_KNEE + (1 - LIMIT_KNEE) * Math.tanh((a - LIMIT_KNEE) / (1 - LIMIT_KNEE));
  return x < 0 ? -y : y;
}

class DHOProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        // pre-tanh multiplier on the final mix -- since it's applied before
        // the closing soft clip, a large value doesn't just get "too loud",
        // it saturates into a gentle compressor/limiter, which is exactly
        // what's needed to compensate for how quiet some settings can be
        // (high sparsity, unlucky random gain/damping draws, etc.) without
        // risking clipping/instability -- tanh bounds the output regardless
        // of how large this gets.
        name: 'masterGain',
        defaultValue: 0.6,
        minValue: 0,
        maxValue: 10,
        automationRate: 'k-rate',
      },
      {
        // input gain: scales every force entering the instrument --
        // sustained/noise forces, strikes and pulses, on both layers --
        // before the nodes, so (unlike masterGain, which comes after each
        // node's tanh) it sets how hard the per-node tanh saturations are
        // driven.  For matching a trained instrument's operating level: if
        // the demo's control levels are louder than the training data's,
        // the same instrument distorts here when it didn't in training.
        // Everything up to the per-node tanh is linear in the forces, so
        // this scales layer 1's output, and with it the FM it applies to
        // layer 2, by the same factor (as louder input would in training).
        name: 'inputGain',
        defaultValue: 1,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate',
      },
      {
        // live scalar on the routed tension-modulation (FM) contribution --
        // 1 = depth as generated by tensionRouter, down to 0.1 = mostly
        // tamed, up to 10 = well beyond the random router range for
        // deliberately extreme FM. TENSION_ROUTER_RANGE in main.js still
        // sets what a "normal" randomized depth looks like at modDepth=1.
        name: 'modDepth',
        defaultValue: 1,
        minValue: 0.1,
        maxValue: 10,
        automationRate: 'k-rate',
      },
    ];
  }

  constructor() {
    super();

    this.n1 = 8;
    this.n2 = 8;
    this.layer1 = makeNodes(this.n1);
    this.layer2 = makeNodes(this.n2);
    // Routing matrices are flat, row-major Float32Arrays (row k = target
    // node k's weights over source nodes j, i.e. router[k * nCols + j])
    // rather than arrays-of-arrays -- see flattenRouter.
    // tensionRouter: influence of layer1 node j's output on layer2 node k's tension
    this.tensionRouter = new Float32Array(this.n2 * this.n1);
    // forceRouter1: layer 1's own self-routing -- how much raw control
    // weight j (the same weight every layer routes from) drives layer-1
    // node k's own excitation. Mirrors parallel.py, where *every* layer
    // (not just the last) routes the shared control signal through its own
    // n x n force_router.
    this.forceRouter1 = new Float32Array(this.n1 * this.n1);
    // forceRouter2: how much pressing/targeting layer1 node j drives layer2 node k's excitation
    this.forceRouter2 = new Float32Array(this.n2 * this.n1);

    this.targetWeights = new Float32Array(this.n1);
    // routedForce1/routedForce2 are forceRouter1 @ targetWeights and
    // forceRouter2 @ targetWeights, smoothed every sample with the same
    // one-pole filter that used to be applied to the raw weights before
    // routing. This is valid because the routing matrix and the smoothing
    // recurrence are both linear, so they commute: smoothing the routed
    // output converges to the same place, sample-for-sample, as routing the
    // smoothed input would. That lets the O(n^2) matvec itself
    // (recomputeRoutedForce) run only when targetWeights or the router
    // matrices change -- i.e. at UI-event rate -- instead of 48000+ times/sec
    // in process() while the target sits unchanged (the common case of
    // holding a note rather than actively dragging).
    this.routedForce1Target = new Float32Array(this.n1);
    this.routedForce1 = new Float32Array(this.n1);
    this.routedForce2Target = new Float32Array(this.n2);
    this.routedForce2 = new Float32Array(this.n2);
    this.targetDampWeights = new Float32Array(this.n1);
    this.currentDampWeights = new Float32Array(this.n1);
    // a second, independent damping-boost strip targeting layer 2 directly
    this.targetDampWeights2 = new Float32Array(this.n2);
    this.currentDampWeights2 = new Float32Array(this.n2);
    // tension-mod ("deformation") strips, one per layer, sharing each
    // layer's damping strip but as its own bottom zone
    this.targetTensionModWeights = new Float32Array(this.n1);
    this.currentTensionModWeights = new Float32Array(this.n1);
    this.targetTensionModWeights2 = new Float32Array(this.n2);
    this.currentTensionModWeights2 = new Float32Array(this.n2);
    this.weightSmoothing = 1 - Math.exp(-1 / (sampleRate * WEIGHT_SMOOTHING_TAU));

    // latest inputGain AudioParam value, kept for 'strike' messages, which
    // are handled between process() calls and so can't read parameters
    this.inputGain = 1;

    this.excitationMode = 'sustained';
    this.pulseRate = 100; // Hz
    this.pulsePhase = 0;

    this.engine = defaultEngine();
    this.outputFilter = null;
    // the 'resonator' engine's bow (sawtooth) phase, in cycles -- one bow
    // for the whole instrument, as in parallel.py
    this.bowPhase = 0;

    this.port.onmessage = (event) => this.handleMessage(event.data);
  }

  handleMessage(msg) {
    if (msg.type === 'configure') {
      this.n1 = msg.n1;
      this.n2 = msg.n2;
      this.layer1 = makeNodes(this.n1);
      this.layer2 = makeNodes(this.n2);
      this.tensionRouter = new Float32Array(this.n2 * this.n1);
      this.forceRouter1 = new Float32Array(this.n1 * this.n1);
      this.forceRouter2 = new Float32Array(this.n2 * this.n1);
      this.targetWeights = new Float32Array(this.n1);
      this.routedForce1Target = new Float32Array(this.n1);
      this.routedForce1 = new Float32Array(this.n1);
      this.routedForce2Target = new Float32Array(this.n2);
      this.routedForce2 = new Float32Array(this.n2);
      this.targetDampWeights = new Float32Array(this.n1);
      this.currentDampWeights = new Float32Array(this.n1);
      this.targetDampWeights2 = new Float32Array(this.n2);
      this.currentDampWeights2 = new Float32Array(this.n2);
      this.targetTensionModWeights = new Float32Array(this.n1);
      this.currentTensionModWeights = new Float32Array(this.n1);
      this.targetTensionModWeights2 = new Float32Array(this.n2);
      this.currentTensionModWeights2 = new Float32Array(this.n2);
    } else if (msg.type === 'setParams') {
      const p = msg.params;
      if (p.engine) {
        // a different model's state means nothing to the new one -- start silent
        this.engine = { ...defaultEngine(), ...p.engine };
        this.outputFilter = makeOutputFilter(this.engine);
        for (const node of [...this.layer1, ...this.layer2]) {
          node.energy = 0;
          node.zr = 0;
          node.zi = 0;
          node.pendingDrive = 0;
        }
      }
      this.applyNodeParams(this.layer1, p.layer1);
      this.applyNodeParams(this.layer2, p.layer2);
      if (p.tensionRouter) this.tensionRouter = flattenRouter(p.tensionRouter, this.n2, this.n1);
      if (p.forceRouter1) this.forceRouter1 = flattenRouter(p.forceRouter1, this.n1, this.n1);
      if (p.forceRouter2) this.forceRouter2 = flattenRouter(p.forceRouter2, this.n2, this.n1);
      this.recomputeRoutedForce();
    } else if (msg.type === 'excite') {
      for (let j = 0; j < this.n1 && j < msg.weights.length; j++) {
        this.targetWeights[j] = msg.weights[j];
      }
      this.recomputeRoutedForce();
    } else if (msg.type === 'dampen') {
      const target = msg.layer === 2 ? this.targetDampWeights2 : this.targetDampWeights;
      const n = msg.layer === 2 ? this.n2 : this.n1;
      for (let j = 0; j < n && j < msg.weights.length; j++) {
        target[j] = msg.weights[j];
      }
    } else if (msg.type === 'tensionmod') {
      const target = msg.layer === 2 ? this.targetTensionModWeights2 : this.targetTensionModWeights;
      const n = msg.layer === 2 ? this.n2 : this.n1;
      for (let j = 0; j < n && j < msg.weights.length; j++) {
        target[j] = msg.weights[j];
      }
    } else if (msg.type === 'strike' && this.engine.type !== 'legacy') {
      // trained engines: a strike is parallel.py's impulse excitation -- the
      // routed force, landing on *both* layers (each through its own force
      // router, as every layer routes the shared forces in parallel.py).
      // The resonator weights it by the trained strike share and adds it
      // as the next sample's drive; the envelope engine adds force / mass
      // to the envelope, as one frame of parallel.py's `sequential` does.
      const eng = this.engine;
      const scale = this.inputGain * eng.forceScale;
      const layers = [[this.layer1, this.forceRouter1], [this.layer2, this.forceRouter2]];
      for (const [nodes, router] of layers) {
        for (let k = 0; k < nodes.length; k++) {
          let routed = 0;
          const base = k * this.n1;
          for (let j = 0; j < this.n1 && j < msg.weights.length; j++) {
            routed += router[base + j] * msg.weights[j];
          }
          const force = routed * scale * nodes[k].excitationGain;
          if (eng.type === 'resonator') {
            nodes[k].pendingDrive += force * eng.excitationMix[0];
          } else {
            nodes[k].energy += force / nodes[k].mass;
          }
        }
      }
    } else if (msg.type === 'strike') {
      // one-shot energy bump for 'impulse' mode -- applied immediately
      // (handleMessage runs on the audio thread between process() calls),
      // not stored as a continuous target the way 'excite' weights are.
      // Routed through forceRouter1 like every other layer-1 excitation
      // pathway (see process()'s layer-1 loop), so a strike on node j can
      // also land on whichever nodes forceRouter1 couples it to.
      for (let k = 0; k < this.n1; k++) {
        let routed = 0;
        const base = k * this.n1;
        for (let j = 0; j < this.n1 && j < msg.weights.length; j++) {
          routed += this.forceRouter1[base + j] * msg.weights[j];
        }
        this.layer1[k].energy += routed * STRIKE_ENERGY_SCALE * this.inputGain;
      }
    } else if (msg.type === 'setExcitationMode') {
      this.excitationMode = msg.mode;
    } else if (msg.type === 'setPulseRate') {
      this.pulseRate = msg.rate;
    }
  }

  applyNodeParams(nodes, params) {
    if (!params) return;
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const src = params[i];
      if (!src) continue;
      if (src.mass !== undefined) n.mass = src.mass;
      if (src.tensionExp !== undefined) n.tensionExp = src.tensionExp;
      if (src.tensionMod !== undefined) n.tensionMod = src.tensionMod;
      if (src.dampLambda !== undefined) n.dampLambda = src.dampLambda;
      if (src.dampMod !== undefined) n.dampMod = src.dampMod;
      if (src.excitationGain !== undefined) n.excitationGain = src.excitationGain;
      if (src.gain !== undefined) n.gain = src.gain;
      if (src.filterTaps !== undefined) {
        n.filterTaps = this.engine.type === 'legacy'
          ? Float32Array.from(src.filterTaps)
          : resampleTaps(src.filterTaps, sampleRate / this.engine.trainSampleRate);
        n.filterHistory = new Float32Array(n.filterTaps.length);
        n.filterPos = 0;
      }
      if (src.filterMix !== undefined) n.filterMix = src.filterMix;
    }
  }

  // Recomputes the O(n^2) routed-force targets from the current (raw) target
  // weights and router matrices. Called on 'excite'/'setParams' -- i.e. at
  // UI-event rate -- not from process(), which just smooths routedForce1/
  // routedForce2 toward these targets every sample (see their declaration
  // in the constructor for why that's equivalent to routing the smoothed
  // weights directly).
  recomputeRoutedForce() {
    for (let k = 0; k < this.n1; k++) {
      let sum = 0;
      const base = k * this.n1;
      for (let j = 0; j < this.n1; j++) sum += this.forceRouter1[base + j] * this.targetWeights[j];
      this.routedForce1Target[k] = sum;
    }
    for (let k = 0; k < this.n2; k++) {
      let sum = 0;
      const base = k * this.n1;
      for (let j = 0; j < this.n1; j++) sum += this.forceRouter2[base + j] * this.targetWeights[j];
      this.routedForce2Target[k] = sum;
    }
  }

  applyFilter(node, x) {
    // usually FILTER_TAPS long, but a trained instrument's taps are
    // resampled to the playback rate (see applyNodeParams)
    const taps = node.filterTaps;
    const m = taps.length;
    node.filterHistory[node.filterPos] = x;
    let filt = 0;
    for (let i = 0; i < m; i++) {
      const idx = (node.filterPos - i + m) % m;
      filt += taps[i] * node.filterHistory[idx];
    }
    node.filterPos = (node.filterPos + 1) % m;
    return filt;
  }

  stepNode(node, force, dt) {
    // Zero-order-hold discretization of d(energy)/dt = -lambda*energy + force/mass,
    // so steady-state energy stays sample-rate independent (see
    // `parallel_sr_independent` in parallel.py). Using the training code's
    // per-control-frame recurrence directly at full audio rate would integrate
    // the driving force far too often and blow up.
    // dampMod (a per-node base value, plus any live gesture-driven boost
    // added by the caller) shifts the decay rate additively before use,
    // matching `damp = torch.abs(damping + damp_mod)` in parallel.py.
    const lambda = Math.abs(node.dampLambda + node.dampMod);
    const alpha = Math.exp(-lambda * dt);
    const beta = lambda > 1e-8 ? (1 - alpha) / lambda : dt;
    node.energy = alpha * node.energy + beta * (force / node.mass);

    const shapeAlpha = this.engine.timeScale / (2 * node.mass);
    const omega = this.nodeOmega(node, shapeAlpha, dt);
    const phi = Math.atan2(shapeAlpha, omega);

    let cosPhi = Math.cos(phi);
    if (Math.abs(cosPhi) < MIN_COS_PHI) {
      cosPhi = cosPhi < 0 ? -MIN_COS_PHI : MIN_COS_PHI;
    }
    const a = 1 / cosPhi;

    // Integrate phase (phase += omega*dt) rather than using omega * absolute
    // elapsed time, as the training code does for whole fixed-length clips.
    // The latter makes the cos() argument jump by roughly Δomega * t whenever
    // omega changes -- harmless for a bounded offline clip, but produces an
    // audible click on any tension change (tensionMod, routed FM, ...) in a
    // continuously-running real-time instrument, worse the longer it's been
    // running. Integrating the increment each sample keeps the waveform
    // phase-continuous regardless of how omega varies over time.
    node.phase += omega * dt;
    return a * node.energy * Math.cos(node.phase - phi);
  }

  // A node's angular frequency in rad/s: sqrt(|tension - shapeAlpha^2|),
  // floored at MIN_OMEGA_HZ and capped below Nyquist -- parallel.py's
  // compute_omega.  Trained engines cap at the *training* run's Nyquist,
  // since the model never heard anything above it (and running a node
  // above the playback Nyquist would alias).
  nodeOmega(node, shapeAlpha, dt) {
    const tensionEff = Math.pow(10, node.tensionExp + node.tensionMod);
    let omega = Math.sqrt(Math.abs(tensionEff - shapeAlpha * shapeAlpha));
    const nyquist = this.engine.type === 'legacy'
      ? 0.5 / dt
      : Math.min(0.5 / dt, 0.5 * this.engine.trainSampleRate);
    const maxOmega = 2 * Math.PI * MAX_OMEGA_FRACTION * nyquist;
    const minOmega = 2 * Math.PI * MIN_OMEGA_HZ;
    if (omega < minOmega) omega = minOmega;
    if (omega > maxOmega) omega = maxOmega;
    return omega;
  }

  // 'resonator' engine: one sample of parallel.py's sample_rate_oscillator,
  //   z[n] = exp(-alpha + i * omega * dt) * z[n-1] + drive[n] / mass
  //   out  = Re(z) + (x / omega) * Im(z),   x = timeScale / (2 * mass)
  // alpha is the node's decay rate (dampLambda, 1/sec, plus any gesture
  // boost) per sample.  x and omega are both in real-time units here (the
  // share export scales tension by timeScale^2), so x / omega is the same
  // ratio as in training.
  stepResonator(node, drive, dt) {
    const lambda = Math.abs(node.dampLambda + node.dampMod);
    const decay = Math.exp(-lambda * dt);
    const shapeAlpha = this.engine.timeScale / (2 * node.mass);
    const omega = this.nodeOmega(node, shapeAlpha, dt);
    const c = Math.cos(omega * dt) * decay;
    const s = Math.sin(omega * dt) * decay;
    const b = (drive + node.pendingDrive) / node.mass;
    node.pendingDrive = 0;
    const zr = c * node.zr - s * node.zi + b;
    const zi = s * node.zr + c * node.zi;
    node.zr = zr;
    node.zi = zi;
    return zr + (shapeAlpha / omega) * zi;
  }

  process(inputs, outputs, parameters) {
    if (this.engine.type !== 'legacy') return this.processTrained(outputs, parameters);

    const output = outputs[0];
    const channel = output[0];
    const masterGain = parameters.masterGain[0];
    const modDepth = parameters.modDepth[0];
    const inputGain = parameters.inputGain[0];
    this.inputGain = inputGain;
    const dt = 1 / sampleRate;

    for (let i = 0; i < channel.length; i++) {
      // 'pulse' mode ticks on its own clock, independent of any incoming
      // messages -- each tick is a periodic energy bump (see below) rather
      // than a continuous force, amplitude-scaled by whichever nodes are
      // currently held.
      let pulseTick = false;
      if (this.excitationMode === 'pulse') {
        this.pulsePhase += this.pulseRate * dt;
        if (this.pulsePhase >= 1) {
          this.pulsePhase -= 1;
          pulseTick = true;
        }
      }

      const z1 = new Array(this.n1);
      for (let j = 0; j < this.n1; j++) {
        this.routedForce1[j] +=
          (this.routedForce1Target[j] - this.routedForce1[j]) * this.weightSmoothing;
        this.currentDampWeights[j] +=
          (this.targetDampWeights[j] - this.currentDampWeights[j]) * this.weightSmoothing;
        this.currentTensionModWeights[j] +=
          (this.targetTensionModWeights[j] - this.currentTensionModWeights[j]) * this.weightSmoothing;

        const node = this.layer1[j];

        // Layer 1 self-routes the same raw control weights every layer
        // routes from -- a control-plane press on node j can excite other
        // layer-1 nodes too, not just itself, mirroring forceRouter2 below
        // (and parallel.py's per-layer force_router, applied to every
        // layer, not just the last). The O(n^2) matvec itself runs in
        // recomputeRoutedForce(); routedForce1[j] above is just a cheap O(n)
        // smoothed read of that result.
        const routedForce = this.routedForce1[j] * inputGain;

        // 'impulse' and 'pulse' modes drive energy directly (a one-shot bump
        // from a 'strike' message, or a periodic tick below) rather than
        // through a continuous force -- so `force` stays 0 for them here.
        let force = 0;
        if (this.excitationMode === 'sustained') {
          force = routedForce * node.excitationGain;
        } else if (this.excitationMode === 'noise') {
          force = routedForce * node.excitationGain * (Math.random() * 2 - 1);
        } else if (this.excitationMode === 'pulse' && pulseTick) {
          node.energy += routedForce * node.excitationGain * PULSE_ENERGY_SCALE;
        }

        // "increase damping" targeting adds a temporary boost on top of the
        // node's own (randomized) base dampMod, restored right after -- same
        // save/restore pattern used for layer 2's tensionMod below. Tension
        // mod ("deformation") works the same way: pressing always raises
        // tension/pitch, releasing lets it settle back to the base value.
        const savedDampMod = node.dampMod;
        node.dampMod = savedDampMod + this.currentDampWeights[j] * DAMP_MOD_MAX;
        const savedTensionMod = node.tensionMod;
        node.tensionMod = savedTensionMod + this.currentTensionModWeights[j] * TENSION_MOD_MAX;
        const raw1 = this.stepNode(node, force, dt);
        node.tensionMod = savedTensionMod;
        node.dampMod = savedDampMod;

        // layer 1's own FIR filter shapes the signal that goes on to modulate
        // layer 2's tension -- consistent with execute_parallel_layer, where
        // every layer's filter is applied before its output feeds the next
        // layer, not just the final/audible one. Layer 1 still skips the
        // gain/tanh stage, since only layer 2 is meant to saturate/be audible.
        const filt1 = this.applyFilter(node, raw1);
        z1[j] = node.filterMix[0] * raw1 + node.filterMix[1] * filt1;
      }

      let sample = 0;
      for (let k = 0; k < this.n2; k++) {
        const node = this.layer2[k];
        this.currentDampWeights2[k] +=
          (this.targetDampWeights2[k] - this.currentDampWeights2[k]) * this.weightSmoothing;
        this.currentTensionModWeights2[k] +=
          (this.targetTensionModWeights2[k] - this.currentTensionModWeights2[k]) * this.weightSmoothing;

        this.routedForce2[k] +=
          (this.routedForce2Target[k] - this.routedForce2[k]) * this.weightSmoothing;

        // which layer-1 node(s) are being targeted also determines how much
        // *force* (not just tension) each layer-2 node receives, via its own
        // routing row -- so different input nodes excite different parts of
        // layer 2, not just the same chord with a subtle tension tint. Routed
        // the same way as layer 1's routedForce above: the O(n^2) matvec
        // lives in recomputeRoutedForce(), this is just the smoothed read.
        const force = this.routedForce2[k] * node.excitationGain * inputGain;

        // tensionRouter routes z1 -- the actual audio-rate oscillator output
        // from layer 1, not a smoothable control value -- so this matvec
        // genuinely has to run every sample; there's no equivalent shortcut.
        let routed = 0;
        const tensionBase = k * this.n1;
        for (let j = 0; j < this.n1; j++) routed += this.tensionRouter[tensionBase + j] * z1[j];
        routed *= modDepth;

        // layer 2's own tension-mod ("deformation") strip adds directly to
        // the same tensionMod field that the routed FM contribution above
        // already lands in -- both are just additive shifts on the same sum.
        const savedMod = node.tensionMod;
        node.tensionMod = savedMod + routed + this.currentTensionModWeights2[k] * TENSION_MOD_MAX;
        // layer 2's own damping strip works the same way as layer 1's --
        // a temporary boost on top of the node's base dampMod, restored
        // right after this stepNode call.
        const savedDampMod2 = node.dampMod;
        node.dampMod = savedDampMod2 + this.currentDampWeights2[k] * DAMP_MOD_MAX;
        const z2 = this.stepNode(node, force, dt);
        node.dampMod = savedDampMod2;
        node.tensionMod = savedMod;

        // per-node FIR filter + dry/filtered blend, applied before the
        // nonlinearity (mirrors filters -> filters_mix -> tanh(gain) in
        // execute_parallel_layer).
        const filt = this.applyFilter(node, z2);
        const blended = node.filterMix[0] * z2 + node.filterMix[1] * filt;

        // only layer 2 (z2) is ever mixed into the audible output -- layer 1
        // (z1) exists purely to drive tension/force routing above and must
        // never be added into `sample` directly. Each node's own gain and
        // tanh saturation are applied here, per node, before summing --
        // mirroring `torch.tanh(x * instrument.gains[..., None])` in
        // parallel.py -- rather than a single nonlinearity on the mix.
        sample += Math.tanh(blended * node.gain);
      }

      // average rather than sum so overall loudness doesn't scale with node
      // count, then a final soft clip as a safety margin (not a modeled
      // nonlinearity -- the per-node tanh above is the faithful one).
      const mixed = this.n2 > 0 ? sample / Math.sqrt(this.n2) : 0;
      channel[i] = Math.tanh(mixed * masterGain);
    }

    // mirror to any additional output channels (e.g. stereo)
    for (let c = 1; c < output.length; c++) {
      output[c].set(channel);
    }
    return true;
  }

  // The trained engines ('envelope', 'resonator'; see defaultEngine):
  // parallel.py's LayerController.forward, per sample.  Both layers are
  // driven by the same control weights through their own force routers;
  // layer 1's output, tanh(filtered * gain) * output_gain, is routed into
  // layer 2's log10 tension as FM; layer 2's per-node tanh(filtered * gain)
  // * output_gain is summed (not averaged) into the output.
  processTrained(outputs, parameters) {
    const output = outputs[0];
    const channel = output[0];
    const masterGain = parameters.masterGain[0];
    const modDepth = parameters.modDepth[0];
    const inputGain = parameters.inputGain[0];
    this.inputGain = inputGain;
    const dt = 1 / sampleRate;
    const eng = this.engine;
    const resonator = eng.type === 'resonator';
    const forceScale = inputGain * eng.forceScale;

    // parallel.py scales noise and bow so one control frame of either
    // carries a strike's energy (1 / sqrt(control_rate), plus sqrt(3) to
    // make each unit variance).  Running faster than the training rate, a
    // resonator's per-sample decay is proportionally smaller, so it
    // accumulates more of a sustained drive: noise power by rateScale^-1,
    // a periodic (bow) drive's amplitude by rateScale^-1.  Undo both.
    const rateScale = eng.trainSampleRate / sampleRate;
    const sustainScale = 1 / Math.sqrt(eng.controlRate);
    const noiseScale = Math.sqrt(3) * sustainScale * Math.sqrt(rateScale) * eng.excitationMix[1];
    const bowScale = Math.sqrt(3) * sustainScale * rateScale * eng.excitationMix[2];
    // the bow's rate, capped as sample_rate_dho.sawtooth_excitation does
    // (to [1e-5, 0.5] cycles per *training* sample)
    const bowHz = Math.min(Math.max(eng.bowHz, 1e-5 * eng.trainSampleRate), 0.5 * eng.trainSampleRate);
    const bowInc = Math.min(bowHz * dt, 0.5);
    // the envelope engine's continuous force: parallel.py adds force / mass
    // to the envelope once per control frame, i.e. trainSampleRate /
    // controlRate times a second (see stepNode's units)
    const envelopeForceScale = eng.trainSampleRate / eng.controlRate;

    const z1 = this.z1 && this.z1.length === this.n1 ? this.z1 : (this.z1 = new Float64Array(this.n1));

    for (let i = 0; i < channel.length; i++) {
      let pulseTick = false;
      if (!resonator && this.excitationMode === 'pulse') {
        this.pulsePhase += this.pulseRate * dt;
        if (this.pulsePhase >= 1) {
          this.pulsePhase -= 1;
          pulseTick = true;
        }
      }

      let saw = 0;
      if (resonator && bowScale !== 0) {
        this.bowPhase += bowInc;
        this.bowPhase -= Math.floor(this.bowPhase);
        saw = 2 * this.bowPhase - 1 - polyBlep(this.bowPhase, bowInc);
      }

      // the drive for one node, from its routed (smoothed) control weight
      const step = (node, routed) => {
        const force = routed * forceScale * node.excitationGain;
        if (resonator) {
          let drive = 0;
          if (force !== 0) {
            drive = force * (noiseScale * (Math.random() * 2 - 1) + bowScale * saw);
          }
          return this.stepResonator(node, drive, dt);
        }
        let f = 0;
        if (this.excitationMode === 'sustained') {
          f = force * envelopeForceScale;
        } else if (this.excitationMode === 'noise') {
          f = force * envelopeForceScale * (Math.random() * 2 - 1);
        } else if (this.excitationMode === 'pulse' && pulseTick) {
          node.energy += force / node.mass;
        }
        return this.stepNode(node, f, dt);
      };

      for (let j = 0; j < this.n1; j++) {
        this.routedForce1[j] +=
          (this.routedForce1Target[j] - this.routedForce1[j]) * this.weightSmoothing;
        this.currentDampWeights[j] +=
          (this.targetDampWeights[j] - this.currentDampWeights[j]) * this.weightSmoothing;
        this.currentTensionModWeights[j] +=
          (this.targetTensionModWeights[j] - this.currentTensionModWeights[j]) * this.weightSmoothing;
        const node = this.layer1[j];

        const savedDampMod = node.dampMod;
        node.dampMod = savedDampMod + this.currentDampWeights[j] * DAMP_MOD_MAX;
        const savedTensionMod = node.tensionMod;
        node.tensionMod = savedTensionMod + this.currentTensionModWeights[j] * TENSION_MOD_MAX;
        const raw1 = step(node, this.routedForce1[j]);
        node.tensionMod = savedTensionMod;
        node.dampMod = savedDampMod;

        const filt1 = this.applyFilter(node, raw1);
        const blended1 = node.filterMix[0] * raw1 + node.filterMix[1] * filt1;
        z1[j] = Math.tanh(blended1 * node.gain) * eng.outputGain1;
      }

      let sample = 0;
      for (let k = 0; k < this.n2; k++) {
        const node = this.layer2[k];
        this.currentDampWeights2[k] +=
          (this.targetDampWeights2[k] - this.currentDampWeights2[k]) * this.weightSmoothing;
        this.currentTensionModWeights2[k] +=
          (this.targetTensionModWeights2[k] - this.currentTensionModWeights2[k]) * this.weightSmoothing;
        this.routedForce2[k] +=
          (this.routedForce2Target[k] - this.routedForce2[k]) * this.weightSmoothing;

        let routed = 0;
        const tensionBase = k * this.n1;
        for (let j = 0; j < this.n1; j++) routed += this.tensionRouter[tensionBase + j] * z1[j];
        routed *= modDepth;

        const savedMod = node.tensionMod;
        node.tensionMod = savedMod + routed + this.currentTensionModWeights2[k] * TENSION_MOD_MAX;
        const savedDampMod2 = node.dampMod;
        node.dampMod = savedDampMod2 + this.currentDampWeights2[k] * DAMP_MOD_MAX;
        const raw2 = step(node, this.routedForce2[k]);
        node.dampMod = savedDampMod2;
        node.tensionMod = savedMod;

        const filt = this.applyFilter(node, raw2);
        const blended = node.filterMix[0] * raw2 + node.filterMix[1] * filt;
        sample += Math.tanh(blended * node.gain);
      }

      const y = applyOutputFilter(this.outputFilter, sample * eng.outputGain2 * masterGain);
      channel[i] = softLimit(y);
    }

    for (let c = 1; c < output.length; c++) {
      output[c].set(channel);
    }

    return true;
  }
}

registerProcessor('dho-processor', DHOProcessor);
