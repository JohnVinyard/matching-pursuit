// AudioWorkletProcessor implementing a two-layer bank of damped harmonic
// oscillators (DHOs), where layer 1's raw output modulates layer 2's tension
// through a learned-style routing matrix -- a lightweight, pseudo-physics
// stand-in for spring-mass FM synthesis.
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
  }
}

function makeNodes(n) {
  const nodes = [];
  for (let i = 0; i < n; i++) nodes.push(new Node());
  return nodes;
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
    // tensionRouter[k][j]: influence of layer1 node j's output on layer2 node k's tension
    this.tensionRouter = this.layer2.map(() => this.layer1.map(() => 0));
    // forceRouter2[k][j]: how much pressing/targeting layer1 node j drives layer2 node k's excitation
    this.forceRouter2 = this.layer2.map(() => this.layer1.map(() => 0));

    this.targetWeights = new Float32Array(this.n1);
    this.currentWeights = new Float32Array(this.n1);
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

    this.excitationMode = 'sustained';
    this.pulseRate = 100; // Hz
    this.pulsePhase = 0;

    this.port.onmessage = (event) => this.handleMessage(event.data);
  }

  handleMessage(msg) {
    if (msg.type === 'configure') {
      this.n1 = msg.n1;
      this.n2 = msg.n2;
      this.layer1 = makeNodes(this.n1);
      this.layer2 = makeNodes(this.n2);
      this.tensionRouter = this.layer2.map(() => this.layer1.map(() => 0));
      this.forceRouter2 = this.layer2.map(() => this.layer1.map(() => 0));
      this.targetWeights = new Float32Array(this.n1);
      this.currentWeights = new Float32Array(this.n1);
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
      this.applyNodeParams(this.layer1, p.layer1);
      this.applyNodeParams(this.layer2, p.layer2);
      if (p.tensionRouter) this.tensionRouter = p.tensionRouter;
      if (p.forceRouter2) this.forceRouter2 = p.forceRouter2;
    } else if (msg.type === 'excite') {
      for (let j = 0; j < this.n1 && j < msg.weights.length; j++) {
        this.targetWeights[j] = msg.weights[j];
      }
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
    } else if (msg.type === 'strike') {
      // one-shot energy bump for 'impulse' mode -- applied immediately
      // (handleMessage runs on the audio thread between process() calls),
      // not stored as a continuous target the way 'excite' weights are.
      for (let j = 0; j < this.n1 && j < msg.weights.length; j++) {
        this.layer1[j].energy += msg.weights[j] * STRIKE_ENERGY_SCALE;
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
      if (src.filterTaps !== undefined) n.filterTaps = Float32Array.from(src.filterTaps);
      if (src.filterMix !== undefined) n.filterMix = src.filterMix;
    }
  }

  applyFilter(node, x) {
    node.filterHistory[node.filterPos] = x;
    let filt = 0;
    for (let m = 0; m < FILTER_TAPS; m++) {
      const idx = (node.filterPos - m + FILTER_TAPS) % FILTER_TAPS;
      filt += node.filterTaps[m] * node.filterHistory[idx];
    }
    node.filterPos = (node.filterPos + 1) % FILTER_TAPS;
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

    const shapeAlpha = 1 / (2 * node.mass);
    const tensionEff = Math.pow(10, node.tensionExp + node.tensionMod);
    const omega = Math.sqrt(Math.abs(tensionEff - shapeAlpha * shapeAlpha));
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

  process(inputs, outputs, parameters) {
    const output = outputs[0];
    const channel = output[0];
    const masterGain = parameters.masterGain[0];
    const modDepth = parameters.modDepth[0];
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
        this.currentWeights[j] +=
          (this.targetWeights[j] - this.currentWeights[j]) * this.weightSmoothing;
        this.currentDampWeights[j] +=
          (this.targetDampWeights[j] - this.currentDampWeights[j]) * this.weightSmoothing;
        this.currentTensionModWeights[j] +=
          (this.targetTensionModWeights[j] - this.currentTensionModWeights[j]) * this.weightSmoothing;

        const node = this.layer1[j];

        // 'impulse' and 'pulse' modes drive energy directly (a one-shot bump
        // from a 'strike' message, or a periodic tick below) rather than
        // through a continuous force -- so `force` stays 0 for them here.
        let force = 0;
        if (this.excitationMode === 'sustained') {
          force = this.currentWeights[j] * node.excitationGain;
        } else if (this.excitationMode === 'noise') {
          force = this.currentWeights[j] * node.excitationGain * (Math.random() * 2 - 1);
        } else if (this.excitationMode === 'pulse' && pulseTick) {
          node.energy += this.currentWeights[j] * node.excitationGain * PULSE_ENERGY_SCALE;
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

        // which layer-1 node(s) are being targeted also determines how much
        // *force* (not just tension) each layer-2 node receives, via its own
        // routing row -- so different input nodes excite different parts of
        // layer 2, not just the same chord with a subtle tension tint.
        let routedForce = 0;
        const forceRow = this.forceRouter2[k];
        if (forceRow) {
          for (let j = 0; j < this.n1; j++) routedForce += forceRow[j] * this.currentWeights[j];
        }
        const force = routedForce * node.excitationGain;

        let routed = 0;
        const row = this.tensionRouter[k];
        if (row) {
          for (let j = 0; j < this.n1; j++) routed += row[j] * z1[j];
        }
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
}

registerProcessor('dho-processor', DHOProcessor);
