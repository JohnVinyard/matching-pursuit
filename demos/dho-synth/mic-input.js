// Microphone-driven energy injection.
//
// Projects a live microphone's spectrogram down onto the energy plane's n1
// per-node excitation weights via a fixed bank of overlapping Hann-shaped
// frequency bands -- node 0 always picks up the lowest frequencies, node
// n1-1 the highest, with each node's band crossfading smoothly into its
// neighbors' (see frequencyBandProjectionMatrix below), rather than a fully
// random projection. The control plane (which bins are currently loud) is
// still effectively unpredictable moment to moment, but *where* on the
// energy plane low vs. high frequencies land is now fixed and consistent
// across randomizations, so it's something a user can actually learn over
// time. Lets a tap on a table, a vocal sound, etc. drive energy injection
// directly, while the damping and tension-mod planes keep working exactly
// as before through the existing 2D UI (unaffected -- this module only ever
// touches the energy plane's messages).
//
// A loaded ES module (for consistency with ar-mode.js's lazy-import
// pattern), though it only uses plain Web Audio APIs internally.

// Smaller than a typical spectral-analysis window, trading frequency
// resolution for responsiveness -- the window itself only costs
// FFT_SIZE / sampleRate (~11.6ms at 44.1kHz) of latency, and
// frequencyBandProjectionMatrix's bands scale with however many bins this
// leaves (frequencyBinCount = FFT_SIZE / 2), so they stay well-formed at
// any reasonable n1.
const FFT_SIZE = 512;

// AnalyserNode smooths each new FFT frame toward the previous one by this
// much by default (0.8), which is a far bigger latency cost than the window
// above -- at 0.8 and ~60fps polling, a transient takes ~10 frames
// (~170ms) to mostly settle. Lowered here for a much snappier response;
// still > 0 so the mic-driven excitation isn't jumping around on every
// single frame.
const SMOOTHING_TIME_CONSTANT = 0.2;

// Builds an (nOut x nIn) matrix of overlapping Hann-shaped frequency bands,
// ordered low to high: band i is centered at bin (i + 0.5) * nIn / nOut,
// with a half-width equal to the spacing between centers, so each band's
// raised-cosine falls to zero exactly at its neighbors' centers -- adjacent
// bands crossfade smoothly (like overlap-add windows) rather than leaving
// gaps or hard edges, and every bin still gets picked up by at least one
// band.
function frequencyBandProjectionMatrix(nOut, nIn) {
  const bandWidth = nIn / nOut;
  const rows = [];
  for (let i = 0; i < nOut; i++) {
    const center = (i + 0.5) * bandWidth;
    const row = new Float32Array(nIn);
    for (let j = 0; j < nIn; j++) {
      const dist = Math.abs(j - center);
      row[j] = dist < bandWidth ? 0.5 * (1 + Math.cos((Math.PI * dist) / bandWidth)) : 0;
    }
    rows.push(row);
  }
  return rows;
}

function dot(row, vec) {
  let total = 0;
  for (let i = 0; i < row.length; i++) total += row[i] * vec[i];
  return total;
}

let session = null;

export async function startMicInput() {
  if (session) return; // already running

  const bridge = window.DHOSynth;
  if (!bridge) {
    console.error('mic-input: window.DHOSynth is not available yet');
    return;
  }

  // guarantees audioCtx/workletNode exist regardless of whether "Start
  // audio" has already been clicked -- same convention the AR module
  // deliberately does *not* need (it never synthesizes, only drives
  // existing worklet messages, but those messages are no-ops without a
  // worklet to receive them)
  await bridge.ensureAudio();
  const audioCtx = bridge.getAudioContext();

  const stream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
  const source = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = FFT_SIZE;
  analyser.smoothingTimeConstant = SMOOTHING_TIME_CONSTANT;
  source.connect(analyser);
  // deliberately never connected onward to destination -- we only ever read
  // its frequency data, the mic is never actually played back

  const freqData = new Float32Array(analyser.frequencyBinCount);

  let n1 = bridge.state.n1;
  let matrix = frequencyBandProjectionMatrix(n1, freqData.length);

  let rafHandle = null;
  function frame() {
    rafHandle = requestAnimationFrame(frame);

    if (bridge.state.n1 !== n1) {
      n1 = bridge.state.n1;
      matrix = frequencyBandProjectionMatrix(n1, freqData.length);
    }

    analyser.getFloatFrequencyData(freqData); // dB, roughly [-100, -30]/quiet-to-loud by default
    for (let i = 0; i < freqData.length; i++) {
      freqData[i] = Math.max(0, freqData[i] + 100) / 100;
    }

    const projected = new Float32Array(n1);
    let mean = 0;
    for (let j = 0; j < n1; j++) {
      projected[j] = dot(matrix[j], freqData);
      mean += projected[j];
    }
    mean /= n1;

    // same "abs, subtract the frame's own mean, relu" idiom
    // ControlSignalCreator uses in parallel.py -- the frame's own mean
    // becomes its own noise floor (so a quiet/constant room tone
    // self-cancels) while whatever stands out from it -- a tap, a vocal
    // sound -- comes through, with no fixed calibration threshold needed.
    // The remaining scale from that rectified difference up to a weight
    // comparable to the pointer-driven planes' STRIKE_PEAK (1.6) is
    // state.micGain -- read live every frame, so micGainSlider (wired in
    // main.js's wireMicInput()) adjusts sensitivity in real time.
    const gain = bridge.state.micGain;
    const weights = new Array(n1);
    for (let j = 0; j < n1; j++) {
      weights[j] = Math.max(0, Math.abs(projected[j]) - Math.abs(mean)) * gain;
    }

    const workletNode = bridge.getWorkletNode();
    if (workletNode) workletNode.port.postMessage({ type: 'excite', weights });
  }

  frame();

  const energyPlaneEl = document.getElementById('energyPlane');
  energyPlaneEl.style.pointerEvents = 'none';
  energyPlaneEl.classList.add('mic-driven');
  bridge.planes.energyPlane.clearPointers();

  session = {
    stream,
    energyPlaneEl,
    cancel: () => cancelAnimationFrame(rafHandle),
  };
}

export function stopMicInput() {
  if (!session) return;

  session.cancel();
  session.stream.getTracks().forEach((t) => t.stop());
  session.energyPlaneEl.style.pointerEvents = '';
  session.energyPlaneEl.classList.remove('mic-driven');

  session = null;
}
