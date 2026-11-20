// Phone-as-drumstick motion input: a small, instrument-agnostic library.
//
// Treats the phone as a stick held by its bottom edge, with its top edge
// (device +Y) pointing away from the player. It reports two things:
//
//   - where the stick is pointing, as a heading/elevation pair and as a
//     normalized `position` in [0, 1] across a horizontal arc centered on
//     wherever the player last recentered (0 = left edge, 1 = right edge);
//   - strikes: a fast downward flick of the tip, reported as it *stops*
//     (the moment a real stick would land), with a 0..1 strength derived
//     from the flick's peak angular speed.
//
// Nothing here knows about any particular instrument -- consumers map
// `position` to whatever targets they have and decide what a strike does.
//
// Layers, each usable on its own:
//
//   MotionTracker   pure math, no browser APIs. Feed it orientation
//                   quaternions and gyro samples; it returns pointing frames
//                   and strike events. Unit-testable, and could equally be
//                   fed by WebXR poses or recorded data.
//   sensor sources  Generic Sensor API (AbsoluteOrientationSensor --
//                   accelerometer + gyroscope + magnetometer fusion -- or
//                   RelativeOrientationSensor, plus Gyroscope), falling back
//                   to deviceorientation(absolute)/devicemotion events.
//   MotionInput     an EventTarget wiring a source to a tracker:
//                     const motion = new MotionInput({ arcDegrees: 90 });
//                     motion.addEventListener('frame', (e) => e.detail.position);
//                     motion.addEventListener('strike', (e) => e.detail.strength);
//                     await motion.start();
//                     motion.recenter();
//
// Drift: heading is the only quantity that drifts. 'absolute' heading is
// anchored to magnetic north (no accumulated drift, but local magnetic
// interference can skew it); 'relative' uses only gyro + accelerometer
// (immune to magnets, drifts slowly). Elevation and strikes never drift --
// gravity anchors the former, and the latter are momentary.
// `autoRecenterSeconds` optionally pulls the center toward the player's
// average heading, as a slow correction for whichever drift remains.
//
// Coordinate conventions: device frame is the W3C one (X right, Y toward
// the top edge, Z out of the screen, in the device's natural orientation);
// world frame is east-north-up. Orientation quaternions are [x, y, z, w]
// rotating device coordinates into world coordinates.

const DEG = 180 / Math.PI;

export const DEFAULT_OPTIONS = Object.freeze({
  heading: 'absolute', // 'absolute' (magnetometer-anchored) | 'relative'
  arcDegrees: 90, // total width of the playing arc, left edge to right edge
  frequency: 60, // requested sensor rate (Hz); browsers may cap it
  autoRecenterSeconds: 0, // 0 = off; otherwise time constant of center drift-correction
  strikeThreshold: 6, // rad/s of downward tip speed that arms a strike
  strikeRelease: 0.5, // fire once speed falls below this fraction of the peak
  strikeFullScale: 25, // rad/s peak speed that maps to strength 1
  strikeRefractoryMs: 60, // minimum gap between strikes
  strikeDirection: 'down', // 'down' | 'any'
});

// ---------------------------------------------------------------------------
// math helpers

export function rotate(q, v) {
  // v' = q v q*, written out for a unit quaternion
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function wrapPi(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// W3C deviceorientation angles (degrees; intrinsic Z-X'-Y'') -> quaternion.
export function quaternionFromEuler(alphaDeg, betaDeg, gammaDeg) {
  const a = alphaDeg / DEG / 2;
  const b = betaDeg / DEG / 2;
  const g = gammaDeg / DEG / 2;
  const qz = [0, 0, Math.sin(a), Math.cos(a)];
  const qx = [Math.sin(b), 0, 0, Math.cos(b)];
  const qy = [0, Math.sin(g), 0, Math.cos(g)];
  return mul(mul(qz, qx), qy);
}

function mul(p, q) {
  const [px, py, pz, pw] = p;
  const [qx, qy, qz, qw] = q;
  return [
    pw * qx + px * qw + py * qz - pz * qy,
    pw * qy - px * qz + py * qw + pz * qx,
    pw * qz + px * qy - py * qx + pz * qw,
    pw * qw - px * qx - py * qy - pz * qz,
  ];
}

// ---------------------------------------------------------------------------
// MotionTracker: pure, sample-driven

export class MotionTracker {
  constructor(options = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.quaternion = null;
    this.center = null; // heading (rad) mapped to position 0.5
    this.heading = 0; // rad, clockwise from north (or from the relative frame's origin)
    this.elevation = 0; // rad, + = tip up
    this.lastT = null;
    this.speed = 0;
    this.strike = { armed: false, peak: 0, peakPosition: 0.5, lastFire: -Infinity };
  }

  configure(options) {
    Object.assign(this.options, options);
  }

  recenter() {
    this.center = this.quaternion ? this.heading : null;
  }

  get position() {
    if (this.center === null) return 0.5;
    const arc = this.options.arcDegrees / DEG;
    const offset = wrapPi(this.heading - this.center);
    return Math.min(1, Math.max(0, 0.5 + offset / arc));
  }

  // Orientation sample: unit quaternion [x, y, z, w], device -> world.
  // Returns the pointing frame.
  updateOrientation(quaternion, t) {
    this.quaternion = quaternion;
    const tip = rotate(quaternion, [0, 1, 0]);
    this.elevation = Math.asin(Math.max(-1, Math.min(1, tip[2])));
    // heading of the tip; when the stick points nearly straight up/down the
    // tip's horizontal projection vanishes, so use the screen normal's
    // projection instead (keeps heading stable through steep angles)
    const horiz = Math.hypot(tip[0], tip[1]);
    if (horiz > 0.2) {
      this.heading = Math.atan2(tip[0], tip[1]);
    } else {
      const back = rotate(quaternion, [0, 0, -Math.sign(tip[2]) || -1]);
      this.heading = Math.atan2(back[0], back[1]);
    }
    if (this.center === null) this.center = this.heading;

    const tau = this.options.autoRecenterSeconds;
    if (tau > 0 && this.lastT !== null) {
      const dt = Math.max(0, (t - this.lastT) / 1000);
      this.center += wrapPi(this.heading - this.center) * Math.min(1, dt / tau);
    }
    this.lastT = t;
    return this.frame(t);
  }

  frame(t) {
    return {
      t,
      position: this.position,
      heading: this.heading * DEG,
      elevation: this.elevation * DEG,
      center: this.center === null ? null : this.center * DEG,
    };
  }

  // Gyro sample: angular velocity [x, y, z] in rad/s, device frame.
  // Returns a strike event or null.
  updateRotationRate(rate, t) {
    if (!this.quaternion) return null;
    const o = this.options;
    // velocity of the stick's tip on the unit sphere, in world coordinates;
    // its vertical component is the downward speed (rad/s when level)
    const omega = rotate(this.quaternion, rate);
    const tip = rotate(this.quaternion, [0, 1, 0]);
    const v = cross(omega, tip);
    const speed = o.strikeDirection === 'down' ? Math.max(0, -v[2]) : Math.hypot(v[0], v[1], v[2]);
    this.speed = speed;

    const s = this.strike;
    if (!s.armed) {
      if (speed >= o.strikeThreshold && t - s.lastFire >= o.strikeRefractoryMs) {
        s.armed = true;
        s.peak = speed;
        s.peakPosition = this.position;
      }
      return null;
    }
    if (speed > s.peak) {
      s.peak = speed;
      s.peakPosition = this.position;
    }
    if (speed > s.peak * o.strikeRelease) return null;

    // the flick has stopped: fire at the position the stick was aimed at
    // during the swing (the tip can wobble sideways as it decelerates)
    s.armed = false;
    s.lastFire = t;
    return {
      t,
      position: s.peakPosition,
      peakRate: s.peak,
      strength: Math.min(1, s.peak / o.strikeFullScale),
    };
  }
}

// ---------------------------------------------------------------------------
// sensor sources: each calls onOrientation(q, t) and onRotationRate(w, t),
// and returns { stop() }. Throw (or reject) if unavailable.

async function requestSensorPermissions(names) {
  if (!navigator.permissions?.query) return;
  const results = await Promise.all(
    names.map((name) => navigator.permissions.query({ name }).catch(() => null)),
  );
  const denied = names.filter((_, i) => results[i]?.state === 'denied');
  if (denied.length) throw new Error(`sensor permission denied: ${denied.join(', ')}`);
}

function startSensor(sensor) {
  return new Promise((resolve, reject) => {
    const onError = (e) => {
      sensor.removeEventListener('reading', onReading);
      reject(e.error || new Error('sensor error'));
    };
    const onReading = () => {
      sensor.removeEventListener('error', onError);
      resolve();
    };
    sensor.addEventListener('error', onError, { once: true });
    sensor.addEventListener('reading', onReading, { once: true });
    sensor.start();
  });
}

export async function genericSensorSource({ heading, frequency }, onOrientation, onRotationRate) {
  const Orientation = heading === 'absolute' ? window.AbsoluteOrientationSensor : window.RelativeOrientationSensor;
  if (!Orientation || !window.Gyroscope) throw new Error('Generic Sensor API unavailable');
  await requestSensorPermissions(
    heading === 'absolute' ? ['accelerometer', 'gyroscope', 'magnetometer'] : ['accelerometer', 'gyroscope'],
  );
  const orientation = new Orientation({ frequency, referenceFrame: 'device' });
  const gyro = new Gyroscope({ frequency, referenceFrame: 'device' });
  orientation.addEventListener('reading', () => onOrientation(Array.from(orientation.quaternion), orientation.timestamp));
  gyro.addEventListener('reading', () => onRotationRate([gyro.x, gyro.y, gyro.z], gyro.timestamp));
  try {
    await Promise.race([
      Promise.all([startSensor(orientation), startSensor(gyro)]),
      new Promise((_, reject) => setTimeout(() => reject(new Error('no sensor readings')), 2000)),
    ]);
  } catch (err) {
    orientation.stop();
    gyro.stop();
    throw err;
  }
  return {
    name: `Generic Sensor (${heading === 'absolute' ? 'AbsoluteOrientationSensor' : 'RelativeOrientationSensor'})`,
    stop() {
      orientation.stop();
      gyro.stop();
    },
  };
}

export async function deviceEventSource({ heading }, onOrientation, onRotationRate) {
  if (typeof DeviceMotionEvent === 'undefined') throw new Error('device motion events unavailable');
  // iOS-style permission gate (a no-op elsewhere); must run inside a user gesture
  for (const Ev of [DeviceMotionEvent, window.DeviceOrientationEvent]) {
    if (Ev?.requestPermission && (await Ev.requestPermission()) !== 'granted') {
      throw new Error('motion permission denied');
    }
  }
  const orientationEvent = heading === 'absolute' && 'ondeviceorientationabsolute' in window
    ? 'deviceorientationabsolute'
    : 'deviceorientation';
  const onOrient = (e) => {
    if (e.alpha === null) return;
    onOrientation(quaternionFromEuler(e.alpha, e.beta, e.gamma), e.timeStamp);
  };
  const onMotion = (e) => {
    const r = e.rotationRate;
    if (!r || r.alpha === null) return;
    // rotationRate: alpha about Z, beta about X, gamma about Y, in deg/s
    onRotationRate([r.beta / DEG, r.gamma / DEG, r.alpha / DEG], e.timeStamp);
  };
  window.addEventListener(orientationEvent, onOrient);
  window.addEventListener('devicemotion', onMotion);
  return {
    name: `device events (${orientationEvent} + devicemotion)`,
    stop() {
      window.removeEventListener(orientationEvent, onOrient);
      window.removeEventListener('devicemotion', onMotion);
    },
  };
}

// ---------------------------------------------------------------------------
// MotionInput: source + tracker, as an EventTarget
//
// Events (detail):
//   'frame'  { t, position, heading, elevation, center } -- every orientation sample
//   'strike' { t, position, strength, peakRate }
// `tracker.speed` holds the latest downward tip speed (rad/s), handy for
// showing a meter while tuning strikeThreshold/strikeFullScale.

export class MotionInput extends EventTarget {
  constructor(options = {}) {
    super();
    this.tracker = new MotionTracker(options);
    this.source = null;
    this.sourceName = null;
    this.counts = { orientation: 0, rotationRate: 0 }; // samples received, for diagnostics
  }

  get options() {
    return this.tracker.options;
  }

  get running() {
    return this.source !== null;
  }

  configure(options) {
    const restart = this.running && 'heading' in options && options.heading !== this.options.heading;
    this.tracker.configure(options);
    if (restart) return this.start();
    return Promise.resolve();
  }

  recenter() {
    this.tracker.recenter();
  }

  // Tries the Generic Sensor API first, then device events. Call from a
  // user gesture (button click) so any permission prompt is allowed.
  async start() {
    this.stop();
    const onOrientation = (q, t) => {
      this.counts.orientation++;
      const frame = this.tracker.updateOrientation(q, t);
      this.dispatchEvent(new CustomEvent('frame', { detail: frame }));
    };
    const onRotationRate = (w, t) => {
      this.counts.rotationRate++;
      const strike = this.tracker.updateRotationRate(w, t);
      if (strike) this.dispatchEvent(new CustomEvent('strike', { detail: strike }));
    };
    const errors = [];
    for (const source of [genericSensorSource, deviceEventSource]) {
      try {
        this.source = await source(this.options, onOrientation, onRotationRate);
        this.sourceName = this.source.name;
        return this.sourceName;
      } catch (err) {
        errors.push(`${source.name}: ${err.message || err}`);
      }
    }
    throw new Error(`no motion sensors available (${errors.join('; ')})`);
  }

  stop() {
    this.source?.stop();
    this.source = null;
    this.sourceName = null;
  }
}
