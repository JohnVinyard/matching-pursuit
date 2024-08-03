Next up:

- try multi-band pif loss
- try exponential damping from here: `x(t) = amp * (np.e **((-friction / (2 * mass) * time)))`
    - can this support damping/friction changing over time?
    - can it support later injections of energy, other than the initial amplitude
- try NERF like `f(time, duration, env, z)`