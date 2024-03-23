Time to resolve the problems that are leaving me _just_ short of some really gorgeous reconstructions:

# Primary Issues

- ~~noise/impulse model doesn't have enough variability (think record noise)~~ This _should_ be answered by the more complex noise model
- still not capturing vibrato for violins
- losing transients


# Today

## Losses
- ~~try pif loss~~
- ~~try patch info loss~~
- try re-introducing the discriminator

## Analysis
- try original U-Net

## Synthesis
- **try a return to original noise model** - This is a winner!
- ~~more resonance choices~~
- ~~higher frame rate for time-varying mix~~
- **nearest rather than learned for time-varying mix** - This is a winner!
- ~~try original resonance model~~
- ~~try conv upsampling with batch norm~~
- ~~try increasing context dim~~
- NERF-based time-varying mix
- ~~relu instead of softmax for time-varying mix~~
- sigmoid instead of softmax for time-varying mix

## HyperParameters
- learning rate?


# Ideas

- simply double density of resonances and double number of resonances chosen
- f0 + octave-based resonances
- NERF-based generator
- Oscillator bank + noise generator


# Things that might help
- denser resonances - This definitely helps!
- more resonances to fade between - This definitely helps!
- higher-res noise envelope - pending
- higher time resolution mixture - pending
- learnable resonances
- noise is its own resonance chain
- higher-frequency resolution loss
