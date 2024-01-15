- This model has some really good properties, and does "attend" to unusual events

Next Steps (smoothness helps):
    - try a more perceptually-aligned spectral representation
    - try adding orthogonal loss back in
    - try adding sparsity loss back in

Things to try:
    - run other experiment to 20K
    - try both models with resonance chain
    - try 2d transformer that projects from `channels -> hard_max(n_centroids) -> channels` to get a smoother(?) salience map
    - try patches over multi-band pooled transform
    - try nerf-like event generator
    - iterative matching pursuit model?

music has always been a way of "playing" our auditory system to evoke deep, evolutionary sense memories and emotions.