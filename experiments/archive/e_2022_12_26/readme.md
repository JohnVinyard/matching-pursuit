Try out a middle ground between my physically-based experiments and older
oscillator + noise-based ideas

Also, a passing idea this morning:

- I've been thinking about using a matching pursuit-based approach
 to choose atom locations (best-fit) rather than the network explicitly
 providing it
- I've also struggled a lot with the sparse (mostly flat) gradients I deal
  with when positioning atoms.



Encoders: `[transformer, dilated_stack]`
Scheduling: `[fft_shift, fft_convolve, best_fit, prod_dist_func, none]`


# To Try
- normalized patches with embedded loudness to encoder
- normalized patches + norms for loss
