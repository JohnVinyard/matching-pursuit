Being able to generate realistic audio from quantized spectrogram frames opens up
a lot of possibilities for longer sample generation, but so far GAN-based approaches
have failed me.

Try the dumbest possible thing, a raw-sample domain generator that is optimized based 
on raw-sample (or perceptual feature) MSE loss.