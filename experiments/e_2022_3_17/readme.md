I think my ideal situation is to have:
- oscillators + noise model on the generator side
- complex transform with instantaneous frequency on the discriminator/analysis side

I haven't quite figured out how to make the latter transform differentiable all the way through, so in this experiment, I'm 
settling for the AIM/PIF feature instead.

This is a simple experiment to see if I can learn an autoencoder that uses adversarial loss