I really feel I need some sort of adversarial loss mixed in, but seem to be having a hell of a time avoiding mode collapse, etc.

Try a vanilla GAN with gradient clipping

Before adding in the std loss, this had some of the most interesting results yet, although the generator eventually blew up.