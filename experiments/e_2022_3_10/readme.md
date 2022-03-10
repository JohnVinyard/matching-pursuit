While cool and a nice development, the complex-valued transform's instantaneous 
phase representation still isn't quite right, and feels rather finiicky (i.e., lots
of phase artifacts) when overfitting.

So far, the multi-resolution PIF feature is superior in this regard (i.e., a 
perceptually equivalent signal that might be quite different in l2 space).

That said, I've struggled to get the experiments using multiresolution PIF
off the ground, except for autoencoders, which tend to have a lot of annoying
artifacts.  

This experiment seeks to find out whether a simpler, single-resolution approach
is just as effective.