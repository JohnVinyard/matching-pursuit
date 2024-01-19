# Problems to Solve

1. the super-sparse representation is inherently limited.  How do we represent the same piano note with different durations?
2. updating a single channel at a time and avoiding compensating for other channels' noise are problems running counter to one another


# Physical Model

```python

from torch import Tensor

def time_step(impulse: Tensor, shape: Tensor, hidden: Tensor):
   """
   impulse: the energy input into the system.  This comes from a time-varying
            external source
   shape: a unit-norm vector describing the current state of the resonator.  
          It should never change the _amount_ of energy in the system, only
          the shape/distribution of that energy.  This comes from a time-varying
          external source
   hidden: The energy stored in the system, starts with all zeros
   """

   # this must have the same norm as the impulse
   embedded_impulse = embed(impulse)

   hidden = hidden + embedded_impulse

   # shape can change the distribution of energy, but not the 
   # amount
   w = get_weight(shape)
   b = get_bias(shape)
   hidden = (hidden * w) + b
   hidden = renorm(hidden)

   # this must have the same norm as hidden
   output = to_output(hidden)

   # again, this must have the same norm as the output
   embedded_output = embed(output)

   # the system loses this amount of energy
   hidden = hidden - embedded_output

   return output, hidden

```