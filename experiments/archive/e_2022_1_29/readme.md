One of the learnings from the sine optimization is that x**2 seems to be an ideal 
non-linearity for both amp and freq parameters.  This experiment will repeat the experiment 
from 1-27-2022, only using a different non-linearity.

# Conclusion
This ended with exploding gradients.  It might be more successful with more careful optimization
but seems brittle overall.