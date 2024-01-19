In today's experiment, we'll try out an adversarial loss to rid ourselves of the beating.  The model has been refined to the point that
it overfits well and reliably, but has a lot of perceptually glaring artificats that aren't emphaiszed by the hand-designed loss function.

Early results indicate that the adversarial loss does a great job of getting rid of the beating, but
doesn't necessarily learn a function that adheres closesly to the original audio

Ideas to mitigate:

- dense judgements
- 2d encoder and/or decoder
- force disc to judge mismtached real samples to ensure that it isn't ignoring conditioning