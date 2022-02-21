Based on 2-9 and yesterday's experiment, it seems that fairly dense oscillators are necessary.  
My current working theory is that the noise rate is becoming too high, causing the annoying "beating"
that's occurring starting at band 2048 and worsening from there.

Today's plan of attack will focus on the noise, to see if tweaks thereof can mitigate the beating.

Things I'll try:

- constant and low noise rate to avoid beating approaching 10-20hz
- only allow positive noise coefficients?
- try a noise model that doesn't use FFT


seems to be caused by crowding in higher bands where spacing becomes more logarithmic

- linear in first two bands and log thereafter

or, as linear seems to help with learning: linear bands with adversarial loss?