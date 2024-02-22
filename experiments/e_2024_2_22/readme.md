Add in a discriminator loss, even though I've been trying to avoid it.

Instead of lowering the norm at each step, I should focus on _removing_ information
from one signal and _increasing_ information in the reconstruction.


I have to find a way to incorporate frequency pos encodings into the patches.
The problem I ran into was that by adding pos encodings and then projecting back
down, I ran the risk of model collapse, i.e., if I do this step before patch-ifying,
then it's possible for the model to just output the same patch for everything.

My noise generator is insufficient for certain types of noise.  Ideally, a scratchy
record sound in the background would be a single event, but this may spend the entire
event budget.