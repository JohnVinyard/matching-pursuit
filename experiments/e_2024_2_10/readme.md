Things to try:

- single channel loss C with new multi-band PIF loss
- new single channel loss B



# Desiderata

- events can stand alone, it does not take multiple codes to represent a single real-world event
- as few atoms/events are used as possible to explain an audio segment
- there are no redundant codes/events


 ## Matching Pursuit

 In matching pursuit, we start by performing some number of sparse coding steps
 with a fixed dictionary.  Then, for each unique atom used, we:

 - add all instances of that atom back to the _current_ residual
 - take the average of all locations (after giving them unit norm)
 - remove the new atom from the residual, arriving at the new _current_ residual



 ## Neural Matching Pursuit

 First, analyze audio with a neural network and output some number of sparse events.

 Using some heuristic to sort the events, which might be a combination of:

 - loudest
 - most highly-correlated with their location
 - _earliest_ in time

 We step through the channels/events batch-wise and do the following:

 - add the channel back to the _current_ residual
 - add the distance between the channel and the (channel + residual) to the loss
 - produce a new, temporary event that is some mid-point between actual channel and ideal (channel + residual)
 - remove this temporary atom from the residual to produce the new _current_ residual
