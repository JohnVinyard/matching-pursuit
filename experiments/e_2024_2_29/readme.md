Try the previous switch events experiment with more of a focus on sparsity

# Things to Try

- visualize attn signal
- try sparsification techniques
- instead of a single attn channel, produce sixteen channels, take the max from each, then apply relu and sum
- take a windowed view of the sparsified vectors.  Add a loss that pushes coincident events apart in vector space
- I hate the delicate balance of the sparsity penalty.  Is there a way to gate/rectify and maintain gradient flow?

- ~~visualize audio channels~~
- add sharpened input to analyzer
- add back the "all time means should be the same" loss
- try discriminator _without_ masking.  Maybe this is leading to redundancy in events?

# Observations
- The model _really_ wants to "stick" to certain "pixels"