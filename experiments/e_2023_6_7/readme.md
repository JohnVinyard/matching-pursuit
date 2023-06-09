Toy experiment to determine whether keypoint loss strategy is viable.

The known, generating process will be to choose N triplets of `(atom, amplitude, time)`

I'll try a few different generators:
    - one with a learned dictionary that positions atoms using fft shift
    - one with a fixed dictionary that positions atoms using fft shift
    - one that generates using dense samples

And a few different encoders:
    - one with a learned dictionary
    - one with a fixed dictionary
    - 

There will be several options for the generator:

    - learned dictionary or fixed dictionary from the generating process
    - raw audio fit
     


# TODO
- try the encoder with global pooling.  Is something still learned?
    - No, but that's OK.  Compression isn't really my goal here, rather, I just want to make sure
      that gradients are flowing

- loss with atom relationships
- try a dense generator
- try a dense generator built with learned dictionary and fft_shift positioning
- Can I parameterize atoms beyond just position, atom and amplitude?

# Conclusions

It _is_ possible to learn a model that generates atom encodings using the same dictionary from the loss function

It _is not_ possible to learn positions, amps and atom encodings with a dense model.

Therefore, the ideal of a sparse loss, matching-pursuit-based loss paired with a dense, physics-inspired generator
is not likely to work.

Instead, the loss representation and the generator representation need to be isomorphic, e.g., both must be based
on atoms derived from matching-pursuit, or more complex and differently-parameterized "events" that the loss can derive
and the generator can produce.