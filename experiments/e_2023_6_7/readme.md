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
     

