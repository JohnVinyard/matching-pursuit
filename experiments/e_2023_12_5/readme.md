# Ideas

- 11-28 experiment with softmax scheduling (try softmax with regulariztaion to promote peaky distribution)
    - analyzer should have learned positional encodings

- approx convolutions with many long kernels


- one at a time neural matching pursuit
    - each step tries to maximize the reduction in norm
    - Q: **what is the regularizer?  how do we guarantee that atoms are simple/re-usable?**

- regularized sine layer (input to sine should be as small as possible)

- energy-preserving RNN-like model for physical-modelling synthesis


- pif inversion with sparse coding
    - Q: **How do we avoid making this just another frame-based representation?**

- alternating scheduling and atom-learning steps