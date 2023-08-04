There's a tension between using sparsity and "time compression" as the bottleneck

Sparsity seems more musically compelling and interpretible, but also leads to some problems.

- does the model really understand timing
- does the time coordinate of the event translate directly, especially when models have the
  entire segment as context


Time compression has its own issues:

If it's possible to compress the sparse representation into a single vector, aren't we
back in the frame-based realm?  Why bother with the sparse representation at all?

It's also interesting to think about the possibilities of encapsulating an event
in its entirety (time and other properties) such that everything can be expressed
as relationships between events.  Add change vector to event vector to arrive at 


a self-organizing map might be an intersting way to stick with sparse (single-atom)
representations while also being able to talk about _relationships_ between atoms.