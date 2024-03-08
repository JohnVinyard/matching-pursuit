Getting back to basics, can I factorize audio in one go (rather than with matching pursuit)

# Learnings

Choosing directly (with max()) from an audio signal or a latent representation is problematic,
because the same real-world/latent factor will over-contribute to the output, crowding out quieter 
real-wordl/latent factors.

Also, the larger/longer the atoms, the larger the dictionary would need to be to adequately cover
the space.


## Factorizer Network

If I abandon the idea of discrete codes for a moment, it's possible to imagine a network that
analyzes, then "explodes" via grouped convolutions, with an extra loss term that makes the sum
of the exploded elements equal to the input.  Then, selections can be made in the classical/naive way.

## Large "Virtual" Dictionary

Instead of discrete codes, we can imagine another way of addressing a very large "virtual" dictionary,
where a binary-like code is fed to a generating network that "materializes" the atom.

## Factorize Dictionary into Impulses, Resonances, Room Reverbs

This approach allows for a much larger dictionary, and seems a natural and interpretible way to factorize,
but it's unclear how to make this selection.  **Maybe a chosen latent could expand into these selections**.

## Combine Approximate Convolutions with A "Subset Selector"

Working with a very large dictionary, train a small network to select a small subset of candidate atoms
to be convolved with the signal


Maybe the best next approach is just to decompose the spectrogram directly, one step at a time.