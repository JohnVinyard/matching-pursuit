- pif + orthogonal loss has very promising reconstructions after 20k iterations, but somewhat blurry random generations
- patch-based info loss also has very promising reconstruciotns after 20K, and some decently-nice random generations

Next Steps:
- try pif loss + sparsity
    - initially _very_ noisy
- try pif loss + patch/info loss (with no orthoginal loss)
    - reconstructions are very blurry/noisy
- try pif loss + orthogonal loss + patch/info loss