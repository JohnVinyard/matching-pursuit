# Model variants

- matching pursuit (greedy, one atom at a time)
- sparse transposed conv, followed by GNN/transformer autoencoder
- model only produces atoms, they are positioned based on best fit
- model refines all-at-once sparse reprsentation in a loop (THIS IS NEW)
- model estimates room and n vectors that encapsulate time and event params
- sparse coding followed by clustering
- convolutional recurrence for instrument modelling
- events are modelled as `f(event_vec, pos, t)` and optimized to also emit 0 before the event has begun