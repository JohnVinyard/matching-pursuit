An adversarial loss helps with annoying "beating" artifacts.  Naive MSE loss matches large scale structure well but
includes lots of annoying fine-grained artifacts.  

Today's experiment will seek to combine the best of both worlds by using both a feature-matching loss and a
naive MSE loss.