"""[markdown]

# Resonance Inference

One path toward sparse representations of musical audio is to make assumptions about the
system or underlying process that produced the signal.  Despite the prevalence of synthesizers and electronic music,
much of the audio and music we hear is produced by objects in the physical world resonating.

While a model that makes _no_ assumptions would be more _general_, such a model might also
waste a lot of capacity learning about the physics of sound as well as the dynamics of a musical performance.

The event generator models I've used in my
[iterative decomposition experiments](https://blog.cochlea.xyz/v3blogpost.html) use cascading convolutions to model
the resonances of musical instruments (and the rooms in which they are played) and my recent work on
[playable state-space models](https://blog.cochlea.xyz/ssm.html) also explored the modelling of physical resonance
using a fully-linear network.

This mini-experiment takes these ideas a bit further, overfitting a network to a single audio segment of ~6 seconds.
The "prior" here is that the signal was produced by injecting energy into a low-dimensional "control plane", which is
connected to multiple layers of resonances, gains and non-linearities.  Ideally, this model might be able to capture
some of the nuances of physical resonance that the state-space model could not, and could lead to highly sparse "scores"
which can be seen when looking at the "Control Signal" section of each example.

## Statefulness

One thing that the fully event-based approach misses is that resonating instruments/objects are _stateful_; If I pluck
an already-vibrating string, I'll get a different sound than if I had plucked a string at rest.  This model's RNN-like
structure makes it at least _possible_ to capture those kinds of dynamics.  It is, of course, a tradeoff, and I am
thinking about ways to integrate this model as an event generator in the iterative decomposition framework.

## Object Deformations

Event generators in the [iterative decomposition models](https://blog.cochlea.xyz/v3blogpost.html) interpolate between
several different convolutions to mimic the deformation of resonating objects.  For example, I might pluck a string
and then bend it, causing the resonant properties to change.  This model does not yet incorporate deformations, but might
do so more efficiently than the long-convolution models, since we operate in a recursive, frame-by-frame fashion.


[Model and training code is here!](https://github.com/JohnVinyard/matching-pursuit/blob/main/freqdomain.py)

# Cite this Article

"""

# citation


"""[markdown]

# Examples

## Example 1

"""

"""[markdown]

### Original

"""

# example_1.original

"""[markdown]

### Reconstruction

"""

# example_1.reconstruction

"""[markdown]

### Control Signal

"""

# example_1.sparsity_text

# example_1.cp

"""[markdown]

### Random control signal
 
 We produce a random, sparse control plane, interfaced with the overfit resonance model.
 
"""

# example_1.random



"""[markdown]

## Example 2

"""

"""[markdown]

### Original

"""

# example_2.original

"""[markdown]

### Reconstruction

"""

# example_2.reconstruction

"""[markdown]

### Control Signal

"""

# example_2.sparsity_text

# example_2.cp

"""[markdown]

### Random control signal

 We produce a random, sparse control plane, interfaced with the overfit resonance model.

"""

# example_2.random

"""[markdown]

## Example 3

"""

"""[markdown]

### Original

"""

# example_3.original

"""[markdown]

### Reconstruction

"""

# example_3.reconstruction

"""[markdown]

### Control Signal

"""

# example_3.sparsity_text

# example_3.cp

"""[markdown]

### Random control signal

 We produce a random, sparse control plane, interfaced with the overfit resonance model.

"""

# example_3.random

"""[markdown]

## Example 4

"""

"""[markdown]

### Original

"""

# example_4.original

"""[markdown]

### Reconstruction

"""

# example_4.reconstruction

"""[markdown]

### Control Signal

"""

# example_4.sparsity_text

# example_4.cp

"""[markdown]

### Random control signal

 We produce a random, sparse control plane, interfaced with the overfit resonance model.

"""

# example_4.random



from typing import Dict

import torch
from torch.optim import Adam

from conjure import CompositeComponent, Logger, conjure_article, AudioComponent, S3Collection, ImageComponent, \
    TextComponent, CitationComponent
from data import get_one_audio_segment
from freqdomain import construct_experiment_model, reconstruction_loss, sparsity_loss
from modules.infoloss import CorrelationLoss
from util import device

remote_collection_name = 'freq-domain-resonance-demo'


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def reconstruction_section(logger: Logger, n_iterations: int) -> CompositeComponent:
    n_samples = 2 ** 17
    samplerate = 22050

    model = construct_experiment_model(n_samples)
    optim = Adam(model.parameters(), lr=1e-3)
    #loss_model = CorrelationLoss(n_elements=2048).to(device)

    target = get_one_audio_segment(n_samples, samplerate, device=device)
    target = target.view(1, 1, n_samples).to(device)

    for i in range(n_iterations):
        optim.zero_grad()
        recon, control_signal = model.forward()
        recon_loss = reconstruction_loss(recon, target)
        recon_loss = recon_loss #+ loss_model.forward(target, recon)
        loss = recon_loss + sparsity_loss(control_signal)
        loss.backward()
        print(i, loss.item())
        optim.step()

    with torch.no_grad():
        final_recon, final_control_signal = model.forward()
        rnd, _ = model.random()

    nonzero = model.nonzero_count
    sparsity = model.sparsity

    markdown_text = f'''
The "control signal" has {nonzero} non-zero elements and a sparsity of {(100 * sparsity):.2f}%.  
The control signal has a total of 
`{model.n_frames} x {model.control_plane_dim} = {model.n_frames * model.control_plane_dim} elements`
    '''

    _, orig = logger.log_sound('original', target)
    _, recon = logger.log_sound('recon', final_recon)
    _, random = logger.log_sound('random', rnd)
    _, cp = logger.log_matrix_with_cmap('cp', final_control_signal[0], cmap='hot')

    orig_audio_component = AudioComponent(orig.public_uri, height=200)
    recon_audio_component = AudioComponent(recon.public_uri, height=200)
    random_audio_component = AudioComponent(random.public_uri, height=200)
    control_plane_component = ImageComponent(cp.public_uri, height=200)
    text_component = TextComponent(markdown_text)


    composite = CompositeComponent(
        original=orig_audio_component,
        reconstruction=recon_audio_component,
        random=random_audio_component,
        cp=control_plane_component,
        sparsity_text=text_component,
    )
    return composite


def demo_page_dict() -> Dict[str, any]:
    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    n_iterations = 1000

    example_1 = reconstruction_section(logger, n_iterations=n_iterations)
    example_2 = reconstruction_section(logger, n_iterations=n_iterations)
    example_3 = reconstruction_section(logger, n_iterations=n_iterations)
    example_4 = reconstruction_section(logger, n_iterations=n_iterations)

    citation = CitationComponent(
        tag='johnvinyardresonanceinference2024',
        author='John Vinyard',
        url='https://blog.cochlea.xyz/resonance-inference.html',
        header='Resonance Inference',
        year='2024')

    return dict(
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        citation=citation
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title='Resonance Inference',
        **display)


if __name__ == '__main__':
    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)
    remote.destroy(prefix=b'')
    generate_demo_page()
