"""[markdown]

The key difference here is that we don't have discrete events, but a continuous stream of energy injection into the
system

# Title

Here is the intro

"""

"""[markdown]

## Example 1

"""

# example_1.original
# example_1.recon
# example_1.random
# example_1.cp



"""[markdown]

## Example 2

"""

# example_2.original
# example_2.recon
# example_2.random
# example_2.cp


from typing import Dict

import torch
from torch.optim import Adam

from data import get_one_audio_segment
from freqdomain import construct_experiment_model, transform, reconstruction_loss, sparsity_loss
from conjure import CompositeComponent, Logger, conjure_article, AudioComponent, S3Collection
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
    loss_model = CorrelationLoss(n_elements=2048).to(device)

    target = get_one_audio_segment(n_samples, samplerate, device=device)
    target = target.view(1, 1, n_samples).to(device)

    for i in range(n_iterations):
        optim.zero_grad()
        recon, control_signal = model.forward()
        recon_loss = reconstruction_loss(recon, target)
        recon_loss = recon_loss + loss_model.forward(target, recon)
        loss = recon_loss + sparsity_loss(control_signal)
        loss.backward()
        print(i, loss.item())
        optim.step()

    with torch.no_grad():
        final_recon, final_control_signal = model.forward()
        rnd, _ = model.random()

    _, orig = logger.log_sound('original', target)
    _, recon = logger.log_sound('recon', final_recon)
    _, random = logger.log_sound('random', rnd)
    _, cp = logger.log_matrix('cp', final_control_signal[0])

    orig_audio_component = AudioComponent(orig.public_uri, height=200)
    recon_audio_component = AudioComponent(recon.public_uri, height=200)
    random_audio_component = AudioComponent(random.public_uri, height=200)

    composite = CompositeComponent(
        original=orig_audio_component,
        reconstruction=recon_audio_component,
        random=random_audio_component,
        cp=cp
    )
    return composite


def demo_page_dict() -> Dict[str, any]:
    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    n_iterations = 100

    example_1 = reconstruction_section(logger, n_iterations=n_iterations)
    example_2 = reconstruction_section(logger, n_iterations=n_iterations)

    return dict(
        example_1=example_1,
        example_2=example_2,
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
