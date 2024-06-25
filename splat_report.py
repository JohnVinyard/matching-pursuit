from typing import Tuple
from experiments.e_2024_3_31.experiment import \
    Model, single_channel_loss_3, perceptual_loss, multiband_loss
import numpy as np
import json
import torch

from modules.normalization import max_norm
from train.optim import optimizer
from util.playable import playable
from util.reporting import Section, create_audio_data_url, create_numpy_data_url, html_doc
import zounds
from sklearn.manifold import TSNE
from conjure import LocalCollectionWithBackup, numpy_conjure, pickle_conjure
import requests
from librosa import load
from io import BytesIO
from util import device
from torch import nn

collection = LocalCollectionWithBackup(
    local_path='splatting', 
    remote_bucket='zounds-blog-media', 
    is_public=True, 
    local_backup=True, 
    cors_enabled=True)

@numpy_conjure(collection, read_hook=lambda x: f'Reading audio {x} from cache')
def get_audio_segment(
        url: str, 
        target_samplerate: int, 
        start_sample: int, 
        duration_samples: int):
    
    resp = requests.get(url)
    bio = BytesIO(resp.content)
    bio.seek(0)
    
    samples, _ = load(bio, sr=target_samplerate, mono=True)
    
    segment = samples[start_sample: start_sample + duration_samples]
    
    diff = duration_samples - segment.shape[0]
    if diff > 0:
        segment = np.pad(segment, [(0, diff)])
    
    return segment.astype(np.float32)
    

@pickle_conjure(collection, read_hook=lambda x: f'Reading splatting model from {x}')
def train_splatting_model(
        url: str, 
        target_samplerate: int, 
        start_sample: int, 
        duration_samples: int,
        learning_rate: float = 1e-3, 
        training_iterations: int = 6000):
    
    samples = get_audio_segment(
        url, target_samplerate, start_sample, duration_samples)
    print(f'Got audio segment from url {url}')
    
    target = torch.from_numpy(samples).to(device).view(1, 1, duration_samples)
    target = max_norm(target)
    
    model = Model(n_resonance_octaves=128).to(device)
    optim = optimizer(model, lr=learning_rate)
    
    learning_rates = torch.linspace(1e-2, 1e-4, steps=training_iterations)
    
    for i in range(training_iterations):
        optim.zero_grad()
        recon, amps = model.forward(None)
        mask = amps > 1e-6
        sparsity = torch.abs(amps * mask).sum() * 0.1
        loss = single_channel_loss_3(target, recon, sort_by_norm=True, coarse_loss=False) + sparsity
        # loss = multiband_loss(recon, target)
        loss.backward()
        optim.step()
        print(f'Iteration {i}: Loss: {loss.item()}')
        
        try:
            new_learning_rate = learning_rates[i]
            print(f'new learning rate is {new_learning_rate.item()}')
            for g in optim.param_groups:
                g['lr'] = new_learning_rate
        except IndexError:
            pass
    
    params = model.state_dict()
    return samples, params
    

def train_and_run_inference(
        url: str, 
        start_sample: int, 
        duration_samples: int = 2**15, 
        target_samplerate: int = 22050, 
        learning_rate: float = 1e-3, 
        training_iterations: int = 6000) -> Tuple[np.ndarray, nn.Module]:
    
    samples, state_dict = train_splatting_model(
        url, 
        target_samplerate, 
        start_sample, 
        duration_samples, 
        learning_rate, 
        training_iterations)
    
    model = Model(n_resonance_octaves=128).to(device)
    print(model.get_parameters().shape)
    model.load_state_dict(state_dict)
    
    return samples, model


def create_report_section(
        title: str,
        anchor: str,
        url: str, 
        start_sample: int, 
        duration_samples: int = 2**15, 
        target_samplerate: int = 22050, 
        learning_rate: float = 1e-3, 
        training_iterations: int = 6000) -> Section:
    
    orig, model = train_and_run_inference(
        url=url,
        start_sample=start_sample,
        duration_samples=duration_samples,
        target_samplerate=target_samplerate,
        learning_rate=learning_rate,
        training_iterations=training_iterations
    )
    
    orig = torch.from_numpy(orig).view(torch.float32)
    

    raw_params = model.get_parameters()
    raw_params = raw_params.data.cpu().numpy().astype(np.float32)
    
    n_atoms, embedding_dim = raw_params.shape
    
    # re-normalize for display
    normalized = raw_params - raw_params.min(axis=0, keepdims=True)
    normalized = normalized / (normalized.max(axis=0, keepdims=True) + 1e-8)
    numpy_data_url = create_numpy_data_url(normalized)
    
    tsne = TSNE(n_components=2, verbose=10)
    two_d = tsne.fit_transform(raw_params)
    
    registered, amps = model.forward(None, return_unpositioned_atoms=False)
    
    recon = torch.sum(registered, dim=1, keepdim=True)
    
    orig_element = create_audio_data_url(
        playable(
            max_norm(orig),
            zounds.SR22050(), 
            normalize=True
        ),
        format='mp3',
        samplerate=zounds.SR22050())
    
    recon_element = create_audio_data_url(
        playable(
            max_norm(recon), 
            zounds.SR22050(), 
            normalize=True
        ), 
        format='mp3', 
        samplerate=zounds.SR22050())
    
    segments = registered.view(-1, registered.shape[-1])
    norms = torch.norm(segments, dim=-1)
    sorted_indices = torch.argsort(norms, descending=True)
    
    ps = [create_audio_data_url(
        playable(
            registered[0, i, :], 
            zounds.SR22050(), 
            normalize=True,
            pad_with_silence=False), 
        format='mp3', 
        samplerate=zounds.SR22050()) for i in sorted_indices]
    
    
    point_data = [{
        'x': float(vec[0]),
        'y': float(vec[1]),
        'startSeconds': 0,
        'duration_seconds': 0,
        'url': create_audio_data_url(
            playable(
                registered[0, i, :], 
                zounds.SR22050(), 
                normalize=True,
                pad_with_silence=False), 
                format='mp3', 
                samplerate=zounds.SR22050()),
    } for i, vec in enumerate(two_d)]
    
    
    return Section(
        title=title,
        anchor=anchor,
        content=f'''
            <div class="outer-container">
                <div class="recon-container">
                
                    <div class="orig-panel">
                        <h2>Original</h2>
                        <audio-view src="{orig_element}" height="50" scale="1" samples="256"></audio-view>  
                    </div>
                    <div class="recon-panel">
                        <h2>Full Reconstruction (Sum of Segments) after {training_iterations} iterations</h2>
                        <audio-view src="{recon_element}" height="50" scale="1" samples="256"></audio-view>  
                        <h2>Independent Atoms</h2>
                        {''.join([f'<audio-view src="{p}" height="15" scale="1" samples="256"></audio-view>' for p in ps])}
                    </div>
                    
                    <div class="recon-panel">
                        <h2>2D Projection of {embedding_dim}-Dimensional Atom Parameters using T-SNE</h2>
                        <scatter-plot 
                            width="450" 
                            height="450" 
                            radius="0.075" 
                            points='{json.dumps(point_data)}'
                        />
                    </div>
                    
                    <div class="recon-panel">
                        <h2>{embedding_dim}-Dimensional Atom Parameters</h2>
                        <tensor-view src="{numpy_data_url}" type="2D" height="200" width="200" />        
                        <p>
                            Each atom consists of the following parameters:
                            <ul>
                                <li>Mean and variance for the gaussian gain envelope applied to the entire "atom"</li>
                                <li>A scalar, unit value for the position in time of the atom</li>
                                <li>A scalar, unit value representing the mix between the noise impulse and resonance</li>
                                <li>A scalar, unit decay value which is used gto produce a cumulative product, representing the decay of the resonance</li>
                                <li>A scalar, unit value that describes how we cross-fade from starting filter to ending filter</li>
                                <li>A scalar, unit value representing the fundamental frequency (f0) of the resonance</li>
                                <li>A scalar, unit value which represents the decay of the resonance</li>
                                <li>A scalar, unit value which represents the spacing between harmonics (multiples of f0)</li>
                                <li>Mean and variance in the frequency domain for the filter applied to the noise impulse</li>
                                <li>A scalar value representing the overall amplitude/gain of the atom</li>
                                <li>A scalar value representing the choice of reverb impulse responses</li>
                                <li>A scalar value representing the dry/wet mix between the atom and the reverb impulse response</li>
                            </ul>
                        </p>
                    </div>
                    
                </div>
                <hr />
            </div>
        '''
    )       

if __name__ == '__main__':
    
    training_iterations = 3000
    start_second = 17
    
    
    html_content = html_doc(
        styles=f'''
            <style>
                body {{
                    font-family: sans-serif;
                    padding: 0;
                    margin: 0;
                    height: 100%;
                }}
                
                .outer-container {{
                    display: flex;
                    flex-wrap: wrap;
                    flex-direction: column;
                    align-items: stretch;
                    align-content: center;
                    justify-content: center;
                    flex-grow: 1;
                }}
                
                .recon-container {{
                    display: flex;
                    flex-direction: row;
                    flex-wrap: wrap;
                    flex-grow: 1;
                    width: 100%;
                }}
                
                .recon-panel {{
                    flex-grow: 1;
                    width: 450px;
                    margin: 5px;
                    padding: 5px;
                    border: solid 1px #aaa;
                }}
                
                .audio-view-container {{
                    overflow-x: hidden;
                }}
            </style>
        ''',
        title='Gaussian/Gamma Audio Splatting',
        citation_block=f'''
<pre>
    <code>
        @misc{{vinyard2024audio,
            author = {{Vinyard, John}},
            title = {{Gaussian/Gamma Audio Splatting}},
            url = {{https://JohnVinyard.github.io/machine-learning/2024/6/24/gamma-audio-splat.html}},
            year = 2024
        }}
    </code>
</pre>
        ''',
        sections=[
            Section(
                title='Abstract',
                anchor='#abstract',
                content=f'''
                <p>
                    In this work, we apply a <a href="https://arxiv.org/abs/2308.04079">Gaussian Splatting</a>-like approach to audio to produce 
                    a lossy, sparse, interpretable, and manipulatable representation of audio.  We use a source-excitation model for each audio "atom" 
                    implemented by convolving a burst of band-limited noise with a variable-length "resonance", which is built using a number of
                    exponentially decaying harmonics, meant to mimic the resonance of physical objects.  Envelopes are built in both the time and
                    frequency domain using gamma and/or gaussian distributions.  Sixty-four atoms are randomly initialized and then fitted ({training_iterations} iterations) to a short segment of audio 
                    via a loss using multiple STFT resolutions.  A sparse solution, with few active atoms is encouraged by a second, weighted loss term.  Complete code for the experiment can be found 
                    on <a href="https://github.com/JohnVinyard/matching-pursuit/blob/main/experiments/e_2024_3_31/experiment.py">github</a>.  Trained segments come from
                    the <a href="https://www.kaggle.com/datasets/imsparsh/musicnet-dataset">MusicNet dataset</a>.
                </p>
                '''
            ),
            
            create_report_section(
                title='Reconstruction # 1',
                anchor='reconstruction-1',
                url='https://music-net.s3.amazonaws.com/1728',
                start_sample=2**15 * start_second,
                duration_samples=2**15,
                learning_rate=1e-3,
                training_iterations=training_iterations,
            ),
            
            create_report_section(
                title='Reconstruction # 2',
                anchor='reconstruction-2',
                url='https://music-net.s3.amazonaws.com/2379',
                start_sample=2**15 * start_second,
                duration_samples=2**15,
                learning_rate=1e-3,
                training_iterations=training_iterations,
            ),
            
            create_report_section(
                title='Reconstruction # 3',
                anchor='reconstruction-3',
                url='https://music-net.s3.amazonaws.com/2550',
                start_sample=2**15 * start_second,
                duration_samples=2**15,
                learning_rate=1e-3,
                training_iterations=training_iterations,
            ),
            
            create_report_section(
                title='Reconstruction # 4',
                anchor='reconstruction-4',
                url='https://music-net.s3.amazonaws.com/1790',
                start_sample=2**15 * start_second,
                duration_samples=2**15,
                learning_rate=1e-3,
                training_iterations=training_iterations,
            )
        ]
    )
    
    with open('splat-report.html', 'w') as f:
        f.write(html_content)