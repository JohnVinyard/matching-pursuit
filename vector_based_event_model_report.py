from typing import Iterable, Optional, Tuple
from data.audioiter import AudioIterator
from experiments.e_2024_3_21.inference import model
import zounds
import torch
from modules.normalization import max_norm
from uuid import uuid4
from util.reporting import Section, create_audio_data_url, html_doc
import conjure


collection = conjure.S3Collection(
    'conjure-test', 
    is_public=True, 
    cors_enabled=True)


n_samples = 2 ** 15
samplerate = zounds.SR22050()
total_examples = 4

n_events = 16
event_height = 50
timeline_height = n_events * event_height
year = 2024


class AudioTimeline(object):
    def __init__(self, channels: torch.Tensor, amps: torch.Tensor, logits: torch.Tensor, events: torch.Tensor):
        super().__init__()
        self.channels = channels
        self.amps = amps
        self.logits = logits
        self.events = events
        
        logits = logits.view(-1, 128)
        
        # find the index with the max value for each event
        indices = torch.argmax(logits, dim=-1)
        relative_positions = indices / 128
        self.relative_positions = relative_positions
        
        event_norms = torch.norm(channels, dim=-1)
        max_norm = torch.max(event_norms)
        self.max_norm = max_norm
    
    
    def to_html(self):
        return f'''
        <h4>Timeline</h4>
        <div class="timeline">
            {''.join([hidden_audio_element(
                self.channels[0, index], 
                self.events[0, index], 
                self.relative_positions[index], 
                self.max_norm,
                index) for index in range(self.channels.shape[1])])}
        </div>
        '''


# def spectrogram(audio: torch.Tensor, window_size: int = 512, step_size: int = 128):
#     audio = audio.view(1, 1, n_samples)
#     spec = stft(audio, window_size, step_size, pad=True)
#     n_coeffs = window_size // 2 + 1
#     spec = max_norm(spec.view(-1)).view(-1, n_coeffs)
#     spec = spec.data.cpu().numpy()
#     spec = np.rot90(spec)
    
#     img_data = np.zeros((spec.shape[0], spec.shape[1], 4), dtype=np.uint8)
    
#     # image opacity is determined by spectrogram intensity
#     img_data[:, :, 3:] = np.clip((spec[:, :, None] * 255).astype(np.uint8), 0, 255)
    
#     # image color is entirely black
#     img_data[:, :, :3] = 0
    
    
#     img = Image.fromarray(img_data, mode='RGBA')
#     scale = 4
#     x, y = img.size
#     img.thumbnail((x * scale, y * scale), Image.LANCZOS)
    
#     bio = BytesIO()
#     img.save(bio, format='png')
#     bio.seek(0)
    
#     data_url = create_data_url(bio.read(), 'image/png')
    
#     return f'''
#         <div class="spectrogram-image">
#             <img height="{n_coeffs}px" width="128px" src="{data_url}"></img>
#         </div>
#     '''
    
    


def svg_vector(
        vec: torch.Tensor, 
        pixel_height: int, 
        pixel_width: int, 
        max_norm: torch.Tensor,
        audio: torch.Tensor = None):
    
    vec = vec.view(-1)
    
    vec = vec - vec.min()
    vec = vec / (vec.max() + 1e-8)
    
    step = pixel_height / vec.shape[0]
    
    vec_norm = torch.norm(audio) if audio is not None else torch.norm(vec)
    opacity = (vec_norm / (max_norm + 1e-5)).item()
    
    return f'''
        <svg height="{pixel_height}" width="{pixel_width}">
            <g>
                {''.join([f'<rect x="0" y="{i * step}" width="{pixel_width}" height="{step}" fill-opacity="${opacity}" fill="rgb({int(vec[i].item() * 255)}, {int(vec[i].item() * 255)}, {int(vec[i].item() * 255)})"></rect>' for i in range(vec.shape[0])])}
            </g>
        </svg>
    '''


def hidden_audio_element(
        audio: torch.Tensor, 
        event_vector: torch.Tensor, 
        relative_position: float,
        max_norm: torch.Tensor,
        element_num: int):
    
    
    data_url = create_audio_data_url(audio, format='ogg', samplerate=samplerate)
    
    _id = uuid4().hex
    
    percentage_position = int(relative_position * 100)
    
    vertical_position = int((element_num / n_events) * 100)
    
    return f'''
        <div class="event" id="{_id}" style="left:{percentage_position}%; top:{vertical_position}%;">
            <audio-view src="{data_url}" scale="1" height="{event_height}" samples="512"></audio-view>
        </div>        
    '''


def audio_element(audio: torch.Tensor, title: str, subtitle: Optional[str] = None):
    
    data_url = create_audio_data_url(audio, format='ogg', samplerate=samplerate)
    
    return f'''
    <div class="audio-player">
        <h4>{title}</h4>
        {'' if subtitle is None else f'<p>{subtitle}</p>'}
        <audio-view src="{data_url}" scale="1" samples="512" height="100"></audio-view>
    </div>
    '''


def demo_example(
        orig: torch.Tensor, 
        recon: torch.Tensor, 
        random_events: torch.Tensor, 
        random_timings: torch.Tensor, 
        timeline: AudioTimeline,
        positioned: torch.Tensor):
    
    
    return f'''
    <section class="demo-example">
        <div>
            {audio_element(orig, 'Original')}
            {audio_element(recon, 'Recon')}
            {audio_element(random_events, 'With Random Event Vectors', '(based on mean and variance of event vectors for this sample)')}
            {audio_element(random_timings, 'With Random Timings')}
            <div>
                {timeline.to_html()}
            </div>
        </div>
    </section>
    '''

stream = AudioIterator(
    1,
    n_samples,
    samplerate=samplerate,
    normalize=True,
    overfit=False,
    step_size=1,
    pattern='*.wav')


@conjure.conjure(
    content_type='application/octet-stream', 
    storage=collection, 
    func_identifier=conjure.LiteralFunctionIdentifier('batchstats'),
    param_identifier=conjure.LiteralParamsIdentifier(b'stats'),
    serializer=conjure.PickleSerializer(),
    deserializer=conjure.PickleDeserializer(),
    prefer_cache=True,
    read_from_cache_hook=lambda x: print('READING FROM CACHE'))
def get_batch_statistics(batch_size=4):
    print('Computing batch statistics')
    
    stream = AudioIterator(
        batch_size,
        n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=False,
        step_size=1,
        pattern='*.wav')
    
    batch = next(iter(stream)).view(batch_size, 1, n_samples).to('cpu')
    
    final, embeddings, imp, scheduling, amps, mixed = model.forward(batch, return_context=True)
    
    embedding_means = torch.mean(embeddings, dim=(0, 1))
    embeddings_stds = torch.std(embeddings, dim=(0, 1))
    
    
    print(embedding_means.shape, embeddings_stds.shape)
    
    return (embedding_means, embeddings_stds)
    


def create_assets_for_single_item(
        audio: torch.Tensor, 
        event_stats: Tuple[torch.Tensor]):
    
    
    print('===================================')
    
    
    audio = audio.view(1, 1, n_samples)
    
    # full reconstruction
    final, embeddings, _, logits, amps, mixed = model.forward(audio, return_context=True)
    full_recon = torch.sum(final, dim=1, keepdim=True)
    full_recon = max_norm(full_recon)
    positioned = max_norm(final, dim=-1)

    timeline_container = AudioTimeline(mixed, amps, logits, embeddings)
    
    # with random events
    rnd, _, _, _, _, _ = model.forward(audio, return_context=True, random_events=event_stats)
    random_events = torch.sum(rnd, dim=1, keepdim=True)
    random_events = max_norm(random_events)
    
    # with random timings
    tm, _, _, _, _, _ = model.forward(audio, return_context=True, random_timings=True)
    random_timings = torch.sum(tm, dim=1, keepdim=True)
    random_timings = max_norm(random_timings)
    
    
    return audio, full_recon, random_events, random_timings, timeline_container, mixed, positioned


if __name__ == '__main__':
    
    with torch.no_grad():
        sections = []
        
        print('Generating Batch Statistics')
        event_stats = get_batch_statistics()
        
        
        for i, batch in enumerate(iter(stream)):
            batch = batch.to('cpu')
            print('==================================')
            print(f'Generating example {i}')
            
            orig, recon, rnd_events, rnd_timings, timeline, mixed, positioned = create_assets_for_single_item(
                batch, event_stats)
            
            section_html = demo_example(orig, recon, rnd_events, rnd_timings, timeline, positioned)
            sections.append(section_html)
            
            if (i + 1) >= total_examples:
                break
        
        doc = html_doc(
            styles=f'''
                    <style>
                
                    body {{
                        font-family: 'Sans Serif';
                    }}
                    
                    .demo-example {{
                        border: solid 1px #eee;
                        margin: 10px;
                        padding: 10px;
                        box-shadow: 10px 10px 5px 0px rgba(0,0,0,0.75);
                    }}
                    
                    .timeline {{
                        margin: 10px;
                        padding: 10px;
                        height: {timeline_height}px;
                        width: 100%;
                        border: solid 1px #eee;
                        position:relative;
                    }}
                    
                    .event {{
                        position: absolute;
                    }}
                    
                    pre, code {{
                        background-color: #eee;
                    }}
                    
                </style>
            ''',
            title='Sparse Interpretible Audio Model',
            citation_block=f'''
                <pre>
                    <code>
    @misc{{vinyard{year}audio,
        author = {{Vinyard, John}},
        title = {{Sparse Interpetable Audio}},
        url = {{https://JohnVinyard.github.io/machine-learning/2023/11/15/sparse-physical-model.html}},
        year = {year}
    }}
                    </code>
                </pre>
            ''',
            sections=[
                Section(
                    title='Architecture', 
                    anchor='architecture',
                    content=f'''
                        <p>
                        This small model attempts to decompose audio featuring acoustic instruments into the 
                        following components:
                        
                        <ul>
                            <li>Some maximum number of small (16-dimensional) event vectors, representing individual audio events</li>
                            <li><em>Times</em> at which each event occurs</li>
                        </ul>
                        
                        While event data are encoded as real-valued vectors and not discrete values, the 
                        representation learned still lend themselves to a sparse, interpretible, and hopefully easy-to-manipulate encoding.
                        
                        This first draft was trained using the amazing <a href="https://www.kaggle.com/datasets/imsparsh/musicnet-dataset">MusicNet dataset</a>.
                    </p>
                    <p>
                        Each sound sample below includes the following elements:
                        <ol>
                            <li>The original recording</li>
                            <li>The model's reconstruction</li>
                            <li>New audio using the original timings, but <em>random event vectors</em></li>
                            <li>New audio using the original event vectors, but with <em>random timings</em></li>
                        </ol>
                    </p>
                    <h3>Future Directions</h3>
                    <p>
                        There are several areas that could provide further gains in compression and interpretibility:
                        <ul>
                            <li>Imposing more severe sparsity constraints on the number of events produced.  You may notice that there are often many redundant events that could be merged into one.</li>
                            <li>Performing vector quantization on the event vectors such that there is a discrete set of possible events</li>
                        </ul>
                    </p>
                    <div>
                        <img src="https://matching-pursuit-repo-media.s3.amazonaws.com/vector_siam.drawio2.png" />
                    </div>
                    '''
                ),
                Section(
                    title='Sound Samples',
                    anchor='sound-samples',
                    content=''.join(sections))
            ]
        )
        with open('report.html', 'w') as f:
            f.write(doc)
        