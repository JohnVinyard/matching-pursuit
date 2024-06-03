from experiments.e_2024_3_31.inference import model
from experiments.e_2024_3_31.experiment import Model
import numpy as np
import json
import torch

from modules.normalization import max_norm
from util.playable import playable
from util.reporting import Section, create_audio_data_url, create_numpy_data_url, html_doc
import zounds
from sklearn.manifold import TSNE


if __name__ == '__main__':
    

    raw_params = model.get_parameters()
    raw_params = raw_params.data.cpu().numpy().astype(np.float32)
    
    # re-normalize for display
    normalized = raw_params - raw_params.min(axis=0, keepdims=True)
    normalized = normalized / (normalized.max(axis=0, keepdims=True) + 1e-8)
    numpy_data_url = create_numpy_data_url(normalized)
    
    tsne = TSNE(n_components=2, verbose=10)
    two_d = tsne.fit_transform(raw_params)
    
    registered, amps = model.forward(None, return_unpositioned_atoms=True)
    
    recon = torch.sum(registered, dim=1, keepdim=True)
    
    recon_element = create_audio_data_url(playable(
        max_norm(recon), 
        zounds.SR22050(), 
        normalize=True), 
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
                    border: solid 1px black;
                }}
                
                .audio-view-container {{
                    overflow-x: hidden;
                }}
            </style>
        ''',
        title='Splatting',
        citation_block='',
        sections=[
            Section(
                title='Reconstruction #1',
                anchor="reconstruction-1",
                content=f'''
                    <div class="outer-container">
                        <div class="recon-container">
                        
                            <div class="recon-panel">
                                <h2>Full Reconstruction (Sum of Segments)</h2>
                                <audio-view src="{recon_element}" height="50" scale="1" samples="256"></audio-view>  
                                <h2>Independent Atoms</h2>
                                {''.join([f'<audio-view src="{p}" height="15" scale="1" samples="256"></audio-view>' for p in ps])}
                            </div>
                            
                            <div class="recon-panel">
                                <h2>2D Projection of 14-Dimensional Atom Parameters</h2>
                                <scatter-plot 
                                    width="450" 
                                    height="450" 
                                    radius="0.075" 
                                    points='{json.dumps(point_data)}'
                                />
                            </div>
                            
                            <div class="recon-panel">
                                <h2>14-Dimensional Atom Parameters</h2>
                                <tensor-view src="{numpy_data_url}" type="2D" height="200" width="200" />        
                            </div>
                            
                        </div>
                    </div>
                '''
            )            
        ]
    )
    
    with open('splat-report.html', 'w') as f:
        f.write(html_content)