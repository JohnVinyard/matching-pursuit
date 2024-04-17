from typing import Iterable, Literal, Optional, Union
from base64 import b64encode
import zounds
import numpy as np
import torch

from util.playable import playable

def create_data_url(b: bytes, content_type: str):
    return  f'data:{content_type};base64,{b64encode(b).decode()}'


AudioFormat = Literal['wav', 'ogg']

def create_audio_data_url(
    audio: Union[np.ndarray, torch.Tensor], 
    format: AudioFormat = 'ogg', 
    samplerate: Union[int, zounds.SampleRate] = 44100):
    
    audio: zounds.AudioSamples = playable(audio, samplerate)
    
    if format == 'wav':
        fmt, subtype, content_type = 'wav', 'PCM_16', 'audio/wav'
    elif format == 'ogg':
        fmt, subtype, content_type = 'ogg', 'vorbis', 'audio/ogg'
    else:
        raise ValueError(f'Unsupported format {format}')
    
    bio = audio.encode(fmt=fmt, subtype=subtype)
    return create_data_url(bio.read(), content_type)
    

class Section(object):
    def __init__(self, title: str, anchor: str, content: str):
        super().__init__()
        self.title = title
        self.anchor = anchor
        self.content = content
    
    def render_toc_entry(self):
        return f'<a href="#{self.anchor}">{self.title}</a>'
    
    def render(self):
        return f'''
            <a id="{self.anchor}"></a>
            <h3>{self.title}</h3>
            <section>
                {self.content}
            </section>
        '''

class TableOfContents(object):
    def __init__(self, sections: Iterable[Section] = []):
        super().__init__()
        self.sections = sections
    
    def render(self):
        return f'''
            <h3>Table of Contents</h3>
            <ul>
                {''.join([f'<li>{s.render_toc_entry()}</li>' for s in self.sections])}
            </ul>
        '''

def html_doc(
    styles: Optional[str] = '', 
    title: Optional[str] = '',
    citation_block: Optional[str] = '', 
    sections: Iterable[Section] = []):
    
    toc = TableOfContents(sections)
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <title>{title}</title>
        <script src="https://cdn.jsdelivr.net/gh/JohnVinyard/web-components@v0.0.1/build/components/audioview.js"></script>
        {styles}
    </head>
    <body>
        <header>
            <h2>
                {title}
            </h2>
            <p>
                {toc.render()}
            </p>
            <h4>Cite this Work</h4>
            {citation_block}
            
            {''.join([s.render() for s in sections])}
                
        </header>
        
        <footer></footer>
    </body>
    </html>
    '''