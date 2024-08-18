from typing import Any, Dict, Iterable, Literal, Tuple
import tokenize
import markdown
from io import BytesIO

ChunkType = Literal['CODE', 'MARKDOWN']

RenderTarget = Literal['html', 'markdown']


def build_template(page_title: str, content: str):
    template = f'''
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="X-UA-Compatible" content="ie=edge">
                <title>{page_title}</title>
                <script src="https://cdn.jsdelivr.net/gh/JohnVinyard/web-components@v0.0.11/build/components/bundle.js"></script>
                <style>
                    body {{
                        font-family: Arial;
                        margin: 20px 100px;
                    }}
                </style>
            </head>
            <body>
                {content}
            </body>
            </html>
    '''
    return template


class BytesContext:
    
    def __init__(self):
        super().__init__()
        self.bio = BytesIO()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


# with BytesContext() as f:
#     f.bio.write()

class ImageComponent:
    def __init__(self, src: str, height: int):
        super().__init__()
        self.src = src
        self.height = height
    
    def render(self, target: RenderTarget):
        if target == 'html':
            return self.html()
        elif target == 'markdown':
            return self.markdown()
        else:
            raise ValueError(f'Unknown render type "{target}"')
    
    def html(self):
        return f'''
        <img src="{self.src}" height="{self.height}"></img>
        '''
    
    def markdown(self):
        raise NotImplementedError('This component cannot be converted to markdown')
    

class CitationComponent:
    def __init__(self, tag: str, author: str, url: str, header: str, year: str):
        super().__init__()
        self.tag = tag
        self.author = author
        self.url = url
        self.header = header
        self.year = year
    
    def render(self, target: RenderTarget):
        if target == 'html':
            return self.html()
        elif target == 'markdown':
            return self.markdown()
        else:
            raise ValueError(f'Unknown render type "{target}"')
    
    def html(self):
        return f'''
        <citation-block
            tag="{self.tag}"
            author="{self.author}"
            url="{self.url}"
            header="{self.header}"
            year="{self.year}">
        </citation-block>
        '''
    
    def markdown(self):
        raise NotImplementedError('This component cannot be converted to markdown')

class AudioComponent:
    def __init__(self, src: str, height: int, scale: int = 1, controls: bool = True):
        super().__init__()
        self.src = src
        self.height = height
        self.scale = scale
        self.controls = controls
    
    def render(self, target: RenderTarget):
        if target == 'html':
            return self.html()
        elif target == 'markdown':
            return self.markdown()
        else:
            raise ValueError(f'Unknown render type "{target}"')
    
    def html(self):
        return f'''
        <audio-view
            src="{self.src}"
            height="{self.height}"
            scale="{self.scale}"
            {'controls' if self.controls else ''}
        ></audio-view>'''
    
    def markdown(self):
        raise NotImplementedError('This component cannot be converted to markdown')


def chunk_article(filepath: str, target: RenderTarget, **kwargs) -> Iterable[Tuple[str, int, int]]:
    with open(filepath, 'rb') as f:
        structure = tokenize.tokenize(f.readline)
        
        for item in structure:
            if item.type == tokenize.STRING and item.string.startswith('"""[markdown]'):
                no_quotes = item.string.replace('"""' , '')
                no_markdown = no_quotes.replace('[markdown]', '')
                markup = markdown.markdown(no_markdown)
                start, end = item.start[0], item.end[0]
                yield (markup, start, end)
            elif item.type == tokenize.COMMENT:
                content = item.string.replace('# ', '')
                try:
                    component = kwargs[content]
                    rendered = component.render(target)
                    start, end = item.start[0], item.end[0] + 1
                    yield (rendered, start, end)
                except KeyError:
                    continue
                
            
def classify_chunks(filepath: str, target: RenderTarget, **kwargs) -> Iterable[Tuple[ChunkType, str]]:
    with open(filepath, 'r') as f:
        lines = list(f.readlines())
    
    chunks = list(chunk_article(filepath, target, **kwargs))

    current_line = 0

    for markup, start, end in chunks:
        
        if start > current_line:
            yield ('CODE', '\n'.join(lines[current_line: start - 1]))
            current_line = start
        
        
        yield ('MARKDOWN', markup)
        current_line = end

    yield ('CODE', '\n'.join(lines[current_line:]))


def conjure_article(filepath: str, target: RenderTarget, **kwargs: Dict[str, Any]):
    
    final_chunks = classify_chunks(filepath, target, **kwargs)
    
    content = ''
    
    for i, ch in enumerate(final_chunks):
        t, new_content = ch
        
        new_content = new_content.strip()
        
        if t == 'CODE' and len(new_content):
            content += f'\n<code-block language="python">{new_content}</code-block>\n'
        elif t == 'MARKDOWN':
            content += f'\n{new_content}\n'
    
    with open('phaseinvariance.html', 'w') as f:
        f.write(build_template('Blah', content))
