from experiments.e_2024_3_21.inference import model
import zounds
from time import sleep
import requests
from io import BytesIO
from soundfile import SoundFile
import numpy as np
import librosa
import torch
from http import HTTPStatus


proj = np.random.uniform(-1, 1, (16, 512))

filename = 'patternappindexerstate.txt'

def get_cursor() -> str:
    try:
        with open(filename, 'r') as f:
            value = f.read()
            return str(int(value))
    except IOError:
        return str(0)

def set_cursor(_id: int):
    with open(filename, 'w') as f:
        f.write(str(_id))


n_samples = 2 ** 15
samplerate = zounds.SR22050()

def ping_is_successful():
    try:
        resp = requests.get('http://localhost:8888')
        return resp.status_code == HTTPStatus.UNAUTHORIZED 
    except requests.exceptions.ConnectionError:
        return False
    

def get_new_presets():
    current_offset = get_cursor()
    
    print(f'Current offset is {current_offset}')
    
    resp = requests.get(
        'http://localhost:8888/presets', 
        headers={'authorization': str(1)},
        params={ 'offset': current_offset, 'limit': 10 }
    )
    resp.raise_for_status()
    data = resp.json()
    
    for item in data['items']:
        preset_id = item['id']
        media = item['media']
        hq = list(filter(lambda x: x['format'] == 'wav', media))[0]
        yield (preset_id, hq['cache_url'])

    next_offset = data['next_offset']
    
    if next_offset is not None:
        print(f'Setting cursor to {next_offset}')
        set_cursor(next_offset)


def index_preset(preset_id, url):
    
    print(f'attempting to index {preset_id}, {url}')
    
    resp = requests.get(url)
    resp.raise_for_status()
    io = BytesIO(resp.content)
    
    print(f'For preset {preset_id} and url {url} got ')
    with SoundFile(io, mode='r') as sf:
        sound_samplerate = sf.samplerate
        samples = sf.read()
        samples = librosa.resample(samples, orig_sr=sound_samplerate, target_sr=int(samplerate))
        
        for i in range(0, len(samples), n_samples):
            chunk = samples[i: i + n_samples]
            diff = n_samples - chunk.shape[-1]
            
            if diff > 0:
                # ensure that we have a proper-sized chunk
                chunk = np.pad(chunk, pad_width=[(0, diff)])
            
            
            chunk = torch.from_numpy(chunk).float().view(1, 1, n_samples)
            channels, encoding, schedules = model.iterative(chunk)
            print(f'Encoded chunk with shape {encoding.shape}')
            b, n_points, dim = encoding.shape
            
            embeddings = torch.sum(encoding, dim=1)
            embeddings = embeddings.data.cpu().numpy().reshape((-1,))
            embeddings = embeddings @ proj
            embeddings = embeddings.tolist()
            
            start_seconds = i / int(samplerate)
            duration_seconds = n_samples / int(samplerate)
            
            resp = requests.post(
                'http://localhost:8888/chunks', 
                headers={'authorization': str(1)},
                json={
                    'preset_id': preset_id,
                    'embedding': embeddings,
                    'start_seconds': start_seconds,
                    'duration_seconds': duration_seconds,
                    'version': 1
                }
            )
            resp.raise_for_status()
            print(f'Created indexed chunk for {preset_id}, {start_seconds}, {duration_seconds}')
    

def main():
    while True:
        sleep(5)
        
        if not ping_is_successful():
            print('waiting for server to be online')
            continue
        
        new_presets = list(get_new_presets())
        
        print(f'Got {len(new_presets)} new presets')
            
        for preset_id, url in new_presets:
            index_preset(preset_id, url)


if __name__ == '__main__':
    main()