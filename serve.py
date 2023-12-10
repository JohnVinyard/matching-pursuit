from typing import IO, Tuple
import falcon
import gunicorn.app.base
from models import model
import sys
import multiprocessing
import torch
from random import choice, randint
import librosa
import soundfile as sf
from io import BytesIO
import numpy as np
import requests

from modules.normalization import max_norm


# https://github.com/pytorch/pytorch/issues/82843
torch.set_num_threads(1)

json_example = {
    'events': [
        (1, 1.2, 2048), 
        (1, 1.2, 2048), 
    ], 
    'context': [0] * 16
}

N_SAMPLES = 2**15
MUSIC_NET_FILENAMES = '''
1727.wav  1791.wav  2079.wav  2167.wav  2224.wav  2302.wav  2371.wav  2433.wav  2506.wav  2572.wav
1728.wav  1792.wav  2080.wav  2168.wav  2225.wav  2304.wav  2372.wav  2436.wav  2507.wav  2573.wav
1729.wav  1793.wav  2081.wav  2169.wav  2227.wav  2305.wav  2373.wav  2441.wav  2509.wav  2575.wav
1730.wav  1805.wav  2082.wav  2177.wav  2228.wav  2307.wav  2374.wav  2442.wav  2510.wav  2576.wav
1733.wav  1807.wav  2083.wav  2178.wav  2229.wav  2308.wav  2376.wav  2443.wav  2512.wav  2581.wav
1734.wav  1811.wav  2104.wav  2179.wav  2230.wav  2310.wav  2377.wav  2444.wav  2514.wav  2582.wav
1735.wav  1812.wav  2105.wav  2180.wav  2231.wav  2313.wav  2379.wav  2451.wav  2516.wav  2586.wav
1739.wav  1813.wav  2112.wav  2186.wav  2232.wav  2314.wav  2381.wav  2462.wav  2521.wav  2588.wav
1742.wav  1817.wav  2113.wav  2194.wav  2234.wav  2315.wav  2383.wav  2463.wav  2522.wav  2590.wav
1749.wav  1818.wav  2114.wav  2195.wav  2237.wav  2318.wav  2384.wav  2466.wav  2523.wav  2591.wav
1750.wav  1822.wav  2116.wav  2196.wav  2238.wav  2319.wav  2388.wav  2471.wav  2527.wav  2593.wav
1751.wav  1824.wav  2117.wav  2198.wav  2239.wav  2320.wav  2389.wav  2472.wav  2528.wav  2594.wav
1752.wav  1828.wav  2118.wav  2200.wav  2240.wav  2322.wav  2390.wav  2473.wav  2529.wav  2595.wav
1755.wav  1829.wav  2119.wav  2201.wav  2241.wav  2325.wav  2391.wav  2476.wav  2530.wav  2596.wav
1756.wav  1835.wav  2127.wav  2202.wav  2242.wav  2330.wav  2392.wav  2477.wav  2531.wav  2603.wav
1757.wav  1859.wav  2131.wav  2203.wav  2243.wav  2334.wav  2393.wav  2478.wav  2532.wav  2607.wav
1758.wav  1872.wav  2138.wav  2204.wav  2244.wav  2335.wav  2397.wav  2480.wav  2533.wav  2608.wav
1760.wav  1873.wav  2140.wav  2207.wav  2247.wav  2336.wav  2398.wav  2481.wav  2537.wav  2611.wav
1763.wav  1876.wav  2147.wav  2208.wav  2248.wav  2341.wav  2403.wav  2482.wav  2538.wav  2614.wav
1764.wav  1893.wav  2148.wav  2209.wav  2282.wav  2342.wav  2404.wav  2483.wav  2540.wav  2618.wav
1765.wav  1916.wav  2149.wav  2210.wav  2283.wav  2343.wav  2405.wav  2486.wav  2542.wav  2619.wav
1766.wav  1918.wav  2150.wav  2211.wav  2284.wav  2345.wav  2406.wav  2487.wav  2550.wav  2620.wav
1768.wav  1919.wav  2151.wav  2212.wav  2285.wav  2346.wav  2410.wav  2488.wav  2555.wav  2621.wav
1771.wav  1922.wav  2154.wav  2213.wav  2288.wav  2348.wav  2411.wav  2490.wav  2557.wav  2622.wav
1772.wav  1923.wav  2155.wav  2214.wav  2289.wav  2350.wav  2415.wav  2491.wav  2560.wav  2626.wav
1773.wav  1931.wav  2156.wav  2215.wav  2292.wav  2357.wav  2417.wav  2492.wav  2562.wav  2627.wav
1775.wav  1932.wav  2157.wav  2217.wav  2293.wav  2358.wav  2420.wav  2494.wav  2564.wav  2629.wav
1776.wav  1933.wav  2158.wav  2218.wav  2294.wav  2359.wav  2422.wav  2497.wav  2566.wav  2632.wav
1777.wav  2075.wav  2159.wav  2219.wav  2295.wav  2364.wav  2423.wav  2501.wav  2567.wav  2633.wav
1788.wav  2076.wav  2160.wav  2220.wav  2296.wav  2365.wav  2424.wav  2502.wav  2568.wav  2659.wav
1789.wav  2077.wav  2161.wav  2221.wav  2297.wav  2366.wav  2431.wav  2504.wav  2570.wav  2677.wav
1790.wav  2078.wav  2166.wav  2222.wav  2300.wav  2368.wav  2432.wav  2505.wav  2571.wav  2678.wav
'''.strip().split(' ')
MUSIC_NET_IDS = list(
    filter(lambda x: len(x) > 0, map(lambda x: x.split('.')[0], MUSIC_NET_FILENAMES)))


def json_params_to_dense(d: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform d representing JSON parameters into an events and context
    tensor, respectively
    """
    events = d['events']
    encoded = np.zeros((4096, 128), dtype=np.float32)
    for atom, time, amp in events:
        encoded[atom, time] = amp
    encoded = torch.from_numpy(encoded).view(1, 4096, 128)
    context = d['context']
    context = np.array(context).astype(np.float32)
    context = torch.from_numpy(context).view(1, 16)
    return encoded, context
    


def dense_to_json_params(t: torch.Tensor, context: torch.Tensor) -> dict:
    t = t.data.cpu().numpy().reshape((4096, 128))
    a, b = np.nonzero(t)
    
    events = [(int(x), int(y), float(t[x, y])) for x, y in zip(a, b)]
    
    context = context.data.cpu().numpy().reshape((16,))
    context = [float(x) for x in context]
    return {
        'events': events,
        'context': context
    }


def get_segment(musicnet_id = None, start_index=None) -> Tuple[torch.Tensor, IO]:
    """
    Choose from local list
    check if file is downloaded locally
    """
    
    musicnet_id = musicnet_id or choice(MUSIC_NET_IDS)
    
    try:
        audio, sr = librosa.load(f'{musicnet_id}.wav', sr=22050, mono=True)
        
        total_samples = audio.shape[-1]
        
        start_index = start_index or randint(0, total_samples - N_SAMPLES)
        end_index = start_index + N_SAMPLES
        
        segment = audio[start_index: end_index].astype(np.float32)
        segment /= (segment.max() + 1e-8)
        audio_tensor = torch.from_numpy(segment).view(1, 1, N_SAMPLES)
        
        bio = BytesIO()
        sf.write(bio, segment, samplerate=22050, format='wav')
        bio.seek(0)
        
        return audio_tensor, bio, start_index, end_index, total_samples
    except IOError:
        r = requests.get(f'https://music-net.s3.amazonaws.com/{musicnet_id}')
        bio = BytesIO()
        bio.write(r.content)
        bio.seek(0)
        
        # read the bytes and resample them
        audio, sr = librosa.load(bio, sr=22050, mono=True)
        
        # save the entire audio file for later
        sf.write(f'{musicnet_id}.wav', audio, samplerate=22050, format='wav')
        
        total_samples = audio.shape[-1]
        
        # choose a random segment
        start_index = start_index or randint(0, total_samples - N_SAMPLES)
        end_index = start_index + N_SAMPLES
        segment = audio[start_index: end_index].astype(np.float32)
        segment /= (segment.max() + 1e-8)
        audio_tensor = torch.from_numpy(segment).view(1, 1, N_SAMPLES)
        
        bio2 = BytesIO()
        sf.write(bio2, segment, samplerate=22050, format='wav')
        bio2.seek(0)
        
        return audio_tensor, bio2, start_index, end_index, total_samples

def tensor_to_audio(t: torch.Tensor) -> IO:
    arr = t.data.cpu().numpy().astype(np.float32)
    bio = BytesIO()
    sf.write(bio, arr, samplerate=22050, format='wav')
    bio.seek(0)
    return bio


def reconstruct_segment(musicnet_id, start):
    tensor, audio_io, start, end, total = get_segment(musicnet_id, start)
    x, _, _ = model.forward(tensor)
    x = torch.sum(x, dim=1)
    x = x.view(N_SAMPLES)
    bio = tensor_to_audio(x)
    return bio


class Reconstruct(object):
    
    def on_get(self, req: falcon.Request, res: falcon.Response, musicnet, start):
        bio = reconstruct_segment(musicnet, int(start))
        data = bio.read()
        res.set_header('content-type', 'audio/wav')
        res.status = falcon.HTTP_OK
        res.content_length = len(data)
        res.body = data


class Audio(object):
    def on_get(self, req: falcon.Request, res: falcon.Response, musicnet=None, start=None):
        
        _, bio, start, end, total = get_segment(
            musicnet_id=musicnet, 
            start_index=None if start is None else int(start))
        
        data = bio.read()
        
        res.set_header('content-type', 'audio/wav')
        res.set_header('content-range', f'samples {start}-{end}/{total}')
        
        res.status = falcon.HTTP_OK
        res.content_length = len(data)
        res.body = data


class Frontend(object):
    def on_get(self, req: falcon.Request, res: falcon.Response):
        with open('index.htm', 'r') as f:
            doc = f.read()
            
        res.status = falcon.HTTP_OK
        res.content_length = len(doc)
        res.body = doc
        res.set_header('content-type', 'text/html')


class Synth(object):
    
    def on_post(self, req: falcon.Request, res: falcon.Response):
        
        encoding = req.media
        encoded, context = json_params_to_dense(encoding)
        audio, _, _ = model.from_sparse(encoded, context)
        audio = torch.sum(audio, dim=1)
        audio = audio.data.cpu().numpy().reshape((N_SAMPLES,))
        bio = BytesIO()
        sf.write(bio, audio, samplerate=22050, format='wav')
        bio.seek(0)
        data = bio.read()
        res.status = falcon.HTTP_OK
        res.content_length = len(data)
        res.body = data
        res.set_header('content-type', 'audio/wav')


class Encode(object):
    
    def on_get(self, req: falcon.Request, res: falcon.Response, musicnet, start):
        tensor, audio_io, start, end, total = get_segment(musicnet, int(start))
        encoded, dense = model.derive_events_and_context(tensor.view(1, 1, N_SAMPLES))
        data = dense_to_json_params(encoded, dense)
        res.status = falcon.HTTP_OK
        res.media = data
        res.set_header('content-type', 'application/json')
        
        

class App(falcon.API):
    def __init__(self, port):
        super().__init__(middleware=[])
        
        # synthesize
        self.add_route('/synth', Synth())
        
        # encode
        self.add_route('/encode/{musicnet}/{start}', Encode())
        
        # reconstruct
        self.add_route('/reconstruct/{musicnet}/{start}', Reconstruct())
        
        # audio
        self.add_route('/audio/{musicnet}/{start}', Audio())
        self.add_route('/audio/{musicnet}', Audio())
        self.add_route('/audio/', Audio())
        
        
        # HTML/javascript app
        self.add_route('/app', Frontend())
        


class StandaloneApplication(gunicorn.app.base.BaseApplication):

    def __init__(self, app, **options):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def serve(
        port: int = 8888,
        n_workers: int = None,
        revive=True):

    app = App(port=port)

    def worker_int(worker):
        if not revive:
            print('Exit because of worker failure')
            sys.exit(1)

    # worker_count = (multiprocessing.cpu_count() * 2) + 1
    worker_count = 2

    def run():
        standalone = StandaloneApplication(
            app,
            bind=f'0.0.0.0:{port}',
            workers=n_workers or worker_count,
            worker_int=worker_int,
            timeout=60)
        standalone.run()

    p = multiprocessing.Process(target=run, args=())
    p.start()
    return p


p = serve(8888)