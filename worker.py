import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Generator, Union, Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By
import numpy as np
import os
import librosa
from time import sleep
import requests
import torch
from torch import nn

from modules.eventgenerators.overfitresonance import OverfitResonanceModel
from iterativedecomposition import Model as IterativeDecompositionModel


@dataclass
class SynthPreset:
    id: int
    synth_id: int
    parameters: dict
    created_by: int
    created_on: datetime


@dataclass
class PresetFeed:
    items: List[SynthPreset]
    next_offset: Union[None, int]


@dataclass
class CreateIndexRenderChunk:
    preset_id: int
    embedding: List[float]
    start_seconds: float
    duration_seconds: float
    version: int

n_samples = 2 ** 17
samples_per_event = 2048

# this is cut in half since we'll mask out the second half of encoder activations
n_events = (n_samples // samples_per_event) // 2
context_dim = 32

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

n_frames = n_samples // transform_step_size

proj = np.random.uniform(-1, 1, (context_dim, 8192))


def load_model() -> nn.Module:
    hidden_channels = 512
    wavetable_device = 'cpu'

    model = IterativeDecompositionModel(
        in_channels=1024,
        hidden_channels=hidden_channels,
        resonance_model=OverfitResonanceModel(
            n_noise_filters=64,
            noise_expressivity=4,
            noise_filter_samples=128,
            noise_deformations=32,
            instr_expressivity=4,
            n_events=1,
            n_resonances=4096,
            n_envelopes=64,
            n_decays=64,
            n_deformations=64,
            n_samples=n_samples,
            n_frames=n_frames,
            samplerate=samplerate,
            hidden_channels=hidden_channels,
            wavetable_device=wavetable_device,
            fine_positioning=True,
            fft_resonance=True
        ))

    with open('iterativedecomposition14.dat', 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    return model


def iter_chunks(samples: np.ndarray) -> Generator[Tuple[np.ndarray, float, float], None, None]:
    step_size = n_samples // 2

    duration_seconds = n_samples / samplerate

    for i in range(0, samples.shape[0], step_size):
        chunk = samples[i: i + step_size]

        if chunk.sum() == 0:
            # this chunk is silent, skip it
            continue

        if chunk.shape[-1] < n_samples:
            # pad to ensure that the chunk is the correct size for our
            # model
            diff = n_samples - chunk.shape[-1]
            chunk = np.concatenate([chunk, np.zeros(diff)], axis=0)

        start_seconds = i / samplerate
        yield chunk.astype(np.float32), start_seconds, duration_seconds

def project_event_vectors(vectors: torch.Tensor) -> np.ndarray:
    x = vectors.data.cpu().numpy().reshape((-1, context_dim))

    # compute graph edges
    x = x[:, None, :] - x[:, :, None]

    x = x.reshape((-1, context_dim))

    x = x @ proj
    indices = np.argsort(x, axis=-1)[:, -8:]

    sparse = np.zeros_like(x, dtype=np.bool8)
    np.put_along_axis(sparse, indices, values=np.ones_like(indices, dtype=np.bool8), axis=-1,)

    sparse = np.logical_or.reduce(sparse, axis=0)

    return sparse.astype(np.uint8)


class CochleaClient:
    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key

    def preset_feed(self, offset: int = 0, limit: int = 100) -> PresetFeed:
        resp = requests.get(
            f'{self.base_url}/presets',
            headers={'x-api-key': self.api_key},
            params={
                'offset': offset,
                'limit': limit
            }
        )
        data = resp.json()
        presets = [SynthPreset(**x) for x in data['items']]
        return PresetFeed(items=presets, next_offset=data['next_offset'])


    def push_index_chunk(self, indexed_chunk: CreateIndexRenderChunk) -> None:
        resp = requests.post(
            f'{self.base_url}/chunks',
            json=indexed_chunk.__dict__,
            headers={'x-api-key': self.api_key})


class StatefulClient:
    def __init__(self):
        super().__init__()

        with open('worker_config.json', 'r') as f:
            self.config = json.load(f)
            self.base_url = self.config['base_url']
            self.web_url = self.config['web_url']
            self.api_key = self.config['api_key']
            self.limit = self.config['limit']
            self.download_location = self.config['download_location']

        with open('worker_state.json', 'r') as f:
            self.state = json.load(f)

        self.client = CochleaClient(self.base_url, self.api_key)

    @property
    def offset(self):
        return self.state['offset']

    def _update_state(self, offset: Union[int, None]) -> None:

        if offset is None:
            print(f'skipping state update, offset is None')
            return

        print(f'updating state for offset {offset}')
        # update worker state in memory and on-disk
        self.state['offset'] = offset
        with open('worker_state.json', 'w') as f:
            json.dump(self.state, f)

    def _fetch_feed(self) -> Tuple[List[SynthPreset], int]:
        print(f'fetching feed at offset {self.offset}')
        feed = self.client.preset_feed(offset=self.state['offset'], limit=self.limit)
        offset = feed.next_offset
        return feed.items, offset

    def iter_presets(self) -> Generator[SynthPreset, None, None]:
        items, offset = self._fetch_feed()
        self._update_state(offset)

        while items:
            for item in items:
                yield item
            items, offset = self._fetch_feed()
            self._update_state(offset)

    def iter_preset_renders(self) -> Generator[Tuple[SynthPreset, np.ndarray], None, None]:
        for preset in self.iter_presets():
            web_url = f'{self.web_url}/presets/{preset.id}'
            samples = download_render(web_url, self.download_location)
            yield preset, samples

    def listen_for_preset_renders(self) -> Generator[Tuple[SynthPreset, np.ndarray], None, None]:
        while True:
            sleep(5)
            for preset, samples in self.iter_preset_renders():
                yield preset, samples

    def listen_and_index(self):
        model = load_model()
        for preset, samples in self.listen_for_preset_renders():
            for chunk, start_seconds, duration_seconds in iter_chunks(samples):
                chunk = torch.from_numpy(chunk).view(1, 1, n_samples)
                channels, vectors, schedules = model.iterative(chunk)
                projection = project_event_vectors(vectors)

                print(f'Computed index chunk preset {preset.id}, {start_seconds}, {duration_seconds}, {projection.tolist()}')
                index_chunk = CreateIndexRenderChunk(
                    preset_id=preset.id,
                    embedding=projection.tolist(),
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    version=1,
                )

                self.client.push_index_chunk(index_chunk)


def upload_pattern(api_host, api_key: str):
    # get a random musicnet segment
    # encode it
    # transform the encoding into a sequencer pattern
    # push to the API
    raise NotImplementedError()


def download_render(pattern_uri: str, download_path: str) -> np.ndarray:
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_path}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(pattern_uri)
        driver.implicitly_wait(5)
        render_button = driver.find_element(by=By.CSS_SELECTOR, value='.render-pattern')
        render_button.click()

        # TODO: This should be a wait condition in a loop
        driver.implicitly_wait(5)
        download_button = driver.find_element(by=By.CSS_SELECTOR, value='.download-pattern')
        download_button.click()

        sleep(5)

        # get all audio files in the tmp directory
        download_dir_files = os.listdir(download_path)
        print(download_dir_files)
        all_files = filter(lambda x: os.path.splitext(x)[1] == '.wav', download_dir_files)
        # sort from most to least recently modified
        by_download_date = list(sorted(
            all_files,
            key=lambda x: os.stat(os.path.join(download_path, x)).st_mtime,
            reverse=True))
        print(by_download_date)
        render_file = os.path.join(download_path, by_download_date[0])
        print('RENDER FILE', render_file)

        samples, sr = librosa.load(render_file, sr=22050, mono=True)

        os.remove(render_file)

        return samples

    except Exception as e:
        print(e)
    finally:
        driver.quit()


if __name__ == '__main__':
    client = StatefulClient()
    client.listen_and_index()
