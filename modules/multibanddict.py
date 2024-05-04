from collections import Counter, defaultdict, namedtuple
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import zounds
import torch
from modules.stft import stft
from util import device
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose, fft_resample
from modules.matchingpursuit import build_scatter_segments, dictionary_learning_step, sparse_code
from modules.normalization import unit_norm
from torch.nn import functional as F
from librosa.filters import mel, chroma
import librosa
from hashlib import sha256

LocalEventTuple = Tuple[int, int, int, torch.Tensor]
GlobalEventTuple = Tuple[int, int, float, float]

Shape = Tuple
BandEncodingPackage = Tuple[List[LocalEventTuple], Callable, Shape]

class BandSpec(object):
    def __init__(
            self,
            size: int,
            n_atoms: int,
            atom_size: int,
            slce: Optional[slice]=None,
            device=None,
            signal_samples: int = 0,
            samplerate: zounds.SampleRate =zounds.SR22050(),
            local_contrast_norm: bool = False,
            is_lowest_band: bool = False):

        super().__init__()
        
        self.is_lowest_band = is_lowest_band
        
        # The length of the audio being decomposed, in samples,
        # at the "native" samplerate
        self.signal_samples = signal_samples
        
        # The length of the audio band being decomposed, in samples
        self.size = size
        
        # The number of atoms in this band's dictionary
        self.n_atoms = n_atoms
        
        # The length of each atom, in samples
        self.atom_size = atom_size
        
        self.slce = slce
        self.device = device

        self.samplerate = samplerate
        self.local_contrast_norm = local_contrast_norm

        d = torch.zeros(
            n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
        self.d = unit_norm(d)

        self._embeddings = None
    
    def __hash__(self):
        h = sha256(self.d.data.cpu().numpy()).hexdigest()
        return hash(h)
        
    
    @property
    def n_samples_at_native_rate(self):
        """
        Return the length of the _atoms_, in samples, 
        at the native samplerate (specified by self.samplerate)
        """
        ratio = self.signal_samples // self.size
        return self.atom_size * ratio
    
    def resampled_atoms(self) -> torch.Tensor:
        desired_size = self.n_samples_at_native_rate
        
        return fft_resample(
            self.d.view(self.n_atoms, 1, self.atom_size), 
            desired_size, 
            self.is_lowest_band)
    
    # def atom_embeddings(self, window_size=2048) -> torch.Tensor:
        
    
    #     with torch.no_grad():
    #         n_samples = self.n_samples_at_native_rate
    #         rs = self.resampled_atoms().view(self.n_atoms, 1, n_samples)
            
    #         padding = window_size - n_samples
    #         if padding > 0:
    #             rs = F.pad(rs, (0, padding))
            
    #         rs = rs.data.cpu().numpy().reshape((self.n_atoms, -1))
            
    #         output_features = np.zeros((rs.shape[0], 35)) # ?
            
    #         for i, atom in enumerate(rs):
    #             # TODO: concatenate these.  They are in the shape
    #             # (n_features, n_frames)
            
    #             centroid = librosa.feature.spectral_centroid(atom).mean(axis=-1)
    #             flatness = librosa.feature.spectral_flatness(atom).mean(axis=-1)
    #             chroma = librosa.feature.chroma_stft(atom).mean(axis=-1)
    #             mfcc = librosa.feature.mfcc(atom).mean(axis=-1)
    #             zcr = librosa.feature.zero_crossing_rate(atom).mean(axis=-1)
    #             feature = np.concatenate([centroid, flatness, chroma, mfcc, zcr])
    #             output_features[i] = feature
            
    #         output_features = torch.from_numpy(output_features).float().to(self.d.device)
            
    #         return output_features
                
    
    # @property
    # def embedding_dim(self):
    #     return self.embeddings.shape[-1]
    
    def shape(self, batch_size):
        return (batch_size, 1, self.size)

    @property
    def filename(self):
        return f'band_{self.size}.dat'

    @property
    def scatter_func(self):
        return build_scatter_segments(self.size, self.atom_size)

    def get_atom(self, index: int, norm: float):
        return self.d[index] * norm

    def load(self):
        try:
            d = torch.load(self.filename)
            self.d = d
            print(f'loaded {self.filename}')
        except IOError:
            print(f'failed to load ${self.filename}')

    def store(self):
        torch.save(self.d, self.filename)

    def learn(self, batch, steps=16):
        d = dictionary_learning_step(
            batch, 
            self.d, 
            steps, 
            device=self.device, 
            approx=self.slce, 
            local_constrast_norm=self.local_contrast_norm)
        self.d = unit_norm(d)
        return d
    
    def to_global_atom_index(self, index: int, offset: int) -> int:
        return offset + index
    
    def to_local_atom_index(self, index: int, offset: int) -> int:
        return index - offset
    
    def to_unit_time(self, sample_position: int) -> float:
        return sample_position / self.size
    
    def to_sample_time(self, unit_time: float) -> int:
        return int(unit_time * self.size)
    
    def to_amplitude(self, scaled_atom: torch.Tensor) -> int:
        return torch.norm(scaled_atom)
    
    def to_global_tuple(self, event: LocalEventTuple, offset: int) -> GlobalEventTuple:
        """
        Take a local event tuple in the form:
            (atom_index, batch, sample_pos, atom)
        And transform it into:
            (global_atom_index, unit_time, amplitude)
        """
        atom_index, batch, sample_pos, atom = event
        
        return (
            self.to_global_atom_index(atom_index, offset), 
            batch, 
            self.to_unit_time(sample_pos), 
            self.to_amplitude(atom))
    
    def to_local_tuple(self, event: GlobalEventTuple, offset: int) -> LocalEventTuple:
        """
        Take a global event tuple in the form:
            (global_atom_index, unit_time, amplitude)
        And transform it into:
            (atom_index, batch, sample_pos, atom)
        """
        
        global_index, batch, unit_time, amplitude = event
        
        local_index = self.to_local_atom_index(global_index, offset)
        return (
            local_index,
            batch,
            self.to_sample_time(unit_time),
            self.get_atom(local_index, amplitude)
        )
    

    def encode(self, batch, steps=16, extract_embeddings=None) -> BandEncodingPackage:
        encoding = sparse_code(
            batch,
            self.d,
            steps,
            device=self.device,
            approx=self.slce,
            flatten=True,
            extract_atom_embedding=extract_embeddings,
            local_contrast_norm=self.local_contrast_norm)
        
        if extract_embeddings:
            # tuple of (encoding, residual)
            return encoding

        # instances is { atom_index: [(atom_index, batch, position, atom), ...] }
        instances, scatter = encoding

        # this is just flattening into [(atom_index, batch, position, atom)]
        # TODO: use the flatten=True keyword instead
        # all_instances = []
        # for k, v in instances.items():
        #     all_instances.extend(v)
        
        
        return instances, scatter, batch.shape

    def decode(self, shape, all_instances, scatter):
        return scatter(shape, all_instances)

    def recon(self, batch, steps=16):
        # instances, scatter = sparse_code(
        #     batch, self.d, steps, device=self.device, approx=self.slce)
        # all_instances = []
        # for k, v in instances.items():
        #     all_instances.extend(v)

        all_instances, scatter, shape = self.encode(batch, steps)

        # recon = scatter(batch.shape, all_instances)
        recon = self.decode(shape, all_instances, scatter)
        return recon, all_instances, scatter


class MultibandDictionaryLearning(object):
    def __init__(self, specs: List[BandSpec], n_samples: int):
        super().__init__()
        self.bands = {spec.size: spec for spec in specs}
        self.min_size = min(map(lambda spec: spec.size, specs))
        self.n_samples = n_samples
        
        n_atoms = set([spec.n_atoms for spec in specs])
        if len(n_atoms) > 1:
            raise ValueError('Only specs with equal atom counts is currently allowed')
        
        self.n_atoms = list(n_atoms)[0]

        self._embeddings = None
    
    
    def __hash__(self):
        return hash([hash(b) for b in self.bands.values()])
    
    def __len__(self):
        return len(self.bands)
    
    def event_count(self, iterations: int) -> int:
        return len(self) * iterations
    
    # @property
    # def total_atoms(self):
    #     pass
    
    def atom_embeddings(self):
        return torch.eye(self.total_atoms, device=device)
        # return torch.cat([band.atom_embeddings() for size, band in self.bands.items()], dim=0)
    
    
    def event_embeddings(self, batch_size: int, events: List[GlobalEventTuple], atom_embeddings) -> torch.Tensor:
        with torch.no_grad():
            n_elements = len(events) // batch_size
            
            
            # atom_embeddings = self.atom_embeddings()
            base_shape = atom_embeddings.shape[-1]
            
            ge = torch.zeros(batch_size, n_elements, base_shape, device=device)
            
            event_index_counter = defaultdict(Counter)
            
            for event in events:
                global_index, batch, unit_time, amplitude = event
                band_index, band = self.get_band_from_global_atom_index(global_index)
                
                counter = event_index_counter[batch]
                current_index = counter[band_index]
                counter[band_index] += 1
                
                ae = atom_embeddings[global_index]
                
                # ge[batch, current_index, 0] = unit_time
                # ge[batch, current_index, 1] = amplitude
                
                ge[batch, current_index, :] = ae * amplitude.view(1)
            
            return ge
    
    def get_atom(self, size, index, norm):
        return self.bands[size].get_atom(index, norm)

    def size_at_index(self, index):
        return list(self.bands.keys())[index]

    def index_of_size(self, band_size):
        sizes = list([b.size for b in self.bands.values()])
        index = sizes.index(band_size)
        return index
    
    def shape_dict(self, batch_size):
        return {size: band.shape(batch_size) for size, band in self.bands.items()}
    

    def index_of_dict_size(self, size):
        for i, d in enumerate(self.band_dicts.values()):
            if size == d.shape[-1]:
                return i
        
        raise IndexError(f'{size} not found in f{self.shape_dict(1)}')

    @property
    def total_atoms(self):
        return sum(v.n_atoms for v in self.bands.values())

    @property
    def band_dicts(self):
        return {size: band.d for size, band in self.bands.items()}

    @property
    def band_sizes(self):
        return list(self.bands.keys())

    def partial_decoding_dict(self, batch_size):
        return {
            size: (build_scatter_segments(
                size, self.bands[size].atom_size), (batch_size, 1, size))
            for size in self.bands.keys()
        }

    def store(self):
        for band in self.bands.values():
            band.store()

    def load(self):
        for band in self.bands.values():
            band.load()

    def learn(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)
        for size, band in bands.items():
            self.bands[size].learn(band, steps)

    def encode(self, batch, steps, extract_embeddings=None) -> Dict[int, BandEncodingPackage]:
        bands = fft_frequency_decompose(batch, self.min_size)
        return {
            size: band.encode(bands[size], steps, extract_embeddings) 
            for size, band in self.bands.items()
        }
    
    def get_band_from_global_atom_index(self, index: int) -> Tuple[int, BandSpec]:
        band_index = index // self.n_atoms
        return band_index, list(self.bands.values())[band_index]
    
    def flattened_event_tuples(self, encoding: Dict[int, BandEncodingPackage]) -> List[GlobalEventTuple]:
        output = []
        offset = 0
        
        for size, package in encoding.items():
            events, scatter, shape = package
            band = self.bands[size]
            for event in events:
                g = band.to_global_tuple(event, offset)
                output.append(g)
            offset += band.n_atoms
        
        return output
    
    def hierarchical_event_tuples(
            self, 
            encoding: List[GlobalEventTuple], 
            original: Dict[int, BandEncodingPackage]) -> Dict[int, BandEncodingPackage]:
        
        hierarchical = defaultdict(list)
        
        for event in encoding:
            global_index, batch, unit_time, amplitude = event
            index, band = self.get_band_from_global_atom_index(global_index)
            offset = index * self.n_atoms
            local_event = band.to_local_tuple(event, offset)
            hierarchical[band.size].append(local_event)
        
        final = dict()
        for size, events in hierarchical.items():
            _, scatter, shape = original[size]
            final[size] = (events, scatter, shape)
        
        return final
            

    def decode(self, d, shapes=None):
        output = {}
        for size, tup in d.items():
            if shapes is not None:
                all_instances = tup
                scatter = self.bands[size].scatter_func
                shape = shapes[size]
            else:
                all_instances, scatter, shape = tup
            recon = self.bands[size].decode(shape, all_instances, scatter)
            output[size] = recon
        recon = fft_frequency_recompose(output, self.n_samples)
        return recon

    def recon(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)

        recon_bands = {}
        events = {}
        scatter = {}
        for size, spec in self.bands.items():
            r, e, s = self.bands[size].recon(bands[size], steps)
            recon_bands[size] = r
            events[size] = e
            scatter[size] = s

        recon = fft_frequency_recompose(recon_bands, batch.shape[-1])
        return recon, events
