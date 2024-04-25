from collections import defaultdict, namedtuple
from typing import Callable, Dict, List, Optional, Tuple
import zounds
import torch
from util import device
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.matchingpursuit import build_scatter_segments, dictionary_learning_step, sparse_code
from modules.normalization import unit_norm

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
            full_size: int=8192,
            samplerate: zounds.SampleRate =zounds.SR22050(),
            local_contrast_norm: bool = False):

        super().__init__()
        
        # The length of the audio band being decomposed, in samples
        self.size = size
        
        # The number of atoms in this band's dictionary
        self.n_atoms = n_atoms
        
        # The length of each atom, in samples
        self.atom_size = atom_size
        
        self.slce = slce
        self.device = device

        self.full_size = full_size
        self.ratio = full_size // self.atom_size
        self.atom_full_size = int(self.atom_size * self.ratio)
        self.samplerate = samplerate
        self.local_contrast_norm = local_contrast_norm

        d = torch.zeros(
            n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
        self.d = unit_norm(d)

        self._embeddings = None
    
    @property
    def embedding_dim(self):
        return self.embeddings.shape[-1]

    def shape(self, batch_size):
        return (batch_size, 1, self.size)

    def to_embeddings(self, indices):
        return self.embeddings[indices]

    def to_indices(self, embeddings):
        diff = torch.cdist(self.embeddings, embeddings, p=2)
        indices = torch.argmin(diff, dim=0)
        return indices

    @property
    def embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        

        self._embeddings = torch.eye(self.n_atoms, device=device)
        # self._embeddings = torch.zeros(self.n_atoms, 40, device=device).uniform_(-1, 1)
        return self._embeddings
    
    
        # compute embeddings for each element in the dictionary
        """
        1. resample to canonical size/sampling rate
        1. compute spectrogram
        1. compute MFCCs
        1. compute chroma
        1. global pooling operation(s)
        """


        with torch.no_grad():
            n_bands = 128
            kernel_size = 256
            summary_bands = 16
            spec_win_size = n_bands // summary_bands

            samplerate = zounds.SR22050()
            band = zounds.FrequencyBand(20, samplerate.nyquist - 100)
            scale = zounds.MelScale(band, n_bands)
            chroma = zounds.ChromaScale(band)
            chroma_basis = chroma._basis(
                scale, zounds.OggVorbisWindowingFunc())
            chroma_basis = torch.from_numpy(
                chroma_basis).to(self.d.device).float()

            filters = zounds.learn.FilterBank(
                samplerate, kernel_size, scale, 0.1, normalize_filters=True).to(device)
            upsampled = fft_resample(
                self.d.view(self.n_atoms, 1, self.atom_size),
                self.atom_full_size,
                is_lowest_band=True)

            spec = filters.forward(upsampled, normalize=False)

            summary = F.avg_pool1d(spec.permute(
                0, 2, 1), spec_win_size, spec_win_size)
            summary = unit_norm(summary.permute(0, 2, 1), dim=1)
            summary = torch.mean(summary, dim=-1)

            n_frames = self.full_size // 256
            spec = F.avg_pool1d(spec, 512, 256, padding=256)[..., :n_frames]

            mfcc = torch.fft.rfft(spec, dim=1, norm='ortho')
            mfcc = torch.abs(mfcc)
            mfcc = unit_norm(mfcc[:, 1:13, :], dim=1)
            mfcc = torch.mean(mfcc, dim=-1)

            chroma = spec.permute(0, 2, 1) @ chroma_basis.T
            chroma = chroma.permute(0, 2, 1)
            chroma = unit_norm(chroma, dim=1)
            chroma = torch.mean(chroma, dim=-1)

            final = torch.cat([mfcc, chroma, summary], dim=-1)

            self._embeddings = final

            return self._embeddings

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
    
    def to_unit_time(self, sample_position: int) -> int:
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
            extract_atom_embedding=extract_embeddings,
            local_contrast_norm=self.local_contrast_norm)
        
        if extract_embeddings:
            # tuple of (encoding, residual)
            return encoding

        # instances is { atom_index: [(atom_index, batch, position, atom), ...] }
        instances, scatter = encoding

        # this is just flattening into [(atom_index, batch, position, atom)]
        # TODO: use the flatten=True keyword instead
        all_instances = []
        for k, v in instances.items():
            all_instances.extend(v)
        return all_instances, scatter, batch.shape

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

    @property
    def embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        
        # self._embeddings = torch.cat(
        #     [band.embeddings for band in self.bands.values()])

        self._embeddings = torch.eye(self.total_atoms).to(device)
        return self._embeddings

    @property
    def embedding_dim(self):
        return self.embeddings.shape[-1]

    def to_embeddings(self, indices):
        return self.embeddings[indices]

    def to_indices(self, embeddings):
        diff = torch.cdist(self.embeddings, embeddings, p=2)
        indices = torch.argmin(diff, dim=0)
        return indices

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
