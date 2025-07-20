# from iterativedecomposition import Model, OverfitResonanceModel, n_frames, samplerate, n_samples
# import torch
#
# from util import playable
# from util.playable import listen_to_sound
#
# hidden_channels = 512
# device = 'cpu'
#
# resonance_model = OverfitResonanceModel(
#     n_noise_filters=32,
#     noise_expressivity=8,
#     noise_filter_samples=128,
#     noise_deformations=16,
#     instr_expressivity=8,
#     n_events=1,
#     n_resonances=4096,
#     n_envelopes=256,
#     n_decays=32,
#     n_deformations=32,
#     n_samples=n_samples,
#     n_frames=n_frames,
#     samplerate=samplerate,
#     hidden_channels=hidden_channels,
#     wavetable_device=device,
#     fine_positioning=True,
#     context_dim=16
# ).to(device)
#
# model = Model(
#     resonance_model=resonance_model,
#     in_channels=1024,
#     hidden_channels=hidden_channels).to(device)
#
# model.load_state_dict(torch.load(
#     'iterativedecomposition2.dat',
#     map_location=lambda storage, loc: storage))
#
# rnd = model.random_sequence(device=device)
# print(rnd.shape)
# rnd = playable(rnd, samplerate=22050, normalize=True)
# # listen_to_sound(rnd)
#
