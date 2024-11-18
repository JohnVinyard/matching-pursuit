

def train(n_samples: int = 2 ** 16):
    target = get_one_audio_segment(n_samples=n_samples, device=device)


    collection = LmdbCollection(path='mp')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)

    orig_audio(target)

    with torch.no_grad():
        original_features = loss_model.forward(target)

    for i in count():
        optim.zero_grad()
        recon = model.forward(None)
        recon_audio(recon)
        recon_feature = loss_model(recon)
        loss = torch.abs(original_features - recon_feature).sum()
        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    train(n_samples=n_samples)
