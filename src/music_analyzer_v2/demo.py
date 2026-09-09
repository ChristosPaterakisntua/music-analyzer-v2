import torch

from .audio_loader import AudioLoadError, load_audio
from .feature_extractor import extract_features
from .models import load_saved_model
from .utils import outputs_to_predictions, select_device


def demo() -> None:
    device = select_device()
    saved_model_path = input("Saved model path: ").strip().strip('"')
    model, scalar_mean, scalar_std = load_saved_model(saved_model_path, device)

    while True:
        print('Provide an audio path, type "exit" to exit')
        audio_path = input("Audio path: ").strip().strip('"')
        if audio_path == "exit":
            break

        try:
            audio_data = load_audio(audio_path)
        except AudioLoadError as error:
            print(f"Skipping file: {error}")
            continue

        try:
            features = extract_features(audio_data)
        except (AudioLoadError, ValueError, RuntimeError) as error:
            print(f"Skipping file: {error}")
            continue

        log_mel_segments = torch.from_numpy(features.log_mel_segments).unsqueeze(0)
        scalar_features = torch.from_numpy(features.scalar_features)

        if scalar_mean is not None and scalar_std is not None:
            scalar_features = (scalar_features - scalar_mean) / scalar_std

        segment_lengths = torch.tensor(
            [log_mel_segments.shape[1]],
            dtype=torch.long,
        )

        with torch.no_grad():
            outputs = model(
                log_mel_segments=log_mel_segments.to(device),
                segment_lengths=segment_lengths.to(device),
                scalar_features=scalar_features.unsqueeze(0).to(device),
            )

        genre_preds, feeling_preds = outputs_to_predictions(outputs)
        print(f"Genres: {', '.join(genre_preds)}")
        print(f"Feelings: {', '.join(feeling_preds)}")


if __name__ == "__main__":
    demo()
