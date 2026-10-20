from pathlib import Path
import pandas as pd


# Getting the Parent Directory Path Object
PROJECT_ROOT=Path(__file__).resolve().parents[1]


# Below we get the list of path of all the wav files inside all the Actors
audio_files = sorted(
    (PROJECT_ROOT / "data/raw").glob("Actor_*/*.wav")
)

# Selecting and Trying on 1 wav file first
audio_file=audio_files[0]

# Extract filename
sample_id=audio_file.stem
file_path=audio_file.relative_to(PROJECT_ROOT).as_posix()


# Meta-Data File Columns
METADATA_COLUMNS = [
    "sample_id",
    "file_path",
    "actor_id",
    "gender",
    "emotion_code",
    "emotion_label",
    "intensity",
    "statement",
    "repetition",
]

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

records = []

for audio_file in audio_files:
    sample_id=audio_file.stem
    file_path=audio_file.relative_to(PROJECT_ROOT).as_posix()

    parts=sample_id.split("-")
    assert len(parts) == 7, f"Invalid filename: {sample_id}"

    (
        modality,
        vocal_channel,
        emotion_code,
        intensity,
        statement,
        repetition,
        actor,
    ) = parts

    actor_id=int(actor)
    gender="female" if actor_id % 2 == 0 else "male"

    record=[
        sample_id,
        file_path,
        actor_id,
        gender,
        int(emotion_code),
        EMOTION_MAP[emotion_code],
        int(intensity),
        int(statement),
        int(repetition)
    ]

    records.append(record)

metadata=pd.DataFrame(
    records,
    columns=METADATA_COLUMNS
)

print("Metadata shape:", metadata.shape)
print(metadata.head())

# Saving the Audio Extracted Meta-Data
# As checked every emotion has equal distribution except neutral (since neutral does not have strong intensity)
# print("\nEmotion distribution:")
# print(metadata["emotion_label"].value_counts())

# As checked every actor has 60 recordings
# print("\nRecordings per actor:")
# print(metadata["actor_id"].value_counts().sort_index())

output_path = PROJECT_ROOT / "data" / "metadata.csv"
metadata.to_csv(output_path, index=False)
print("Saved:", output_path)
