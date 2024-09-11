# Audio Processing and Transcription System

This project automates audio processing, transcription, speaker diarization, music removal, forced alignment, and punctuation restoration using several state-of-the-art models. It generates transcriptions with speaker labels and timestamps, which can be saved in text or SRT subtitle formats.

## Process Overview

### 1. **Imports and Initial Setup**
   - The script imports the necessary libraries for audio processing, neural network operations, and text manipulation.
   - It sets up logging and imports configurations from `config.json` and external sources (using YAML and other configuration managers).

### 2. **Model Initialization**
   - **WhisperX**: Used for speech recognition.
   - **NeMo**: Utilized for speaker diarization to distinguish between different speakers.
   - **Demucs**: Separates music from speech, improving transcription quality by isolating vocals.
   - **DeepMultilingualPunctuation**: Adds punctuation to the transcribed text.
   - **CTC Forced Aligner**: Aligns transcribed text with the original audio.

### 3. **Audio Processing**
   - **Music Removal**: Uses Demucs to remove background music and enhance transcription quality.
   - **Mono Conversion**: Converts audio to mono format for compatibility with most audio processing models.

### 4. **Transcription and Diarization**
   - **Whisper Model**: Transcribes the speech from the audio.
   - **NeMo Diarization**: Identifies and labels different speakers in the audio.

### 5. **Post-processing**
   - **Forced Alignment**: Aligns the transcribed text with the audio using a CTC model.
   - **Punctuation Restoration**: Adds punctuation marks to the transcription to improve readability.

### 6. **Speaker Mapping and Realignment**
   - **Speaker Mapping**: Assigns the spoken words to specific speakers based on timestamps.
   - **Realignment**: Adjusts speech segments to ensure sentences are correctly attributed to the corresponding speakers.

### 7. **Output**
   - **Transcript Saving**: Saves the transcript to a text file with speaker labels.
   - **SRT Generation**: Generates subtitles in SRT format, including timestamps and speaker labels.

### 8. **Cleanup**
   - The script cleans up temporary files and directories that were created during the audio processing to keep the working environment tidy.

## Configuration (`config.json`)

The application uses a `config.json` file to manage various settings, including paths and model parameters. Below is the structure of the configuration file:

```json
{
  "audio_path": "./data/audio-files",
  "transcript_save_path": "./data/audio-files/transcripts",
  "stemming": false,
  "suppress_numerals": true,
  "model_name": "medium.en",
  "batch_size": 8,
  "language": null,
  "device": "auto"
}
```
## Setup Instructions

### Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/your-username/Speaker-Diarization.git
cd Speaker-Diarization
```
###To build the Docker image, run the following command in the repository root:

```bash
docker build -t Speaker-Diarization
```
###Run the Docker Container with Custom config.json
####Run the Docker container while mounting your custom config.json file and the directory containing your audio files:

```bash
docker run -v /path/to/your/config.json:/app/config.json -v /path/to/audio:/app/data/audio-files Speaker-Diarization
```
Replace /path/to/your/config.json with the actual path to your config.json.
Replace /path/to/audio with the path to the directory containing your audio files.
