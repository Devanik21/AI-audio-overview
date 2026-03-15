# AI Audio Overview

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/AI-audio-overview?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/AI-audio-overview?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> AI Audio Overview — digital signal processing for audio: filtering, feature extraction, and quality restoration.

---

**Topics:** `audio-ml` · `audio-to-text` · `content-summarization` · `deep-learning` · `generative-ai` · `large-language-models` · `multimodal-ai` · `nlp` · `whisper` · `podcast-summarization`

## Overview

AI Audio Overview applies digital signal processing techniques to audio signals, covering filter design, noise reduction, and spectral analysis. It provides both a programmatic API and an interactive interface for exploring audio processing parameters and observing their effects in real time.

The processing pipeline is built on SciPy's signal processing module and LibROSA, combining classical IIR/FIR filter theory with modern audio feature extraction. The project serves both as a practical audio processing tool and as an educational resource for understanding DSP concepts.

All processing steps are visualised with before/after waveform and spectrogram comparisons, and quantitative metrics (SNR, PESQ where applicable) provide objective quality measurement.

---

## Motivation

Audio quality issues — background noise, hum, clipping, reverb — affect recordings across speech, music, and scientific data. This project addresses these challenges with well-understood DSP techniques, providing transparent, configurable processing with measurable results.

---

## Architecture

```
Audio Input (WAV/MP3/FLAC)
        │
  Signal Analysis (STFT, spectrogram)
        │
  Filter/Enhancement Pipeline
        │
  Quality Measurement (SNR, spectral flatness)
        │
  Output Audio + Visualisation
```

---

## Features

### Filter Design
IIR and FIR filter design with configurable type, order, and cutoff frequency.

### Spectral Analysis
STFT, mel spectrogram, and MFCC feature extraction with visualisation.

### Noise Reduction
Multiple denoising methods: filtering, spectral subtraction, Wiener filtering.

### Quality Metrics
SNR before/after comparison, spectral flatness measurement.

### Waveform Visualisation
Before/after waveform and spectrogram comparison plots.

### Batch Processing
Process multiple audio files with the same configuration.

### Audio Playback Interface
In-app playback of original and processed audio for subjective evaluation.

### Export
Save processed audio in WAV, MP3, or FLAC format.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **SciPy** | Filter design and DSP | signal.iirdesign, sosfilt, STFT |
| **LibROSA** | Audio features | Mel spectrogram, MFCC, load/save |
| **NumPy** | Array ops | FFT, framing, overlap-add |
| **Matplotlib** | Visualisation | Waveform, spectrogram, Bode plots |
| **Soundfile** | Audio I/O | Read/write WAV/FLAC |

> **Key packages detected in this repo:** `streamlit` · `PyMuPDF` · `google-generativeai` · `gTTS` · `langdetect` · `googletrans` · `pandas` · `Pillow` · `wordcloud` · `matplotlib`

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JS projects)
- `pip` or `npm` package manager
- Relevant API keys (see Configuration section)

### Installation

```bash
git clone https://github.com/Devanik21/AI-audio-overview.git
cd AI-audio-overview
pip install scipy librosa numpy matplotlib soundfile
python process.py --input audio.wav
```

---

## Usage

```bash
python process.py --input audio.wav --method hpf --cutoff 100
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `--method` | `hpf` | Processing method |
| `--cutoff` | `100` | Filter cutoff frequency (Hz) |
| `--sr` | `16000` | Target sample rate |

> Copy `.env.example` to `.env` and populate all required values before running.

---

## Project Structure

```
AI-audio-overview/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Deep learning denoising model integration
- [ ] Real-time stream processing
- [ ] PESQ and STOI evaluation metrics
- [ ] Multi-channel audio support
- [ ] REST API for audio processing as a service

---

## Contributing

Contributions, issues, and feature requests are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please follow conventional commit messages and ensure any new code is documented.

---

## Notes

Sample rate defaults to 16kHz. Adjust for your audio files.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Crafted with curiosity, precision, and a belief that good software is worth building well.*
