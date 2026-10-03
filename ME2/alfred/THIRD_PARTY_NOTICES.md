# Runtime model provenance

Alfred includes local copies of the reviewed runtime model artifacts. Runtime
code never reads the sibling `ME2-v1` directory.

The two custom wake-word classifiers were created for this project:

- `assets/models/wake_word/hey_alfred.onnx`
- `assets/models/wake_word/im_batman.onnx`

The Mel frontend, embedding model, and Silero VAD were copied from the
openWakeWord model resources audited in `ME2-v1`. The streaming wake-word
feature extraction in `src/alfred/wake_word.py` is a focused adaptation of
openWakeWord's ONNX path. Its precise model SHA-256 hashes and sizes are
recorded in `assets/models/manifest.json`.

Upstream projects:

- openWakeWord: https://github.com/dscripka/openWakeWord
- Silero VAD: https://github.com/snakers4/silero-vad
- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- Moonshine: https://github.com/moonshine-ai/moonshine
- Piper voice models: https://huggingface.co/rhasspy/piper-voices
- Open-Meteo: https://open-meteo.com/
- Google Chat incoming webhooks: https://developers.google.com/workspace/chat/quickstart/webhooks

Alfred's weather action retrieves current conditions from Open-Meteo's public
forecast API. No Open-Meteo code or data is bundled with the application.

Alfred's MESSAGE action posts to a user-configured Google Chat incoming
webhook. The webhook credential is read from a private environment file and is
not stored in the repository.

Alfred packages the `en_GB-alan-medium` Piper VITS voice from the official
sherpa-onnx TTS model release. Its upstream `MODEL_CARD` is retained beside
the model. The archive, model, token, configuration, and phonemizer-data
checksums are recorded in `assets/models/manifest.json`.

Alfred's optional clarification installer downloads the quantized Moonshine v2
Tiny English model from the official sherpa-onnx model release. The model is
excluded from Git, and its upstream `LICENSE` plus installation hashes are
retained in the ignored local model directory. Moonshine's English-language
models are released under the MIT License.

Sherpa-ONNX is distributed under Apache License 2.0. The Piper voice
repository is identified upstream as MIT licensed. The packaged phonemizer
data originates from eSpeak NG; eSpeak NG is distributed under GPL-3.0-or-
later. These components remain unmodified third-party assets.

The corresponding openWakeWord, Silero VAD, sherpa-onnx, and eSpeak NG license
texts are included in `licenses/`.
