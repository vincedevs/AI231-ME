# Alfred — Project Primer

## 1. Project Context

Review `ME2.md` first.

This document describes the Machine Exercise requirements for this project. At a high level, we need to develop an **AI assistant that uses a Voice Command Model (VCM) trained from scratch**.

The project should be approached as a **complete product**, rather than focusing exclusively on the machine learning model. However, the **VCM is a core and essential component** of the project and should receive significant attention in terms of architecture, training, evaluation, and performance.

The AI assistant is named **Alfred**, after Batman’s trusted butler, **Alfred Pennyworth**.

---

## 2. System Overview

Alfred operates as a three-stage pipeline:

1. **Stage 1 — Wake Word**
2. **Stage 2 — Voice Command Model (VCM)**
3. **Stage 3 — Action**

The expected high-level flow is:

```text
Idle / Listening
      │
      ▼
Stage 1: Wake Word Detection
      │
      │ "Hey Alfred"
      ▼
Stage 2: Capture Speech + VCM Inference
      │
      │ Intent + Slots
      ▼
Stage 3: Execute Action
      │
      ▼
Return to Stage 1
```

An additional wake word, **“I’m Batman”**, acts as an easter egg and returns the application directly to Stage 1 after playing a sound.

---

# 3. Stage 1 — Wake Word

## Purpose

Stage 1 continuously listens for a wake word before activating the command-processing pipeline.

The primary wake word is:

> **“Hey Alfred”**

When this wake word is detected with sufficient confidence, Alfred proceeds to Stage 2.

## Wake Word Model

The wake word model has already been created using **openWakeWord**:

https://github.com/dscripka/openWakeWord/

There is **no need to train a new wake word model**.

The trained model has already been exported to **ONNX**. The remaining task is to implement the inference and audio-processing pipeline required to run the openWakeWord model correctly.

## Expected Behavior

1. Continuously listen for audio input.
2. Pass the required audio representation through the openWakeWord pipeline.
3. Evaluate the model's wake-word confidence score.
4. If **“Hey Alfred”** exceeds the configured confidence threshold:

   * Play `wake_word_detected.wav`.
   * Proceed to **Stage 2**.
5. Otherwise, remain in Stage 1 and continue listening.

The confidence threshold should be configurable rather than unnecessarily hard-coded throughout the application.

---

## Easter Egg — “I’m Batman”

A second wake-word model exists for:

> **“I’m Batman”**

When this phrase is detected:

1. Randomly select one of five easter egg sounds:

   * `easter_egg_1.wav`
   * `easter_egg_2.wav`
   * `easter_egg_3.wav`
   * `easter_egg_4.wav`
   * `easter_egg_5.wav`
2. Play the selected sound.
3. **Do not proceed to Stage 2.**
4. Return immediately to **Stage 1** and resume listening for **“Hey Alfred”**.

The intended flow is therefore:

```text
"I'm Batman"
      │
      ▼
Play random easter egg sound
      │
      ▼
Return to Stage 1
```

---

# 4. Stage 2 — Voice Command Model (VCM)

## Purpose

Stage 2 captures the user's spoken command and passes it through the **Voice Command Model (VCM)**.

The VCM is responsible for:

* **Intent classification**
* **Slot extraction**

The resulting intent and slots are then passed to Stage 3.

---

## Command Capture

When Stage 1 successfully detects **“Hey Alfred”**:

1. Play `listening_start.wav` to indicate that Alfred is ready for the user's command.
2. Start listening for human speech using **Silero VAD**.
3. Use Silero VAD to detect and capture the relevant speech segment.
4. Pass the captured speech to the VCM for inference.

Silero VAD is used so that the application does not simply record arbitrary background audio, silence, or noise.

### Timeout Behavior

If no valid human speech is detected within approximately **5 seconds**:

1. Stop listening.
2. Play `listening_end.wav`.
3. Return to **Stage 1**.

The timeout should preferably be configurable.

---

# 5. VCM Inference

The VCM performs two primary tasks:

```text
Audio Command
     │
     ▼
Voice Command Model
     │
     ├── Intent Classification
     │
     └── Slot Extraction
              │
              ▼
       Intent + Slots
```

For example, conceptually:

```text
User:
"Play Bohemian Rhapsody"

Intent:
PLAY_MUSIC

Slots:
{
    "song": "Bohemian Rhapsody"
}
```

The exact supported intents and slots must follow the project's predefined schema.

---

## VCM Deployment and ONNX Export

The final trained Voice Command Model should be exported as a **self-contained ONNX inference model**.

The exported ONNX model should include the deterministic feature-extraction operations required by the trained VCM wherever technically feasible. The Alfred application should **not require a separate Python feature-extraction pipeline** to convert captured speech into spectrograms, Mel features, MFCCs, or other model inputs before ONNX inference.

The intended deployment interface is:

```text
Captured Speech Waveform
        │
        ▼
┌───────────────────────────────────┐
│          VCM ONNX Model           │
│                                   │
│  Audio preprocessing              │
│          ↓                        │
│  Feature extraction               │
│          ↓                        │
│  Neural network backbone          │
│          ↓                        │
│  Intent + Slot prediction heads   │
└───────────────────────────────────┘
        │
        ▼
Intent + Slots + Confidence
```

Conceptually, application inference should be approximately:

```python
result = vcm.predict(audio_waveform)
```

rather than:

```python
features = extract_features(audio_waveform)
result = vcm.predict(features)
```

### Model Input

Prefer a simple waveform-level input contract such as:

```text
Input:
    audio waveform
    dtype: float32
    shape: [batch, samples]
    sample rate: fixed project sample rate
```

The exact tensor shape and sample rate should be determined during model development and documented as part of the exported model contract.

Audio acquisition should produce audio using the expected sample rate whenever practical. Resampling does not need to be embedded into the VCM ONNX model unless there is a clear technical reason to do so.

### Operations That Should Be Included

Where supported by the selected architecture and ONNX runtime, the exported model should include deterministic preprocessing used during training, such as:

* waveform normalization
* framing and windowing
* STFT
* magnitude or power-spectrum computation
* Mel filterbank transformation
* logarithmic compression
* MFCC computation, if MFCCs are used
* fixed feature normalization required by the model
* the VCM neural network backbone
* intent-classification heads
* slot-extraction heads

The exact frontend depends on the chosen VCM architecture.

Most importantly, **training and inference must use the same feature-extraction implementation** so that there is no training-serving skew.

### What Remains Outside the Model

The ONNX model is not responsible for the entire Alfred audio pipeline.

The following remain application responsibilities:

```text
Microphone capture
        ↓
Silero VAD / speech segmentation
        ↓
Captured command waveform
        ↓
VCM ONNX
```

Therefore:

* **Silero VAD remains outside the VCM.**
* Microphone and audio-device handling remain outside the VCM.
* Wake-word detection remains Stage 1.
* Application-level confidence policies may remain outside the ONNX model.
* Action execution remains Stage 3.

The ONNX artifact should encapsulate the **ML inference pipeline**, not the complete Alfred application.

### ONNX Export as a Research and Deployment Requirement

ONNX export should be considered during architecture development rather than treated only as a final conversion step.

A serious candidate architecture should ideally satisfy all of the following:

1. It can be trained from scratch according to the Machine Exercise requirements.
2. Its deterministic feature-extraction frontend can be represented reliably in ONNX.
3. The exported ONNX model produces predictions numerically consistent with the original training framework.
4. It can perform inference with acceptable latency on the target hardware.
5. It does not require a separate application-side feature-extraction implementation.

Perform an **early ONNX export smoke test** for candidate architectures before committing substantial compute to large-scale training.

At minimum, validate:

```text
PyTorch / TensorFlow inference
              ↓
        Same raw waveform
              ↓
        ONNX inference
              ↓
Compare outputs
```

The framework and ONNX outputs should be numerically equivalent within an appropriate tolerance.

This prevents selecting and extensively training an architecture that later proves difficult or unreliable to deploy.

### Final VCM Artifact

The preferred final artifact is:

```text
vcm.onnx
```

with the conceptual contract:

```text
Raw command waveform
        ↓
      vcm.onnx
        ↓
Intent + Slots + Confidence
```

The exact ONNX outputs may use tensors rather than application-level dictionaries. The Alfred application may perform lightweight decoding of these tensors into the canonical intent and slot representation.

However, **feature extraction should not need to be reimplemented separately by the Alfred application**.

---

## Confidence Handling

VCM predictions must satisfy a configurable confidence threshold before an action is executed.

There are three general outcomes.

### 1. Confident Prediction

If the model predicts a supported intent with sufficient confidence:

1. Inform the user of the command Alfred understood.
2. Proceed to **Stage 3**.
3. Execute the corresponding action.

Alfred should communicate what it is about to do **before performing the action**.

Conceptually:

```text
User:
"Play Bohemian Rhapsody"

Alfred:
"Playing Bohemian Rhapsody."

→ Execute action
```

---

### 2. Low-Confidence Prediction

If an intent is predicted but does **not** meet the required confidence threshold:

1. Do not execute the action.
2. Ask the user to clarify or repeat the command.
3. Return to **Stage 1**.

The purpose of this behavior is to avoid executing potentially incorrect actions when the model is uncertain.

---

### 3. Unknown / Unsupported Command

If the command does not correspond to any supported intent:

1. Inform the user that the command is unknown or unsupported.
2. Do not execute an action.
3. Return to **Stage 1**.

The implementation should distinguish, where possible, between:

```text
Known intent + low confidence
```

and:

```text
Unknown / unsupported intent
```

---

# 6. Audio and Playback Behavior

Audio playback must interact correctly with the wake-word pipeline.

In particular:

> **If Alfred is currently playing music and a wake word is detected, the music should pause so that the user's command can be heard clearly.**

A conceptual interaction is:

```text
Music Playing
     │
     │ "Hey Alfred"
     ▼
Pause Music
     │
     ▼
Play wake_word_detected.wav
     │
     ▼
Stage 2
     │
     ▼
Process Command
```

After the command has been processed, playback behavior can depend on the resulting action.

---

# 7. Stage 3 — Action

## Purpose

Stage 3 converts the VCM output into an actual application action.

Actions are implemented as functions, service calls, or API calls.

Each supported intent should map to an appropriate action handler.

Conceptually:

```text
Intent + Slots
      │
      ▼
Intent → Action Mapping
      │
      ▼
Function / API Call
```

For example:

```text
Intent:
PLAY_MUSIC

Slots:
{
    "song": "Bohemian Rhapsody"
}

Mapped Action:
play_music(song="Bohemian Rhapsody")
```

---

## Intent and Slot Mapping

Each intent corresponds to a particular function or API operation.

Slots are used as arguments to those functions.

Conceptually:

```python
intent = "SET_TIMER"

slots = {
    "duration": 300
}

set_timer(duration=300)
```

If an optional slot is unavailable, incomplete, or unsupported, the corresponding action may use an appropriate **default argument**, as defined by the project's intent/action schema.

Do not invent intent mappings or slot behavior when an authoritative mapping already exists in the project files.

---

# 8. Voice Command Model

The **Voice Command Model (VCM)** is the primary machine learning component of Alfred.

Although Alfred should be developed as a complete product, the VCM should receive particular attention because its quality directly determines how reliably the assistant understands commands.

The goal is therefore not merely to make the model function, but to achieve **strong and measurable performance** for both:

* Intent classification
* Slot extraction

The VCM itself must be **trained from scratch** according to the requirements defined in `ME2.md`.

---

# 9. Training Datasets

The following datasets should be considered for training the VCM:

### Primary Datasets

1. **Fluent Speech Commands**
2. **Smart Speaker Command Dataset — University of Rochester**
3. **SLURP**
4. **SNIPS**
5. **STOP**
6. **Timers and Such**

Because these datasets may use different taxonomies, labels, slot representations, audio characteristics, and dataset structures, they should be normalized into the project's agreed-upon intent/slot schema before training.

---

## Supplemental Data

The following sources may be used to improve robustness and dataset coverage:

### ESC-50

Use **ESC-50** as a source of environmental/background noise.

Potential uses include:

* Noise augmentation
* Robustness testing
* Mixing environmental noise with command audio

### Kokoro TTS

Use **Kokoro TTS** to generate additional synthetic command samples when useful.

Potential uses include:

* Increasing samples for underrepresented intents
* Increasing linguistic variation
* Increasing speaker/acoustic diversity
* Generating additional examples for slot values

Synthetic data should supplement rather than blindly replace real speech data.

---

# 10. Candidate VCM Architectures

The following model architectures should be considered and evaluated:

1. **DS-CNN**
2. **TC-ResNet8 / TC-ResNet14**
3. **BC-ResNet**
4. **CRNN with Attention**
5. **Tiny Conformer**

The final architecture should not be selected solely because it is the most complex model.

Consider factors such as:

* Intent-classification performance
* Slot-extraction performance
* Generalization
* Robustness to noise
* Inference latency
* Model size
* CPU requirements
* Memory consumption
* Suitability for real-time inference
* Training complexity
* Deployment complexity
* ONNX export compatibility
* Ability to embed deterministic feature extraction into the exported ONNX model

Since Alfred is an interactive voice assistant, **real-time usability and inference efficiency are important alongside predictive performance**. ONNX deployability should be validated early rather than only after final model selection.

---

# 11. Authoritative Intent and Slot Schema

Refer to:

> **`Dataset Schema.xlsx`**

This file contains the agreed-upon schema for the VCM.

It should be treated as the authoritative reference for:

* Supported intents
* Supported slots
* Intent names
* Slot names
* Dataset normalization
* Label mapping

When importing external datasets, map their original labels into this schema rather than creating arbitrary new intent or slot names.

---

# 12. Intent-to-Action Mapping

Refer to:

> **`Intent Schema.xlsx`**

This file defines how each VCM intent maps to the corresponding:

* Function
* API
* Action handler
* Expected arguments / slots

It should be treated as the authoritative source when implementing **Stage 3**.

The intended architecture is approximately:

```text
Dataset Schema.xlsx
        │
        ▼
Canonical Intent + Slot Schema
        │
        ▼
VCM Prediction
        │
        ▼
Intent Schema.xlsx
        │
        ▼
Action Handler
        │
        ▼
Function / API
```

---

# 13. Overall Application State Flow

The application should behave approximately as a state machine:

```text
┌───────────────────────────┐
│ Stage 1                   │
│ Wake Word Detection       │◄──────────────────────┐
└─────────────┬─────────────┘                       │
              │                                     │
     "Hey Alfred"                                   │
              │                                     │
              ▼                                     │
┌───────────────────────────┐                       │
│ Stage 2                   │                       │
│ Speech Capture + VCM      │                       │
└─────────────┬─────────────┘                       │
              │                                     │
       ┌──────┴───────┐                             │
       │              │                             │
   Confident       Invalid /                       │
   Prediction      Timeout /                       │
       │           Low Confidence                   │
       │              │                             │
       ▼              └─────────────────────────────┤
┌───────────────────────────┐                       │
│ Stage 3                   │                       │
│ Execute Action            │                       │
└─────────────┬─────────────┘                       │
              │                                     │
              └─────────────────────────────────────┘
```

The easter egg follows a separate path:

```text
Stage 1
   │
   │ "I'm Batman"
   ▼
Play Random Easter Egg Sound
   │
   └──────────────► Stage 1
```

---

# 14. Development Priorities

When working on this project, keep the following priorities in mind:

1. **Treat Alfred as a complete application.**
   The ML model is central, but audio capture, wake-word inference, state management, action execution, feedback sounds, and error handling are also part of the product.

2. **Treat the VCM as a first-class component.**
   Training, evaluation, dataset preparation, architecture selection, and inference quality should be handled rigorously.

3. **Maintain clear separation between the three stages.**
   Wake-word detection, VCM inference, and action execution should remain modular.

4. **Use the provided schemas as the source of truth.**
   Do not arbitrarily redefine intents, slots, or intent-to-action mappings.

5. **Prefer configurable parameters.**
   Values such as confidence thresholds, VAD timeout durations, model paths, and audio paths should be configurable where practical.

6. **Design for real-time interaction.**
   The user should receive immediate and understandable audio feedback as Alfred transitions between listening, understanding, and executing states.

7. **Avoid executing uncertain commands.**
   When confidence is insufficient, clarification is preferable to performing the wrong action.

---

# 15. Important Project Files

Before making architectural or implementation decisions, inspect the relevant project files.

| File                                    | Purpose                                                |
| --------------------------------------- | ------------------------------------------------------ |
| `ME2.md`                                | Machine Exercise requirements and constraints          |
| `Dataset Schema.xlsx`                   | Canonical intent and slot definitions                  |
| `Intent Schema.xlsx`                    | Mapping between intents, functions/APIs, and arguments |
| Wake-word ONNX model(s)                 | Pretrained openWakeWord models                         |
| `vcm.onnx`                               | Final self-contained VCM inference artifact            |
| `wake_word_detected.wav`                | Feedback when “Hey Alfred” is detected                 |
| `listening_start.wav`                   | Feedback indicating that Alfred is listening           |
| `listening_end.wav`                     | Feedback when listening ends                           |
| `easter_egg_1.wav` – `easter_egg_5.wav` | Random sounds for the “I’m Batman” easter egg          |

When implementation details conflict with assumptions in this primer, prioritize the explicit requirements in `ME2.md` and the agreed-upon schemas in the Excel files.
