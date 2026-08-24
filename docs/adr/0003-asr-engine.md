# ADR-0003: faster-whisper comme moteur ASR

## Status

Accepted

## Context

L'ASR doit tourner localement sur GPU (2× TITAN RTX), multilingue (focus
français), avec horodatage et score de confiance par segment, tout en
restant assez rapide pour des benchmarks itératifs.

## Decision

faster-whisper (port CTranslate2 de Whisper) : device CUDA, compute_type
float16, taille `small` par défaut (`tiny` en tests, `large-v3` possible),
détection de langue auto, confiance = moyenne exp(logprob) des tokens.
Les poids `Systran/faster-whisper-*` sont ouverts (non gated).

## Alternatives considered

- openai/whisper : plus lent (Passthrough PyTorch), pas de batching efficace.
- whisperX : apporte alignement+diarisation mais dépend de pyannote (gated).
- API cloud (OpenAI, Google) : contredit local-first.

## Consequences

### Positive

- Rapide sur GPU, mémoire maîtrisée (int8/float16), API simple.

### Negative

- CTranslate2 ajoute une binaire natif à la chaîne d'installation.

## Reconsideration criteria

Si WER français insuffisant → passer à large-v3 (même interface), ou
évaluer un modèle alternatif derrière `AsrEngine`.
