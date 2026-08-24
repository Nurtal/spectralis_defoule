# ADR-0006: Dataset synthétique — Piper TTS FR + mixage DSP

## Status

Accepted

## Context

Évaluer chaque étage exige une vérité terrain contrôlée (locuteurs,
timings, textes, SNR, overlap). Les corpus réels annotés conversation-par-
conversation sont rares ; le README prévoit un générateur synthétique.

## Decision

Générateur seedé produisant :

1. Scénario aléatoire (n_conversations, locuteurs/conv, gaps exponentiels,
   longueurs d'énoncés, gains, SNR cible) ;
2. Synthèse TTS locale **Piper** (voix fr_FR distinctes : siwis, tom,
   upmc…), resampling 22,05 kHz→16 kHz ;
3. Mixage numpy : placement temporel + gains + bruit blanc filtré au SNR
   exact ;
4. Sorties `mixed.wav` + `ground_truth.json` (schéma identique à l'export
   pipeline).

## Alternatives considered

- edge-tts : voix correctes mais service cloud (contredit local-first).
- XTTS-v2 : gated HF + licence Coqui restrictive.
- Signaux DSP purs : tests rapides mais ASR inopérant dessus (gardé pour
  les tests unitaires déterministes).

## Consequences

### Positive

- Vérité terrain parfaite et paramétrable ; reproductibilité par seed.

### Negative

- Écart domaine TTS↔voix humaines (DER/WER optimistes vs réel).

## Reconsideration criteria

Besoin de réalisme accru → corpus réel (ex. AMI, Common Voice) en
complément, même schéma GT.
