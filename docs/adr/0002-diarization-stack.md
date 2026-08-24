# ADR-0002: Diarisation DIY — Silero VAD + ECAPA-TDNN + clustering

## Status

Accepted

## Context

La diarisation de référence (pyannote.audio 3.1) utilise des modèles gated :
token HuggingFace obligatoire et acceptation préalable des conditions.
Cela freine l'installation locale et la reproduction.

## Decision

Assembler notre diarisation :

1. VAD Silero v6 (`silero-vad`, poids embarqués dans le wheel) ;
2. Embeddings locuteur ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`,
   open), L2-normalisés ;
3. Clustering agglomératif sklearn (linkage average, distance cosinus),
   k imposé ou seuil de distance (~0.75).

Chaque brique est derrière une interface (`VadModel`, `SpeakerEmbedder`,
`Clusterer`) remplaçable indépendamment.

## Alternatives considered

- pyannote.audio 3.1 : meilleure DER out-of-the-box mais gated.
- NVIDIA NeMo : écosystème lourd, images volumineuses.
- whisperX : tire pyannote en dépendance (gated).

## Consequences

### Positive

- Installation locale sans compte ni token (principe local-first).
- Contrôle total de chaque étage ; instrumentation facile pour les métriques.

### Negative

- DER probablement inférieur à pyannote sur audio réel.
- Seuil de clustering à calibrer.

## Reconsideration criteria

DER insuffisant sur le benchmark malgré réglages → adaptateur pyannote si
un token est disponible (l'interface le permet sans changer le pipeline).
