# ADR-0001: Python 3.12 + uv comme socle du projet

## Status

Accepted

## Context

Le projet assemble des composants ML lourds (torch, speechbrain,
faster-whisper, sentence-transformers) sous forme de pipeline modulaire.
Il faut un outillage reproductible, rapide et compatible GPU CUDA.

## Decision

Python 3.12 géré par `uv` (verrouillage `uv.lock`, environnements
reproductibles). Torch installé depuis l'index CUDA cu124 (TITAN RTX,
sm_75 supporté). Qualité : ruff (lint+format), pytest.

## Alternatives considered

- poetry : plus lent, résolution moins prévisible pour les index extra.
- pip + requirements.txt : pas de verrouillage multi-plateforme propre.
- conda : plus lourd que nécessaire hors notebooks.

## Consequences

### Positive

- Résolution et installation très rapides ; lock déterministe.
- Gestion propre d'un index PyTorch explicite.

### Negative

- uv est jeune (API susceptible d'évoluer).

## Reconsideration criteria

Si un composant clé n'expose pas de wheels compatibles Python 3.12.
