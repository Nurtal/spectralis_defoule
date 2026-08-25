# Conversation Deconvolution

> **Déconvolution de conversations multi-locuteurs à partir d'un
> enregistrement audio unique**

## 🎯 Vision

Conversation Deconvolution est un projet visant à transformer un
**enregistrement audio contenant plusieurs conversations simultanées**
en une représentation structurée des conversations individuelles.

### Entrée

Un unique flux audio, par exemple :

``` text
meeting_room.wav
```

contenant :

-   plusieurs locuteurs ;
-   plusieurs conversations parallèles ;
-   des prises de parole qui se chevauchent ;
-   du bruit ambiant ;
-   des interruptions ;
-   des changements de sujet.

### Sortie

Le système doit identifier les voix, transcrire leurs interventions et
reconstruire les conversations auxquelles elles appartiennent :

``` text
Conversation 1
────────────────────────────────
SPEAKER_A  → "Tu viens demain ?"
SPEAKER_B  → "Oui, vers 14 heures."
SPEAKER_A  → "Ok, parfait."

Conversation 2
────────────────────────────────
SPEAKER_C  → "Tu veux un café ?"
SPEAKER_D  → "Oui merci."
```

L'objectif n'est donc pas uniquement de faire de la **speech-to-text**
ou de la **diarisation**, mais d'inférer la structure latente des
conversations présentes dans un mélange audio.

------------------------------------------------------------------------

# 🧠 Problème

Le projet peut être décomposé en plusieurs sous-problèmes.

``` text
                         RAW AUDIO
                             │
                             ▼
                    ┌─────────────────┐
                    │ Voice Activity  │
                    │ Detection       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Diarisation     │
                    │ des locuteurs   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
             Speaker embeddings   Overlap detection
                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Speech          │
                    │ Separation      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ASR             │
                    │ (transcription) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Utterances      │
                    │ structurées     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Conversation    │
                    │ Reconstruction  │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
           Conversation A          Conversation B
```

## 1. Speaker diarization

Déterminer **qui parle quand** :

``` text
00:00.0 → 00:02.3   SPEAKER_1
00:01.8 → 00:04.1   SPEAKER_2
00:03.9 → 00:06.2   SPEAKER_1
```

Le cas d'usage cible comprend le **speech overlap**, ce qui rend la
diarisation classique insuffisante.

## 2. Speech separation

Séparer les différentes sources vocales lorsque plusieurs personnes
parlent simultanément :

``` text
                 MIXED AUDIO
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Voice A      Voice B      Voice C
```

Le projet pourra s'appuyer dans un premier temps sur des modèles
pré-entraînés de source separation plutôt que d'entraîner immédiatement
un modèle from scratch.

## 3. Automatic Speech Recognition

Transcrire chaque intervention en conservant ses métadonnées :

``` python
Utterance(
    speaker="SPEAKER_03",
    start=123.42,
    end=127.81,
    text="Tu as reçu le mail de Martin ?",
    confidence=0.91,
)
```

## 4. Conversation reconstruction

C'est le cœur du projet.

Le système doit déterminer quelles interventions appartiennent au même
fil conversationnel.

Deux interventions peuvent être reliées par plusieurs signaux :

-   proximité temporelle ;
-   chevauchement ;
-   identité du locuteur ;
-   alternance des locuteurs ;
-   durée des silences ;
-   embeddings des locuteurs ;
-   similarité sémantique ;
-   relation question/réponse ;
-   continuité du sujet ;
-   indices prosodiques ;
-   éventuellement informations spatiales provenant de plusieurs
    microphones.

À terme, le problème peut être modélisé comme un **graphe
d'interventions** :

``` text
                  Utterance A1
                       │
                  0.93 │
                       ▼
                  Utterance B1
                       │
                  0.87 │
                       ▼
                  Utterance A2


                  Utterance C1
                       │
                  0.91 │
                       ▼
                  Utterance D1
```

Chaque intervention constitue un nœud et chaque relation entre deux
interventions possède un score indiquant la probabilité qu'elles
appartiennent au même fil conversationnel.

------------------------------------------------------------------------

# 🏗️ Architecture cible

L'architecture sera volontairement modulaire afin de permettre le
remplacement indépendant des modèles et composants.

``` text
                    ┌────────────────────┐
                    │     Audio Input    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Pre-processing     │
                    │ / VAD              │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Speaker            │
                    │ Diarization        │
                    └─────────┬──────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Speaker IDs      Overlap info      Embeddings
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌────────────────────┐
                    │ Speech Separation  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ ASR                │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Utterance Store    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Conversation       │
                    │ Reconstruction     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Structured Output  │
                    └────────────────────┘
```

------------------------------------------------------------------------

# 🧪 MVP

Le premier objectif n'est **pas** de créer un nouveau modèle de deep
learning.

Le MVP doit assembler des composants existants afin de valider la
faisabilité du pipeline.

### Pipeline MVP

1.  Charger un fichier audio.
2.  Détecter les segments de parole.
3.  Effectuer une première diarisation.
4.  Détecter les zones de chevauchement.
5.  Appliquer la séparation audio uniquement lorsque nécessaire.
6.  Transcrire les segments.
7.  Associer chaque transcription à un locuteur.
8.  Construire les relations entre interventions.
9.  Regrouper les interventions en conversations.
10. Exporter le résultat en JSON.

Exemple :

``` json
{
  "conversations": [
    {
      "id": "conversation_01",
      "participants": ["speaker_01", "speaker_02"],
      "utterances": [
        {
          "speaker": "speaker_01",
          "start": 12.4,
          "end": 15.2,
          "text": "Tu viens demain ?"
        },
        {
          "speaker": "speaker_02",
          "start": 15.4,
          "end": 18.1,
          "text": "Oui, vers quatorze heures."
        }
      ]
    }
  ]
}
```

------------------------------------------------------------------------

# 📊 Évaluation

Le projet doit séparer l'évaluation de chaque composant.

  Composant                     Métriques envisagées
  ----------------------------- ----------------------------------
  VAD                           Precision / Recall / F1
  Diarisation                   DER
  Speaker attribution           Accuracy / F1
  Speech separation             SI-SDR / SDR
  ASR                           WER
  Overlap detection             Precision / Recall / F1
  Conversation reconstruction   ARI / NMI / F1
  Pipeline global               End-to-end conversation accuracy

## Dataset synthétique

Un dataset synthétique permettra de disposer d'une vérité terrain
contrôlée.

``` text
Conversation A ───────┐
Conversation B ───────┼──► MIXER ──► mixed.wav
Conversation C ───────┘
```

Les paramètres pourront être randomisés :

-   nombre de locuteurs ;
-   nombre de conversations ;
-   durée ;
-   SNR ;
-   volume relatif ;
-   taux de chevauchement ;
-   silences ;
-   interruptions ;
-   vitesse de parole ;
-   décalage temporel.

Cela permettra d'évaluer précisément les performances et d'identifier
les limites de chaque étape.

------------------------------------------------------------------------

# 🧩 Technologies envisagées

Le projet privilégie initialement les modèles et outils open source.

Composants possibles :

-   **Python** comme langage principal ;
-   modèles de **speaker diarization** ;
-   modèles de **speech separation** ;
-   **Whisper** ou équivalent pour l'ASR ;
-   embeddings de locuteurs ;
-   modèles d'embeddings sémantiques ;
-   algorithmes de clustering / community detection ;
-   éventuellement modèles de graph machine learning pour la
    reconstruction des conversations.

Les choix technologiques précis seront documentés dans les ADR.

------------------------------------------------------------------------

# 📝 Architectural Decision Records

Le projet utilise des **ADR --- Architectural Decision Records**.

Un ADR documente une décision d'architecture importante, son contexte,
les alternatives envisagées et les raisons du choix.

Les ADR seront stockés dans :

``` text
docs/adr/
```

Exemple :

``` text
docs/
└── adr/
    ├── 0001-use-python.md
    ├── 0002-diarization-framework.md
    ├── 0003-asr-engine.md
    ├── 0004-audio-separation-strategy.md
    └── 0005-conversation-reconstruction-model.md
```

### Format recommandé

``` markdown
# ADR-000X: Decision title

## Status

Accepted

## Context

Pourquoi cette décision est nécessaire.

## Decision

Décision retenue.

## Alternatives considered

- Alternative A
- Alternative B
- Alternative C

## Consequences

### Positive

- ...

### Negative

- ...

## Reconsideration criteria

Conditions pouvant justifier la révision de cette décision.
```

Les ADR sont considérés comme une partie intégrante de l'architecture du
projet et non comme une documentation ajoutée a posteriori.

------------------------------------------------------------------------

# 🗺️ ROADMAP

## Phase 0 --- Architecture & cadrage

-   [ ] Initialiser le repository.
-   [ ] Définir la structure du projet.
-   [ ] Mettre en place les ADR.
-   [ ] Définir le format `Utterance`.
-   [ ] Définir le format de sortie `Conversation`.
-   [ ] Définir les métriques d'évaluation.

**Objectif :** disposer d'une architecture et d'interfaces stables avant
de multiplier les expérimentations.

------------------------------------------------------------------------

## Phase 1 --- Baseline audio → transcription

-   [ ] Chargement audio.
-   [ ] VAD.
-   [ ] ASR.
-   [ ] Horodatage des segments.
-   [ ] Export JSON.
-   [ ] Première visualisation timeline.

**Objectif :** obtenir une transcription temporelle exploitable.

------------------------------------------------------------------------

## Phase 2 --- Diarisation

-   [ ] Intégrer un modèle de diarisation.
-   [ ] Identifier les locuteurs.
-   [ ] Générer les speaker embeddings.
-   [ ] Évaluer le DER.
-   [ ] Gérer les changements de locuteur.
-   [ ] Évaluer les cas multi-locuteurs.

**Objectif :**

``` text
timestamp + speaker + text
```

------------------------------------------------------------------------

## Phase 3 --- Overlap & speech separation

-   [ ] Détecter les chevauchements.
-   [ ] Identifier les segments nécessitant une séparation.
-   [ ] Intégrer un modèle de speech separation.
-   [ ] Comparer séparation systématique vs séparation conditionnelle.
-   [ ] Évaluer la qualité audio.
-   [ ] Mesurer l'impact sur le WER.

**Objectif :** améliorer la transcription lorsque plusieurs personnes
parlent simultanément.

------------------------------------------------------------------------

## Phase 4 --- Reconstruction des conversations

-   [ ] Définir un modèle de relation entre utterances.
-   [ ] Implémenter une baseline heuristique.
-   [ ] Ajouter les features temporelles.
-   [ ] Ajouter les speaker embeddings.
-   [ ] Ajouter les embeddings sémantiques.
-   [ ] Tester le clustering.
-   [ ] Tester une représentation en graphe.
-   [ ] Évaluer ARI / NMI / F1.

**Objectif :**

``` text
Utterances
    ↓
Conversation clusters
```

------------------------------------------------------------------------

## Phase 5 --- Dataset synthétique

-   [ ] Construire le générateur de conversations.
-   [ ] Mixer automatiquement plusieurs conversations.
-   [ ] Contrôler le SNR.
-   [ ] Contrôler le taux d'overlap.
-   [ ] Générer la vérité terrain.
-   [ ] Mettre en place les benchmarks.
-   [ ] Automatiser les évaluations.

**Objectif :** pouvoir mesurer objectivement chaque évolution du
système.

------------------------------------------------------------------------

## Phase 6 --- Conversation Graph

Passer d'un clustering simple à une représentation explicite en graphe.

-   [ ] Définir les nœuds.
-   [ ] Définir les arêtes.
-   [ ] Construire les features.
-   [ ] Apprendre la probabilité de connexion.
-   [ ] Tester community detection.
-   [ ] Tester graph neural networks si nécessaire.
-   [ ] Comparer avec les baselines.

**Objectif :**

> Inférer le graphe latent des conversations à partir du flux audio.

------------------------------------------------------------------------

## Phase 7 --- Robustesse

Tester le système dans des conditions réalistes :

-   [ ] bruit de fond ;
-   [ ] réverbération ;
-   [ ] conversations simultanées ;
-   [ ] interruptions ;
-   [ ] locuteurs très proches ;
-   [ ] locuteurs éloignés ;
-   [ ] conversations longues ;
-   [ ] changements fréquents de sujet ;
-   [ ] nombre variable de locuteurs.

------------------------------------------------------------------------

## Phase 8 --- Produit / démonstrateur

Créer une interface permettant :

1.  d'uploader un fichier audio ;
2.  de visualiser les locuteurs ;
3.  de voir la timeline ;
4.  de visualiser les chevauchements ;
5.  de consulter les transcriptions ;
6.  de voir les conversations reconstruites ;
7.  de corriger manuellement les associations ;
8.  d'exporter les résultats.

------------------------------------------------------------------------

# 🚀 Vision long terme

À terme, le système pourrait évoluer vers une représentation complète :

``` text
                         AUDIO
                           │
                           ▼
                  ┌─────────────────┐
                  │ Acoustic World  │
                  └────────┬────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Speaker / Utterance │
                │ extraction          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Conversation Graph  │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          Speaker A    Speaker B    Speaker C
              │            │            │
              └────────────┼────────────┘
                           ▼
                    Conversation 1
                    Conversation 2
                    Conversation 3
```

Le système ne serait alors plus simplement un outil de transcription,
mais un **moteur d'analyse de scènes conversationnelles** capable
d'inférer :

-   qui parle ;
-   quand ;
-   avec qui ;
-   de quoi ;
-   dans quelle conversation ;
-   avec quelles incertitudes.

------------------------------------------------------------------------

# ⚠️ Principes du projet

### Modularité

Chaque composant doit pouvoir être remplacé sans réécrire l'ensemble du
pipeline.

### Mesurabilité

Chaque évolution doit pouvoir être comparée à une baseline.

### Reproductibilité

Les expériences doivent être reproductibles avec une configuration
explicite.

### Local-first

Lorsque cela est possible, les traitements audio et les modèles doivent
pouvoir fonctionner localement.

### ADR-driven architecture

Les décisions structurantes doivent être documentées dans des ADR.

### Ne pas sur-ingénieriser trop tôt

Le projet doit privilégier :

``` text
baseline fonctionnelle
        ↓
mesure
        ↓
identification du bottleneck
        ↓
amélioration
        ↓
mesure
```

plutôt que de construire immédiatement un système complexe.

------------------------------------------------------------------------

# 📁 Structure initiale proposée

``` text
conversation-deconvolution/
│
├── README.md
├── pyproject.toml
│
├── src/
│   └── conversation_deconvolution/
│       ├── audio/
│       ├── diarization/
│       ├── separation/
│       ├── asr/
│       ├── conversation/
│       ├── graph/
│       └── evaluation/
│
├── tests/
│
├── experiments/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
└── docs/
    └── adr/
```

------------------------------------------------------------------------

# 🚀 Quickstart

``` bash
# installation (Python 3.12 + uv, GPU CUDA recommandé)
uv sync

# générer un dataset synthétique (2 conversations × 2 locuteurs, ~20 s)
uv run deconvolute synth --out data/synthetic/sample --seed 0

# lancer le pipeline complet sur un audio
uv run deconvolute run data/synthetic/sample/mixed.wav \
    -o out.json -n 4 -p timeline.png

# évaluer une prédiction contre la vérité terrain
uv run deconvolute evaluate -p out.json -g data/synthetic/sample/ground_truth.json

# benchmark complet (N datasets seedés → rapport Markdown)
make benchmark
```

Structure détaillée : `docs/superpowers/specs/` (design), `docs/adr/`
(décisions), `docs/ROADMAP.md` (jalons), `docs/superpowers/plans/`
(plan d'implémentation).

# 📌 Statut

**Projet :** expérimental / R&D

**Phase actuelle :** Phases 0→5 implémentées (baseline mesurable) ;
M6 clôturé (ADR-0009, code conservé derrière `--reconstructor graph`) ;
N4 : séparation ON évaluée (splice in-place, ADR-0008 renforcé) — trois
variantes ASR-par-tige dégradent WER overlap vs OFF, verdict OFF maintenu ;
N3 fait (normalisation WER).

**Prochaine étape :** itérations amont qualité transcription — N1 (refonte
SM-TFNet → Lite-TFNet) et N2 (beam search dépendant-locuteur) pour
réduire le WER non-overlap de 0,57 ; ré-évaluer ensuite si la séparation
peut devenir rentable.
