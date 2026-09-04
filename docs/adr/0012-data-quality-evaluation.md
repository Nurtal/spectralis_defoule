# ADR 0012: Évaluation de l'impact de la qualité des données

**Date:** 2026-09-03
**Statut:** Accepté

## Contexte

Après l'obtention du baseline OFF (WER non-overlap 0.638) et du speaker beam (WER 0.573), nous avons cherché à comprendre si le bottleneck de performance venait de la qualité des données synthétiques (Piper TTS) ou de l'architecture du pipeline.

Deux hypothèses ont été testées :

1. **Le TTS Piper est trop médiocre** → remplacer par un meilleur TTS
2. **Les données synthétiques ne suffisent pas** → utiliser des données réelles (AMI Meeting Corpus)

## Décisions

### 1. Test TTS amélioré (edge-tts Microsoft)

Backend `edge-tts` ajouté dans `tts.py` avec mapping de 6 voix FR. Les données benchmark ont été générées avec les mêmes seeds que le baseline Piper.

**Résultats :**

| Métrique | Piper | Edge-TTS | Δ relatif |
|---|---|---|---|
| DER | 0.087 | 0.134 | +53% |
| WER non-overlap | 0.638 | 0.667 | +4.6% |
| pairwise_F1 | 0.507 | 0.458 | −9.7% |
| ARI | 0.292 | 0.198 | −32% |

**Conclusion :** Le problème n'est PAS la qualité TTS. Même avec des voix neurales Microsoft haute qualité, les métriques se dégradent. Le DER augmente significativement, ce qui suggère que les voix Microsoft ont des caractéristiques spectrales différentes de celles pour lesquelles le diariseur a été calibré.

### 2. Test données réelles (AMI Meeting Corpus)

Script `ami_fabricate.py` créé pour :
- Télécharger les segments AMI depuis HuggingFace (`edinburghcstr/ami`)
- Extraire les pistes individuelles par locuteur (16 speakers, 12643 segments)
- Fabriquer des scénarios de conversations parallèles en superposant des pistes de meetings différents

**Résultats :**

| Métrique | AMI fabriqué |
|---|---|
| DER | 1.231 |
| WER non-overlap | 1.000 |
| pairwise_F1 | 0.226 |
| ARI | −0.083 |

**Conclusion :** Le pipeline échoue complètement sur les données AMI. Causes identifiées :
- **Incompatibilité linguistique** : pipeline calibré pour le français, AMI est en anglais
- **Durée** : audio de 229s vs ~15s pour les synthétiques
- **Dynamiques naturelles** : meetings avec backchanneling, hésitations, chevauchements que le diariseur ne gère pas

## Conséquences

Le pipeline est **fortement dépendant des caractéristiques des données synthétiques**. Les améliorations doivent porter sur :

1. **L'architecture du pipeline** : le diariseur VAD + clustering est trop simpliste pour des données réelles
2. **Le modèle ASR** : Whisper small est insuffisant pour la reconnaissance de parole chevauchante
3. **Les données d'entraînement** : si on veut utiliser des données réelles, il faut un entraînement/finetuning sur ce type de données

## Fichiers concernés

- `src/conversation_deconvolution/synthetic/tts.py` : ajout `EdgeTts` + `create_tts()`
- `src/conversation_deconvolution/synthetic/ami_fabricate.py` : nouveau script d'extraction/fabrication AMI
- `src/conversation_deconvolution/core/config.py` : ajout `tts_backend` à `SyntheticConfig`
- `reports/benchmark_off.md` : baseline OFF
- `reports/benchmark_edge_tts.md` : résultats edge-tts
