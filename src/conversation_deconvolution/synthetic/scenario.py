from dataclasses import dataclass, field

import numpy as np

from conversation_deconvolution.core.types import Conversation, Utterance

TOPIC_BANKS = [
    [
        "Tu viens au cafe demain midi ?",
        "Oui je passe vers douze heures.",
        "Parfait je reserve une table.",
        "Tu veux que j amene quelque chose ?",
        "Non ne t embarrasse pas merci.",
        "On se retrouve devant l entree alors.",
        "D accord a demain bonsoir.",
    ],
    [
        "Le rapport final est-il termine ?",
        "Oui je le termine cet apres-midi.",
        "Tu peux l envoyer au directeur ?",
        "Je l envoie des que c est corrige.",
        "Merci beaucoup pour ton aide.",
        "Il faudra aussi preparer la presentation.",
        "Pas de probleme je m en occupe.",
    ],
    [
        "Tu as vu le match hier soir ?",
        "Oui quelle victoire incroyable non ?",
        "Le dernier but etait magnifique.",
        "J espere qu ils gagnent encore dimanche.",
        "On regarde ca ensemble chez moi ?",
        "Bonne idea j apporte les boissons.",
    ],
    [
        "La voiture est reparee finalement ?",
        "Oui le mecanique a change la courroie.",
        "Ca coute cher combien au total ?",
        "Environ trois cents euros je crois.",
        "C est raisonnable pour ce travail.",
    ],
]

VOICES = ["siwis", "tom", "upmc", "mls", "mls_1840", "gilles"]


@dataclass
class ScenarioLine:
    speaker: str
    voice: str
    text: str


@dataclass
class ScenarioThread:
    speakers: list[str] = field(default_factory=list)
    lines: list[ScenarioLine] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    topic: int = 0


def generate_scenario(
    n_conversations: int,
    speakers_per_thread: int,
    n_lines: tuple[int, int],
    rng: np.random.Generator,
    mean_gap_sec: float = 0.8,
) -> list[ScenarioThread]:
    threads = []
    voice_pool = VOICES.copy()
    rng.shuffle(voice_pool)
    for conv in range(n_conversations):
        voices = [voice_pool[(conv * speakers_per_thread + k) % len(voice_pool)] for k in range(speakers_per_thread)]
        speakers = [f"speaker_{conv + 1:02d}_{chr(65 + k)}" for k in range(speakers_per_thread)]
        bank = TOPIC_BANKS[conv % len(TOPIC_BANKS)]
        n = rng.integers(n_lines[0], n_lines[1] + 1)
        lines = []
        gaps = []
        for k in range(int(n)):
            spk_idx = k % speakers_per_thread
            text = bank[int(rng.integers(0, len(bank)))]
            if k > 0 and lines and lines[-1].text == text:
                text = bank[(bank.index(text) + 3) % len(bank)]
            lines.append(
                ScenarioLine(
                    speaker=speakers[spk_idx],
                    voice=voices[spk_idx],
                    text=text,
                )
            )
            gaps.append(max(0.15, float(rng.exponential(mean_gap_sec))))
        threads.append(
            ScenarioThread(speakers=speakers, lines=lines, gaps=gaps[1:], topic=conv)
        )
    return threads


def thread_to_conversation(thread: ScenarioThread, utterances: list[Utterance], conv_id: str) -> Conversation:
    return Conversation(id=conv_id, participants=list(thread.speakers), utterances=utterances)
