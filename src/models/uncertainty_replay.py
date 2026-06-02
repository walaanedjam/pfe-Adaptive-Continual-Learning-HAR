"""
Uncertainty-Weighted Replay Buffer — contribution originale du PFE.

Idée centrale :
  Au lieu de tirer aléatoirement des exemples du tampon de rejeu,
  on priorise les exemples sur lesquels le modèle est le plus
  INCERTAIN (entropie de prédiction élevée).

Motivation :
  Un modèle qui oublie une ancienne classe va devenir incertain
  sur les exemples de cette classe AVANT de les oublier complètement.
  En les rejouant en priorité, on intervient exactement au bon moment.

Référence conceptuelle :
  Inspiré de "Prioritized Experience Replay" (Schaul et al., 2016)
  adapté au cadre de l'apprentissage continu pour la HAR.

Différence avec le replay classique :
  Replay classique     : P(x) = 1/|M|  (uniforme)
  Uncertainty replay   : P(x) ∝ H(f_θ(x))  (entropie de prédiction)

  où H(p) = -∑ p_i log(p_i) est l'entropie de Shannon.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .replay_buffer import ReplayBuffer


class UncertaintyWeightedReplayBuffer(ReplayBuffer):
    """
    Tampon de rejeu avec échantillonnage pondéré par l'incertitude.

    À chaque appel de sample_uncertain(), on calcule l'entropie
    de prédiction du modèle sur tout le buffer, puis on tire
    préférentiellement les exemples les plus incertains.

    Args:
        alpha: puissance de pondération (1.0 = proportionnel à l'entropie,
               0.0 = uniforme comme le replay classique)
        update_freq: recalculer les scores d'incertitude tous les N appels
    """

    def __init__(self, *args, alpha: float = 1.0,
                 update_freq: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha        = alpha
        self.update_freq  = update_freq
        self._call_count  = 0
        self._scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Calcul des scores d'incertitude
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_uncertainty_scores(self,
                                    model: torch.nn.Module,
                                    device: torch.device,
                                    batch_size: int = 256) -> np.ndarray:
        """
        Calcule l'entropie de prédiction pour chaque exemple du buffer.

        H(x) = -∑_c p_c log(p_c)   où p = softmax(f_θ(x))

        Retourne un vecteur de scores normalisé (somme = 1).
        """
        if len(self) == 0:
            return np.array([])

        model.eval()
        entropies = []

        X = self._X  # (N, T, C)
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start + batch_size]).to(device)
            logits = model(batch)                          # (B, n_classes)
            probs  = F.softmax(logits, dim=-1)             # (B, n_classes)
            H      = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B,)
            entropies.append(H.cpu().numpy())

        scores = np.concatenate(entropies)                 # (N,)
        # Normalise pour obtenir une distribution de probabilité
        scores = scores ** self.alpha
        total  = scores.sum()
        if total > 0:
            scores = scores / total
        else:
            scores = np.ones(len(scores)) / len(scores)

        return scores

    # ------------------------------------------------------------------
    # Échantillonnage pondéré
    # ------------------------------------------------------------------

    def sample_uncertain(self,
                          model: torch.nn.Module,
                          n: int,
                          device: torch.device,
                          as_tensor: bool = True
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Échantillonne n exemples en favorisant les plus incertains.

        Si alpha=0, équivalent au replay uniforme classique.
        Si alpha=1, proportionnel à l'entropie.
        Si alpha>1, encore plus agressif vers les plus incertains.
        """
        if len(self) == 0:
            raise RuntimeError("Tampon vide.")

        # Recalcule les scores périodiquement
        self._call_count += 1
        if (self._scores is None or
                len(self._scores) != len(self) or
                self._call_count % self.update_freq == 0):
            self._scores = self.compute_uncertainty_scores(model, device)

        n = min(n, len(self))

        # Tirage pondéré sans remise (si possible)
        try:
            idx = np.random.choice(len(self), n,
                                    replace=False, p=self._scores)
        except ValueError:
            # Fallback si les scores sont dégénérés
            idx = np.random.choice(len(self), n, replace=False)

        X_sample = self._X[idx]
        y_sample = self._y[idx]

        if as_tensor:
            return (torch.from_numpy(X_sample).to(device),
                    torch.from_numpy(y_sample).to(device))
        return X_sample, y_sample

    def mean_uncertainty(self) -> float:
        """Retourne l'entropie moyenne actuelle du buffer (indicateur d'oubli)."""
        if self._scores is None or len(self._scores) == 0:
            return 0.0
        # Re-normalise les scores bruts pour obtenir H moyen
        return float(np.mean(self._scores) * len(self._scores))
