"""
Baseline PPO Agent for MiniGrid environments.

Architecture:
  Visual stream  : 3-layer ConvNet  → flatten → LSTM  (spatial + temporal)
  Text stream    : Embedding        → GRU              (mission encoding)
  Fusion         : concat(LSTM_out, GRU_out)
  Actor head     : FFN → Categorical distribution
  Critic head    : FFN → scalar value

Conforms to the torch-ac RecurrentACModel interface so it can be
plugged directly into torch_ac.PPOAlgo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch_ac


# ──────────────────────────────────────────────
# Vocabulary  — tiny tokenizer for MiniGrid
# mission strings  (e.g. "open the yellow door")
# ──────────────────────────────────────────────

class Vocabulary:
    """Word-level vocabulary built on the fly from MiniGrid missions."""

    PAD_TOKEN = "<PAD>"

    def __init__(self):
        self.word2idx = {self.PAD_TOKEN: 0}
        self.idx2word = [self.PAD_TOKEN]

    def __getitem__(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)

    def tokenize(self, mission: str, max_len: int = 16) -> list[int]:
        """Convert a mission string to a fixed-length list of token ids."""
        tokens = [self[w] for w in mission.lower().split()]
        # Pad or truncate to max_len
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens


# ──────────────────────────────────────────────
# Baseline Agent
# ──────────────────────────────────────────────

class BaselineAgent(nn.Module, torch_ac.RecurrentACModel):
    """
    Recurrent actor-critic model for MiniGrid.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Dict
        Must contain 'image' (7×7×3) and 'mission' (text).
    action_space : gymnasium.spaces.Discrete
        Number of discrete actions.
    vocab : Vocabulary
        Shared vocabulary instance (built during preprocessing).
    """

    # ---- hyper-parameters (easy to override later) ----
    CONV_CHANNELS  = (16, 32, 64)
    LSTM_HIDDEN    = 128
    EMBED_DIM      = 32
    GRU_HIDDEN     = 128
    FFN_HIDDEN     = 64
    MAX_MISSION_LEN = 16

    def __init__(self, obs_space, action_space, vocab: Vocabulary):
        super().__init__()

        n_actions = action_space.n

        # ---------- visual stream ----------
        # Input: (batch, 3, 7, 7)  — channels-first after transpose
        c1, c2, c3 = self.CONV_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )

        # Compute flattened size after conv layers on a 7×7 input
        # Each Conv2d(kernel=2, stride=1, pad=0) reduces spatial dim by 1
        # 7 → 6 → 5 → 4    so output is (c3, 4, 4) = 64 * 16 = 1024
        conv_out_size = c3 * 4 * 4

        self.image_lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=self.LSTM_HIDDEN,
            batch_first=True,
        )

        # ---------- text stream ----------
        self.word_embedding = nn.Embedding(
            num_embeddings=256,          # generous upper bound for MiniGrid vocab
            embedding_dim=self.EMBED_DIM,
            padding_idx=0,
        )
        self.text_gru = nn.GRU(
            input_size=self.EMBED_DIM,
            hidden_size=self.GRU_HIDDEN,
            batch_first=True,
        )

        # ---------- fusion / heads ----------
        fused_dim = self.LSTM_HIDDEN + self.GRU_HIDDEN  # 256

        self.actor = nn.Sequential(
            nn.Linear(fused_dim, self.FFN_HIDDEN),
            nn.ReLU(),
            nn.Linear(self.FFN_HIDDEN, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(fused_dim, self.FFN_HIDDEN),
            nn.ReLU(),
            nn.Linear(self.FFN_HIDDEN, 1),
        )

        # store vocab reference (used by the preprocessor, not by forward)
        self.vocab = vocab

    # ---- torch-ac interface --------------------------------

    @property
    def memory_size(self) -> int:
        """Size of the recurrent hidden state expected by torch-ac."""
        #  We store both (h, c) of the LSTM packed into one vector.
        return 2 * self.LSTM_HIDDEN

    def forward(self, obs, memory):
        """
        Parameters
        ----------
        obs : namespace/object with
            .image  — float tensor (batch, 3, 7, 7)
            .text   — long  tensor (batch, max_mission_len)
        memory : tensor (batch, memory_size)
            Packed LSTM (h, c) from previous step.

        Returns
        -------
        dist   : Categorical  — action distribution
        value  : Tensor       — value estimate  (batch,)
        memory : Tensor       — updated hidden state (batch, memory_size)
        """
        batch_size = obs.image.shape[0]

        # ---- visual stream ----
        x = self.conv(obs.image)                         # (B, 64, 4, 4)
        x = x.reshape(batch_size, -1)                    # (B, 1024)
        x = x.unsqueeze(1)                               # (B, 1, 1024) — single time-step

        # Unpack LSTM hidden state from memory
        h = memory[:, :self.LSTM_HIDDEN].unsqueeze(0).contiguous()   # (1, B, 128)
        c = memory[:, self.LSTM_HIDDEN:].unsqueeze(0).contiguous()   # (1, B, 128)

        x, (h_new, c_new) = self.image_lstm(x, (h, c))  # x: (B, 1, 128)
        visual_out = x.squeeze(1)                        # (B, 128)

        # Pack updated hidden state back into memory
        new_memory = torch.cat(
            [h_new.squeeze(0), c_new.squeeze(0)], dim=1
        )                                                # (B, 256)

        # ---- text stream ----
        emb = self.word_embedding(obs.text)              # (B, L, 32)
        _, h_text = self.text_gru(emb)                   # h_text: (1, B, 128)
        text_out = h_text.squeeze(0)                     # (B, 128)

        # ---- fusion ----
        fused = torch.cat([visual_out, text_out], dim=1) # (B, 256)

        # ---- heads ----
        logits = self.actor(fused)                       # (B, n_actions)
        dist = Categorical(logits=logits)

        value = self.critic(fused).squeeze(1)            # (B,)

        return dist, value, new_memory
