"""
baseline_agent.py — Recurrent actor-critic model for MiniGrid.

Architecture:
  Visual:  3-layer ConvNet → flatten → LSTM (spatial + temporal features)
  Text:    Word embedding  → GRU        (mission string encoding)
  Fusion:  concat both streams → actor + critic heads

Implements the torch-ac RecurrentACModel interface for use with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch_ac


# --- Vocabulary ---
# Simple word-level tokenizer for MiniGrid mission strings
# (e.g. "open the yellow door" → [3, 1, 5, 2])

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


# --- Baseline Agent ---

class BaselineAgent(nn.Module, torch_ac.RecurrentACModel):
    """
    PPO-compatible recurrent agent for MiniGrid tasks.

    Takes dict observations with 'image' (7x7x3) and 'mission' (text),
    outputs action distribution + value estimate.
    """

    # network dimensions
    CONV_CHANNELS  = (16, 32, 64)
    LSTM_HIDDEN    = 128
    EMBED_DIM      = 32
    GRU_HIDDEN     = 128
    FFN_HIDDEN     = 64
    MAX_MISSION_LEN = 16

    def __init__(self, obs_space, action_space, vocab):
        super().__init__()

        n_actions = action_space.n

        # -- visual stream --
        # input: (batch, 3, 7, 7) after NHWC→NCHW transpose
        c1, c2, c3 = self.CONV_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )

        # after 3 conv layers (kernel=2, stride=1): 7→6→5→4
        # so output shape is (c3, 4, 4) = 64*16 = 1024
        conv_out_size = c3 * 4 * 4

        self.image_lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=self.LSTM_HIDDEN,
            batch_first=True,
        )

        # -- text stream --
        self.word_embedding = nn.Embedding(
            num_embeddings=256,         # enough for MiniGrid's small vocab
            embedding_dim=self.EMBED_DIM,
            padding_idx=0,
        )
        self.text_gru = nn.GRU(
            input_size=self.EMBED_DIM,
            hidden_size=self.GRU_HIDDEN,
            batch_first=True,
        )

        # -- actor-critic heads --
        fused_dim = self.LSTM_HIDDEN + self.GRU_HIDDEN  # 128 + 128 = 256

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

        self.vocab = vocab

    # -- torch-ac interface --

    @property
    def memory_size(self):
        """LSTM state size: we pack both h and c into one vector."""
        return 2 * self.LSTM_HIDDEN

    def forward(self, obs, memory):
        """
        obs.image: float (batch, 3, 7, 7)
        obs.text:  long  (batch, max_mission_len)
        memory:    (batch, 2*LSTM_HIDDEN) — packed LSTM state from prev step

        Returns: (action_dist, value, new_memory)
        """
        batch_size = obs.image.shape[0]

        # visual stream
        x = self.conv(obs.image)                         # (B, 64, 4, 4)
        x = x.reshape(batch_size, -1)                    # (B, 1024)
        x = x.unsqueeze(1)                               # (B, 1, 1024)

        # unpack LSTM state
        h = memory[:, :self.LSTM_HIDDEN].unsqueeze(0).contiguous()
        c = memory[:, self.LSTM_HIDDEN:].unsqueeze(0).contiguous()

        x, (h_new, c_new) = self.image_lstm(x, (h, c))
        visual_out = x.squeeze(1)                        # (B, 128)

        new_memory = torch.cat(
            [h_new.squeeze(0), c_new.squeeze(0)], dim=1
        )

        # text stream
        emb = self.word_embedding(obs.text)              # (B, L, 32)
        _, h_text = self.text_gru(emb)                   # (1, B, 128)
        text_out = h_text.squeeze(0)                     # (B, 128)

        # fuse and compute outputs
        fused = torch.cat([visual_out, text_out], dim=1) # (B, 256)

        logits = self.actor(fused)
        dist = Categorical(logits=logits)
        value = self.critic(fused).squeeze(1)

        return dist, value, new_memory
