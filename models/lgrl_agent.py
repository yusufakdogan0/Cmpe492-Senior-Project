"""
lgrl_agent.py — LGRL actor-critic model for MiniGrid.

Same architecture as the baseline but with a longer text input
to accommodate "mission [SEP] subgoal" concatenation.

Architecture:
  Visual:  3-layer ConvNet → flatten → LSTM
  Text:    Word embedding  → GRU  (processes "mission [SEP] subgoal")
  Fusion:  concat both streams → actor + critic heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch_ac


class LGRLAgent(nn.Module, torch_ac.RecurrentACModel):
    """
    PPO-compatible recurrent agent conditioned on both
    the mission string and the current LLM subgoal.

    The dual-text input (mission + subgoal) is concatenated with [SEP]
    by the preprocessor and tokenized into a single sequence.
    """

    # network dimensions (same as baseline)
    CONV_CHANNELS  = (16, 32, 64)
    LSTM_HIDDEN    = 128
    EMBED_DIM      = 32
    GRU_HIDDEN     = 128
    FFN_HIDDEN     = 64
    MAX_MISSION_LEN = 32   # doubled to fit "mission [SEP] subgoal"

    def __init__(self, obs_space, action_space, vocab):
        super().__init__()

        n_actions = action_space.n

        # -- visual stream --
        c1, c2, c3 = self.CONV_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
        )

        # 7→6→5→4, output = (64, 4, 4) = 1024
        conv_out_size = c3 * 4 * 4

        self.image_lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=self.LSTM_HIDDEN,
            batch_first=True,
        )

        # -- text stream --
        self.word_embedding = nn.Embedding(
            num_embeddings=256,
            embedding_dim=self.EMBED_DIM,
            padding_idx=0,
        )
        self.text_gru = nn.GRU(
            input_size=self.EMBED_DIM,
            hidden_size=self.GRU_HIDDEN,
            batch_first=True,
        )

        # -- actor-critic heads --
        fused_dim = self.LSTM_HIDDEN + self.GRU_HIDDEN

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
        obs.text:  long  (batch, MAX_MISSION_LEN)  — "mission [SEP] subgoal"
        memory:    (batch, 2*LSTM_HIDDEN)

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

        # text stream (processes "mission [SEP] subgoal" as one sequence)
        emb = self.word_embedding(obs.text)              # (B, L, 32)
        _, h_text = self.text_gru(emb)                   # (1, B, 128)
        text_out = h_text.squeeze(0)                     # (B, 128)

        # fuse and compute outputs
        fused = torch.cat([visual_out, text_out], dim=1) # (B, 256)

        # heads
        logits = self.actor(fused)                       # (B, n_actions)
        dist = Categorical(logits=logits)

        value = self.critic(fused).squeeze(1)            # (B,)

        return dist, value, new_memory
