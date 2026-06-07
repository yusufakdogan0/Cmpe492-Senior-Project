"""
crafter_agent.py — recurrent actor-critic model for Crafter, plus a
self-contained word vocabulary.

The Crafter analogue of the MiniGrid project's ``lgrl_agent.py`` /
``baseline_agent.py``. A SINGLE architecture serves both the baseline and
the LGRL agent — exactly as in the paper, where the two differ only in
their text input (baseline = mission only; LGRL = "mission [SEP] subgoal").
Which text is fed is decided by the training script's preprocessing, not
by the model.

Streams (mirrors the MiniGrid design, resized for Crafter):
  Visual: 64x64x3 image -> 3-layer Nature-style ConvNet -> flatten -> LSTM
  Text:   word embedding -> GRU
  Fusion: concat -> actor + critic heads

The conv stack is the only real change from the MiniGrid agent: MiniGrid's
7x7 input used three kernel-2/stride-1 layers; Crafter's 64x64 input needs
a downsampling stack (k8s4 -> k4s2 -> k3s1 -> 4x4x64 = 1024), after which
everything downstream (LSTM 128 + GRU 128 -> heads) is identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

import torch_ac


# ---------------------------------------------------------------------------
# Vocabulary (self-contained copy so crafter/ does not depend on models/).
# ---------------------------------------------------------------------------

class Vocabulary:
    """Word-level vocabulary built on the fly from Crafter mission/subgoal
    strings (e.g. 'make a stone pickaxe' -> token ids)."""

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

    def tokenize(self, text: str, max_len: int = 32) -> list[int]:
        tokens = [self[w] for w in text.lower().split()]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CrafterACModel(nn.Module, torch_ac.RecurrentACModel):
    """PPO-compatible recurrent agent for Crafter.

    Dict observations with 'image' (64x64x3) and tokenized 'text'.
    Used by BOTH the baseline (mission-only text) and LGRL
    (mission + subgoal text) training scripts.
    """

    LSTM_HIDDEN = 128
    EMBED_DIM = 32
    GRU_HIDDEN = 128
    FFN_HIDDEN = 64
    MAX_TEXT_LEN = 32

    def __init__(self, obs_space, action_space, vocab):
        super().__init__()
        n_actions = action_space.n

        # -- visual stream: 64x64x3 -> (64,4,4) = 1024 --
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),   # 64 -> 15
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),  # 15 -> 6
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),  # 6  -> 4
        )
        conv_out_size = 64 * 4 * 4  # 1024

        self.image_lstm = nn.LSTM(
            input_size=conv_out_size, hidden_size=self.LSTM_HIDDEN,
            batch_first=True,
        )

        # -- text stream --
        self.word_embedding = nn.Embedding(
            num_embeddings=256, embedding_dim=self.EMBED_DIM, padding_idx=0,
        )
        self.text_gru = nn.GRU(
            input_size=self.EMBED_DIM, hidden_size=self.GRU_HIDDEN,
            batch_first=True,
        )

        # -- actor / critic heads --
        fused_dim = self.LSTM_HIDDEN + self.GRU_HIDDEN  # 256
        self.actor = nn.Sequential(
            nn.Linear(fused_dim, self.FFN_HIDDEN), nn.ReLU(),
            nn.Linear(self.FFN_HIDDEN, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(fused_dim, self.FFN_HIDDEN), nn.ReLU(),
            nn.Linear(self.FFN_HIDDEN, 1),
        )

        self.vocab = vocab

    @property
    def memory_size(self):
        return 2 * self.LSTM_HIDDEN  # packed (h, c)

    def forward(self, obs, memory):
        batch_size = obs.image.shape[0]

        # visual stream
        x = self.conv(obs.image)               # (B, 64, 4, 4)
        x = x.reshape(batch_size, -1)          # (B, 1024)
        x = x.unsqueeze(1)                     # (B, 1, 1024)

        h = memory[:, :self.LSTM_HIDDEN].unsqueeze(0).contiguous()
        c = memory[:, self.LSTM_HIDDEN:].unsqueeze(0).contiguous()
        x, (h_new, c_new) = self.image_lstm(x, (h, c))
        visual_out = x.squeeze(1)              # (B, 128)
        new_memory = torch.cat([h_new.squeeze(0), c_new.squeeze(0)], dim=1)

        # text stream
        emb = self.word_embedding(obs.text)    # (B, L, 32)
        _, h_text = self.text_gru(emb)         # (1, B, 128)
        text_out = h_text.squeeze(0)           # (B, 128)

        # fuse
        fused = torch.cat([visual_out, text_out], dim=1)  # (B, 256)
        dist = Categorical(logits=self.actor(fused))
        value = self.critic(fused).squeeze(1)
        return dist, value, new_memory


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from gymnasium import spaces

    vocab = Vocabulary()
    obs_space = spaces.Dict({"image": spaces.Box(0, 255, (64, 64, 3), np.uint8)})
    act_space = spaces.Discrete(17)
    model = CrafterACModel(obs_space, act_space, vocab)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"CrafterACModel params: {n_params:,}")
    print(f"memory_size: {model.memory_size}")

    B = 4
    image = torch.rand(B, 3, 64, 64)
    text = torch.tensor([vocab.tokenize("make a stone pickaxe [SEP] collect 2 wood")
                         for _ in range(B)], dtype=torch.long)
    obs = torch_ac.DictList({"image": image, "text": text})
    memory = torch.zeros(B, model.memory_size)

    dist, value, new_memory = model(obs, memory)
    print("action logits shape:", dist.logits.shape)        # (4, 17)
    print("value shape:", value.shape)                      # (4,)
    print("new_memory shape:", new_memory.shape)            # (4, 256)
    actions = dist.sample()
    print("sampled actions:", actions.tolist())
    assert dist.logits.shape == (B, 17)
    assert value.shape == (B,)
    assert new_memory.shape == (B, model.memory_size)
    print("crafter_agent self-test OK")
