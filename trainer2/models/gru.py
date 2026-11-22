"""GRU sequence-to-sequence model with attention."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets.vocab import PAD


class Encoder(nn.Module):
    """Bidirectional GRU that reads the Roman word."""
    def __init__(self, vocab: int, emb: int, hid: int, layers: int, drop: float) -> None:
        super().__init__()
        # Character ids -> vectors
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        # Bidirectional GRU (reads left-to-right and right-to-left)
        self.rnn = nn.GRU(
            emb,
            hid,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop if layers > 1 else 0.0,
        )
        # Dropout against over-fitting
        self.drop = nn.Dropout(drop)
        # Joins the two directions into one state
        self.bridge = nn.Linear(2 * hid, hid)
        # Remember the sizes
        self.layers = layers
        self.hid = hid

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the output for each character and the final hidden state."""
        # Real length of each word (padding excluded)
        lengths = (src != PAD).sum(1).clamp(min=1).cpu()
        # Embed the characters
        embedded = self.drop(self.emb(src))
        # Pack the batch so padding is skipped
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        # Run the GRU
        # Move the GRU forward one step
        output, hidden = self.rnn(packed)
        # Unpack back to a padded tensor
        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, batch_first=True, total_length=src.size(1)
        )
        # Split into forward and backward directions
        hidden = hidden.view(self.layers, 2, src.size(0), self.hid)
        # Merge both directions into one state
        hidden = torch.tanh(self.bridge(torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)))
        return output, hidden


class BahdanauAttention(nn.Module):
    """Attention: how much each input character matters for the next output."""
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int) -> None:
        super().__init__()
        # Projects the encoder outputs
        self.W_e = nn.Linear(enc_dim, attn_dim, bias=False)
        # Projects the decoder state
        self.W_d = nn.Linear(dec_dim, attn_dim, bias=True)
        # Turns each pair into one score
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, dec_hidden: torch.Tensor, enc_output: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the context vector and the attention weights."""
        # Compare the decoder state with every input character
        energy = torch.tanh(self.W_e(enc_output) + self.W_d(dec_hidden).unsqueeze(1))
        # One score per input character
        scores = self.v(energy).squeeze(-1)
        # Hide the padding positions
        scores = scores.float().masked_fill(~mask, -1e4)
        # Scores -> attention weights that sum to 1
        weights = F.softmax(scores, dim=-1)
        # Weighted sum of the encoder outputs
        context = torch.bmm(weights.unsqueeze(1).to(enc_output.dtype), enc_output).squeeze(1)
        return context, weights


class AttnDecoder(nn.Module):
    """GRU decoder that looks at the input through attention."""
    def __init__(
        self,
        vocab: int,
        emb: int,
        hid: int,
        enc_dim: int,
        layers: int,
        drop: float,
        attn_dim: int = 256,
    ) -> None:
        super().__init__()
        # Character ids -> vectors
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        # Attention over the input
        self.attn = BahdanauAttention(enc_dim, hid, attn_dim)
        # The GRU gets the character and the attention context
        self.rnn = nn.GRU(
            emb + enc_dim,
            hid,
            layers,
            batch_first=True,
            dropout=drop if layers > 1 else 0.0,
        )
        # Scores for every output character
        self.out = nn.Linear(hid + enc_dim + emb, vocab)
        # Dropout against over-fitting
        self.drop = nn.Dropout(drop)

    def step(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        enc_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score the next character for one step."""
        # Embed the previous character
        embedded = self.drop(self.emb(token))
        # Look at the input, guided by the top layer state
        context, _ = self.attn(hidden[-1], enc_output, mask)
        output, hidden = self.rnn(
            torch.cat([embedded, context], dim=-1).unsqueeze(1), hidden
        )
        # Scores for the next character
        logits = self.out(self.drop(torch.cat([output.squeeze(1), context, embedded], dim=-1)))
        return logits, hidden


class GRUSeq2Seq(nn.Module):
    """Full model: encoder + attention decoder."""
    def __init__(
        self,
        source_vocab: int,
        target_vocab: int,
        emb: int = 192,
        hid: int = 256,
        layers: int = 2,
        drop: float = 0.25,
    ) -> None:
        super().__init__()
        # Reads the input
        self.enc = Encoder(source_vocab, emb, hid, layers, drop)
        # Writes the output
        self.dec = AttnDecoder(target_vocab, emb, hid, 2 * hid, layers, drop)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Training pass: scores for every output position."""
        # True where the input is a real character
        mask = src != PAD
        # Encode the input
        enc_output, hidden = self.enc(src)
        # Scores of every output position
        outputs = []
        # One decoding step per target position
        for index in range(tgt_in.size(1)):
            logits, hidden = self.dec.step(tgt_in[:, index], hidden, enc_output, mask)
            outputs.append(logits)
        # Shape: batch x length x vocabulary
        return torch.stack(outputs, dim=1)

    def encode(
        self, src: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Encode the input once (used by beam search)."""
        # True where the input is a real character
        mask = src != PAD
        # Encode the input
        enc_output, hidden = self.enc(src)
        # Memory for the decoder, plus the starting state
        return (enc_output, mask), hidden

    def step(
        self,
        tokens: torch.Tensor,
        state: torch.Tensor,
        memory: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One beam-search step: log-probabilities of the next character."""
        # Unpack the memory
        enc_output, mask = memory
        # Feed only the last character
        logits, hidden = self.dec.step(tokens[:, -1], state, enc_output, mask)
        # Log-probabilities for beam search
        return F.log_softmax(logits.float(), dim=-1), hidden

    def reorder_state(self, state: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Reorder the hidden state to follow the kept beams."""
        # Pick the states of the kept beams
        return state.index_select(1, indices)
