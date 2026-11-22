"""Character-level Transformer encoder-decoder."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets.vocab import PAD


class PosEnc(nn.Module):
    """Adds position information to the character embeddings."""
    def __init__(self, dimension: int, max_length: int = 128) -> None:
        super().__init__()
        # Table of position codes
        encoding = torch.zeros(max_length, dimension)
        # Position numbers 0, 1, 2, ...
        position = torch.arange(max_length).unsqueeze(1).float()
        # A different wave length for each dimension
        divisor = torch.exp(
            torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension)
        )
        # Even dimensions use sine
        encoding[:, 0::2] = torch.sin(position * divisor)
        # Odd dimensions use cosine
        encoding[:, 1::2] = torch.cos(position * divisor)
        # Store it with the model, but do not train it
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the position codes to the input."""
        # Add the codes, cut to the input length
        return x + self.pe[:, :x.size(1)].to(x.dtype)


class CharTransformer(nn.Module):
    """Character Transformer for transliteration."""
    def __init__(
        self,
        source_vocab: int,
        target_vocab: int,
        dimension: int = 256,
        heads: int = 4,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        feedforward: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Width of the model
        self.dimension = dimension
        # Embeddings for the Roman input
        self.src_emb = nn.Embedding(source_vocab, dimension, padding_idx=PAD)
        # Embeddings for the Devanagari output
        self.tgt_emb = nn.Embedding(target_vocab, dimension, padding_idx=PAD)
        # Position codes
        self.pos = PosEnc(dimension)
        # Dropout against over-fitting
        self.drop = nn.Dropout(dropout)
        # One encoder layer (normalise first)
        encoder_layer = nn.TransformerEncoderLayer(
            dimension,
            heads,
            feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        # One decoder layer (normalise first)
        decoder_layer = nn.TransformerDecoderLayer(
            dimension,
            heads,
            feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        # Stack of encoder layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            encoder_layers,
            norm=nn.LayerNorm(dimension),
            enable_nested_tensor=False,
        )
        # Stack of decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            decoder_layers,
            norm=nn.LayerNorm(dimension),
        )
        # Hidden state -> character scores
        self.out = nn.Linear(dimension, target_vocab, bias=False)
        # Xavier start values for weight matrices
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        # Small random embeddings
        nn.init.normal_(self.src_emb.weight, 0.0, dimension ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, 0.0, dimension ** -0.5)
        # Share weights between output and target embeddings
        self.out.weight = self.tgt_emb.weight
        # Keep the padding embedding at zero
        with torch.no_grad():
            self.src_emb.weight[PAD].zero_()
            self.tgt_emb.weight[PAD].zero_()

    def _embed(self, embedding: nn.Embedding, tokens: torch.Tensor) -> torch.Tensor:
        """Embed, scale, add positions, then apply dropout."""
        return self.drop(self.pos(embedding(tokens) * math.sqrt(self.dimension)))

    def _encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the encoder; also return the padding mask."""
        # True at padding positions
        padding = src == PAD
        # Encoder output for every input character
        memory = self.encoder(
            self._embed(self.src_emb, src), src_key_padding_mask=padding
        )
        return memory, padding

    def _decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        source_padding: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder over the characters so far."""
        # Number of output characters so far
        length = tgt_in.size(1)
        # Mask so a position cannot see later ones
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tgt_in.device), diagonal=1
        )
        # Decode while attending to the encoder output
        hidden = self.decoder(
            self._embed(self.tgt_emb, tgt_in),
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=source_padding,
        )
        # Scores for each character
        return self.out(hidden)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Training pass: scores for every output position."""
        # Encode the input
        memory, padding = self._encode(src)
        return self._decode(tgt_in, memory, padding)

    def encode(
        self, src: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], None]:
        """Encode once for beam search (no extra state needed)."""
        # Encode the input
        memory, padding = self._encode(src)
        # Memory for decoding; no state to carry
        return (memory, padding), None

    def step(
        self,
        tokens: torch.Tensor,
        state: None,
        memory: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, None]:
        """One beam-search step: log-probabilities of the next character."""
        # Unpack the memory
        encoded, padding = memory
        # Decode all characters so far; keep the last position
        logits = self._decode(tokens, encoded, padding)[:, -1]
        # Log-probabilities for beam search
        return F.log_softmax(logits.float(), dim=-1), None

    def reorder_state(self, state: None, indices: torch.Tensor) -> None:
        """Nothing to reorder: this model keeps no hidden state."""
        # No state to reorder
        return None
