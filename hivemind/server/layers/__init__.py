import torch

from hivemind.server.layers.common import FeedforwardBlock, TransformerEncoderLayer, NopExpert
from hivemind.server.layers.dropout import DeterministicDropout, DeterministicDropoutNetwork
from hivemind.server.layers.gpt import GPTInitialBlock, GPTIntermediateBlock, GPTFinalBlock

name_to_block = {'ffn': lambda hid_dim: FeedforwardBlock(hid_dim),
                 'transformer': lambda hid_dim: TransformerEncoderLayer(hid_dim, dim_feedforward=4 * hid_dim, nhead=16),
                 'nop': lambda hid_dim: NopExpert(hid_dim),
                 'det_dropout': lambda hid_dim: DeterministicDropoutNetwork(hid_dim, dropout_prob=0.2),

                 'gpt_initial_128tokens_voc32k': lambda hid_dim: GPTInitialBlock(
                     d_model=hid_dim, nhead=hid_dim // 128, num_layers=1, vocab_size=32_000),
                 'gpt_intermediate_128tokens_voc32k': lambda hid_dim: GPTIntermediateBlock(
                     d_model=hid_dim, nhead=hid_dim // 128, num_layers=1),
                 'gpt_final_128tokens_voc32k': lambda hid_dim: GPTFinalBlock(
                     d_model=hid_dim, nhead=hid_dim // 128, num_layers=0, vocab_size=32_000),
                 }

name_to_input = {'ffn': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'transformer': lambda batch_size, hid_dim:
                 (torch.empty((batch_size, 128, hid_dim)), torch.empty((batch_size, hid_dim), dtype=torch.bool)),
                 'nop': lambda batch_size, hid_dim: torch.empty((batch_size, hid_dim)),
                 'det_dropout': lambda batch_size, hid_dim:
                 (torch.empty((batch_size, hid_dim)), torch.randint(0, 1, (batch_size, hid_dim))),

                 'gpt_initial_128tokens_voc32k': lambda batch_size, hid_dim: torch.randint(0, 10, size=(batch_size, 128)),
                 'gpt_intermediate_128tokens_voc32k': lambda batch_size, hid_dim: torch.randn(batch_size, 128, hid_dim),
                 'gpt_final_128tokens_voc32k': lambda batch_size, hid_dim: (
                     torch.randn(batch_size, 128, hid_dim), torch.randint(0, 10, size=(batch_size, 128))),
                 }
