import torch
import torch.nn as nn
from .seq2seq import Encoder, Decoder, BahdanauAttention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention, raw_params):
        super().__init__()

        self.device = raw_params

        self.encoder = encoder
        self.decoder = decoder
      #  self.device = str(params["device"])

    def forward(self, encoder_input, decoder_input, teacher_forcing=False):
        batch_size = decoder_input.size(0)
        trg_len = decoder_input.size(1)

        outputs = torch.zeros(batch_size, trg_len - 1, self.decoder.output_dim).to(self.device)
        enc_output, hidden = self.encoder(encoder_input)

        dec_input = decoder_input[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(enc_output, dec_input, hidden)
            outputs[:, t - 1] = output
            if teacher_forcing == True:
                dec_input = decoder_input[:, t]
            else:
                dec_input = output

        return outputs

