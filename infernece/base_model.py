import logging
from typing import List, Tuple

import onnxruntime as ort
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T


class OnnxModel:
    def __init__(
        self,
        encoder_model_filename: str,
        decoder_model_filename: str,
        joiner_model_filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        # Enable CUDA execution provider
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=self.providers,
        )

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=self.providers,
        )

        decoder_meta = self.decoder.get_modelmeta().custom_metadata_map
        self.context_size = int(decoder_meta["context_size"])
        self.vocab_size = int(decoder_meta["vocab_size"])

        logging.info(f"context_size: {self.context_size}")
        logging.info(f"vocab_size: {self.vocab_size}")

    def init_joiner(self, joiner_model_filename: str):
        self.joiner = ort.InferenceSession(
            joiner_model_filename,
            sess_options=self.session_opts,
            providers=self.providers,
        )

        joiner_meta = self.joiner.get_modelmeta().custom_metadata_map
        self.joiner_dim = int(joiner_meta["joiner_dim"])

        logging.info(f"joiner_dim: {self.joiner_dim}")

    def run_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C)
          x_lens:
            A 2-D tensor of shape (N,). Its dtype is torch.int64
        Returns:
          Return a tuple containing:
            - encoder_out, its shape is (N, T', joiner_dim)
            - encoder_out_lens, its shape is (N,)
        """
        # Move tensors to CPU for ONNX inference
        x_np = x.cpu().numpy()
        x_lens_np = x_lens.cpu().numpy()
        
        out = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x_np,
                self.encoder.get_inputs()[1].name: x_lens_np,
            },
        )
        # Move results back to GPU
        return torch.from_numpy(out[0]).cuda(), torch.from_numpy(out[1]).cuda()

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A 2-D tensor of shape (N, context_size)
        Returns:
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        # Move tensor to CPU for ONNX inference
        decoder_input_np = decoder_input.cpu().numpy()
        
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input_np},
        )[0]
        
        # Move result back to GPU
        return torch.from_numpy(out).cuda()

    def run_joiner(
        self, encoder_out: torch.Tensor, decoder_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            A 2-D tensor of shape (N, joiner_dim)
          decoder_out:
            A 2-D tensor of shape (N, joiner_dim)
        Returns:
          Return a 2-D tensor of shape (N, vocab_size)
        """
        # Move tensors to CPU for ONNX inference
        encoder_out_np = encoder_out.cpu().numpy()
        decoder_out_np = decoder_out.cpu().numpy()
        
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out_np,
                self.joiner.get_inputs()[1].name: decoder_out_np,
            },
        )[0]
        
        # Move result back to GPU
        return torch.from_numpy(out).cuda()
