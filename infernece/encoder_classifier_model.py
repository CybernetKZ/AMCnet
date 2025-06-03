import logging
import math
from typing import List, Tuple, Dict
import os

import k2
import kaldifeat
import onnxruntime as ort
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
import numpy as np
from classifier_model import AudioClassifier


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

        self.session_opts = session_opts

        self.init_encoder(encoder_model_filename)
        self.init_decoder(decoder_model_filename)
        self.init_joiner(joiner_model_filename)

    def init_encoder(self, encoder_model_filename: str):
        self.encoder = ort.InferenceSession(
            encoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_decoder(self, decoder_model_filename: str):
        self.decoder = ort.InferenceSession(
            decoder_model_filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
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
            providers=["CPUExecutionProvider"],
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
        out = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        return torch.from_numpy(out[0]), torch.from_numpy(out[1])

    def run_decoder(self, decoder_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
          decoder_input:
            A 2-D tensor of shape (N, context_size)
        Returns:
          Return a 2-D tensor of shape (N, joiner_dim)
        """
        out = self.decoder.run(
            [self.decoder.get_outputs()[0].name],
            {self.decoder.get_inputs()[0].name: decoder_input.numpy()},
        )[0]

        return torch.from_numpy(out)

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
        out = self.joiner.run(
            [self.joiner.get_outputs()[0].name],
            {
                self.joiner.get_inputs()[0].name: encoder_out.numpy(),
                self.joiner.get_inputs()[1].name: decoder_out.numpy(),
            },
        )[0]

        return torch.from_numpy(out)


def read_sound_files(
    filenames: List[str], expected_sample_rate: float
) -> List[torch.Tensor]:
    """Read a list of sound files into a list 1-D float32 torch tensors.
    Args:
      filenames:
        A list of sound filenames.
      expected_sample_rate:
        The expected sample rate of the sound files.
    Returns:
      Return a list of 1-D float32 torch tensors.
    """
    ans = []
    for f in filenames:
        wave, sample_rate = torchaudio.load(f)
        
        
        

        if sample_rate != expected_sample_rate:
            print(f"Resampling {f} from {sample_rate} Hz to {expected_sample_rate} Hz")
            resampler = T.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
            wave = resampler(wave)

        
        ans.append(wave[0])
    return ans

def batch_inference(
    encoder_model_filename: str,
    decoder_model_filename: str,
    joiner_model_filename: str,
    classifier_model_path: str,
    audio_files: List[str],
    sample_rate: int = 16000,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Dict]:
    """
    Perform batch inference on audio files to get answering machine predictions.
    
    Args:
        encoder_model_filename: Path to the encoder ONNX model
        decoder_model_filename: Path to the decoder ONNX model
        joiner_model_filename: Path to the joiner ONNX model
        classifier_model_path: Path to the trained classifier model checkpoint
        audio_files: List of paths to audio files
        sample_rate: Expected sample rate of audio files
        batch_size: Number of files to process at once
        device: Device to run the classifier on ('cuda' or 'cpu')
        
    Returns:
        Dictionary mapping audio file paths to their predictions containing:
        - 'prediction': 0 (not machine) or 1 (answering machine)
        - 'probability': probability of being an answering machine
        - 'label': 'not_machine' or 'answering_machine'
    """
    
    model = OnnxModel(
        encoder_model_filename=encoder_model_filename,
        decoder_model_filename=decoder_model_filename,
        joiner_model_filename=joiner_model_filename,
    )

    
    checkpoint = torch.load(classifier_model_path, map_location=device)
    classifier = AudioClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        num_classes=2
    )
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()

    
    opts = kaldifeat.FbankOptions()
    opts.device = "cpu"
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(opts)

    
    results = {}
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        
        
        waves = read_sound_files(
            filenames=batch_files,
            expected_sample_rate=sample_rate,
        )
        
        
        features = fbank(waves)
        feature_lengths = [f.size(0) for f in features]
        
        
        features = pad_sequence(
            features,
            batch_first=True,
            padding_value=math.log(1e-10),
        )
        
        feature_lengths = torch.tensor(feature_lengths, dtype=torch.int64)
        
        
        encoder_out, encoder_out_lens = model.run_encoder(features, feature_lengths)
        
        
        for j, file_path in enumerate(batch_files):
            
            valid_length = encoder_out_lens[j]
            file_embedding = encoder_out[j, :valid_length].mean(dim=0)
            
            
            embedding_tensor = torch.from_numpy(file_embedding.numpy()).unsqueeze(0).to(device)

            
            with torch.no_grad():
                outputs = classifier(embedding_tensor)
                probs = classifier.predict_proba(embedding_tensor)
                pred = outputs.argmax(dim=1).item()
                prob = probs[0][1].item()  

            
            results[file_path] = {
                'prediction': pred,
                'probability': prob,
                'label': 'answering_machine' if pred == 1 else 'not_machine'
            }
            
    return results

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    audio_files = [
        "/home/mahmoud/icefall/egs/eg_vo_me_syth/ASR/real_test_data/pool_test_1_EG_Aloula/1c1f0fe87564c4a3.wav"
    ]

    results = batch_inference(
        encoder_model_filename="./encoder_model/encoder-epoch-28-avg-13.onnx",
        decoder_model_filename="./encoder_model/decoder-epoch-28-avg-13.onnx",
        joiner_model_filename="./encoder_model/joiner-epoch-28-avg-13.onnx",
        classifier_model_path="./models/best_model.pt",
        audio_files=audio_files,
        sample_rate=16000
    )

    for file_path, result in results.items():
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Prediction: {result['label']}")
        print(f"Probability: {result['probability']:.4f}")
        print("---")
