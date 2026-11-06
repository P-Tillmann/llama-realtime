#!/usr/bin/env python3
"""Convert the 16 kHz Silero VAD (pip package silero-vad) to GGUF.

The weights come from the default model of the package (load_silero_vad(), TorchScript); the
silero_vad_16k.safetensors file of the package holds different weights and gives other probabilities.

    pip install silero-vad torch gguf
    python convert_silero_vad.py silero-vad.gguf
"""

import argparse

import gguf
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output GGUF file")
    args = ap.parse_args()

    import silero_vad  # type: ignore  # pip install silero-vad
    version = getattr(silero_vad, "__version__", "unknown")
    jit = silero_vad.load_silero_vad()  # TorchScript, holds the 16 kHz and the 8 kHz model
    names = {
        "stft.forward_basis_buffer": "stft_conv.weight",
        "decoder.rnn.weight_ih": "lstm_cell.weight_ih", "decoder.rnn.weight_hh": "lstm_cell.weight_hh",
        "decoder.rnn.bias_ih": "lstm_cell.bias_ih", "decoder.rnn.bias_hh": "lstm_cell.bias_hh",
        "decoder.decoder.2.weight": "final_conv.weight", "decoder.decoder.2.bias": "final_conv.bias",
    }
    for i in range(4):
        names[f"encoder.{i}.reparam_conv.weight"] = f"conv{i + 1}.weight"
        names[f"encoder.{i}.reparam_conv.bias"] = f"conv{i + 1}.bias"
    sd = {names[k]: v.detach().numpy() for k, v in jit._model.state_dict().items() if k in names}
    expected = {
        "stft_conv.weight": (258, 1, 256),
        "conv1.weight": (128, 129, 3), "conv1.bias": (128,),
        "conv2.weight": (64, 128, 3),  "conv2.bias": (64,),
        "conv3.weight": (64, 64, 3),   "conv3.bias": (64,),
        "conv4.weight": (128, 64, 3),  "conv4.bias": (128,),
        "lstm_cell.weight_ih": (512, 128), "lstm_cell.weight_hh": (512, 128),
        "lstm_cell.bias_ih": (512,), "lstm_cell.bias_hh": (512,),
        "final_conv.weight": (1, 128, 1), "final_conv.bias": (1,),
    }
    for k, shape in expected.items():
        assert k in sd and tuple(sd[k].shape) == shape, f"{k}: expected {shape}, got {sd[k].shape if k in sd else 'missing'}"

    w = gguf.GGUFWriter(args.out, "silero-vad")
    w.add_name("Silero VAD")
    w.add_string("silero-vad.version", version)
    w.add_uint32("silero-vad.sample_rate", 16000)
    w.add_uint32("silero-vad.frame_size", 512)
    w.add_uint32("silero-vad.context_size", 64)
    for k in expected:
        w.add_tensor(k, np.ascontiguousarray(sd[k].astype(np.float32)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out} (silero-vad {version})")


if __name__ == "__main__":
    main()
