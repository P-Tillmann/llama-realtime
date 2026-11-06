#!/usr/bin/env python3
"""Convert a Smart Turn v3.x ONNX file (pipecat-ai/smart-turn-v3 on Hugging Face, fp32 or int8) to GGUF.

    pip install onnx gguf numpy
    python convert_smart_turn.py smart-turn-v3.2-gpu.onnx smart-turn-v3.2.gguf

The exporter gives the attention and FFN projection weights anonymous names and the int8 file wraps
weights in DequantizeLinear, so weights are found by graph structure and by the names of their biases.
"""
import argparse
import re
import numpy as np
import onnx  # type: ignore  # pip install onnx
from onnx import numpy_helper  # type: ignore
import gguf

N_LAYER, D, N_HEAD, D_FF, N_MEL, N_CTX = 4, 384, 6, 1536, 80, 400


def slaney_mel_filters(sr=16000, n_fft=400, n_mels=80, fmin=0.0, fmax=8000.0):
    """Mel filterbank as built by HF WhisperFeatureExtractor (slaney scale + slaney norm). Shape (n_fft//2+1, n_mels)."""
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = 3.0 * f / 200.0
        log_reg = f >= 1000.0
        return np.where(log_reg, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), mel)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = 200.0 * m / 3.0
        return np.where(m >= 15.0, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), f)

    fft_freqs = np.linspace(0, sr // 2, n_fft // 2 + 1)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    filt_freqs = mel_to_hz(mel_pts)
    fdiff = np.diff(filt_freqs)
    slopes = filt_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (filt_freqs[2:n_mels + 2] - filt_freqs[:n_mels])
    return (fb * enorm[None, :]).astype(np.float32)


class Graph:
    def __init__(self, path):
        m = onnx.load(path)
        self.g = m.graph
        self.init = {i.name: numpy_helper.to_array(i) for i in self.g.initializer}
        self.producer = {o: n for n in self.g.node for o in n.output}
        self.consumers = {}
        for n in self.g.node:
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
        for n in self.g.node:  # Constant nodes count as initializers
            if n.op_type == "Constant":
                self.init[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)

    def const(self, name):
        """Resolve a tensor name to a constant numpy array (follows DequantizeLinear / Transpose / Identity / Cast), or None."""
        if name in self.init:
            return self.init[name]
        n = self.producer.get(name)
        if n is None:
            return None
        if n.op_type == "DequantizeLinear":
            x = self.const(n.input[0])
            s = self.const(n.input[1])
            zp = self.const(n.input[2]) if len(n.input) > 2 and n.input[2] else None
            if x is None or s is None:
                return None
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            x = x.astype(np.float32)
            if zp is not None:
                zp = zp.astype(np.float32)
            if s.ndim == 1 and s.size > 1:  # per-channel
                shape = [1] * x.ndim
                shape[axis] = -1
                s = s.reshape(shape)
                zp = zp.reshape(shape) if zp is not None else None
            return (x - (zp if zp is not None else 0.0)) * s
        if n.op_type == "Transpose":
            x = self.const(n.input[0])
            perm = next((list(a.ints) for a in n.attribute if a.name == "perm"), None)
            return None if x is None else np.transpose(x, perm)
        if n.op_type in ("Identity", "Cast"):
            return self.const(n.input[0])
        return None

    def root(self, name):
        """Name of the initializer a constant tensor ultimately comes from (through DequantizeLinear / Transpose / Identity / Cast)."""
        while name not in self.init:
            n = self.producer.get(name)
            if n is None or n.op_type not in ("DequantizeLinear", "Transpose", "Identity", "Cast"):
                return None
            name = n.input[0]
        name = name[:-len("_quantized")] if name.endswith("_quantized") else name
        # upstream exports through a wrapper module named "inner"
        return name[len("inner."):] if name.startswith("inner.") else name

    def skip_qdq(self, name):
        """Follow an activation through QuantizeLinear -> DequantizeLinear pairs (int8 QDQ graphs)."""
        while True:
            cs = self.consumers.get(name, [])
            if len(cs) == 1 and cs[0].op_type == "QuantizeLinear":
                dq = [c for c in self.consumers.get(cs[0].output[0], []) if c.op_type == "DequantizeLinear"]
                if len(dq) == 1:
                    name = dq[0].output[0]
                    continue
            return name

    def nodes(self, op):
        return [n for n in self.g.node if n.op_type == op]

    def bias_after(self, out_name):
        """If the output of a MatMul feeds an Add with a constant operand, return that constant."""
        out_name = self.skip_qdq(out_name)
        for c in self.consumers.get(out_name, []):
            if c.op_type == "Add":
                other = c.input[1] if c.input[0] == out_name else c.input[0]
                b = self.const(other)
                if b is not None:
                    return b, self.root(other)
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("out")
    ap.add_argument("--type", choices=["f32", "f16", "q8_0"], default="f32",
                    help="storage type for the large 2D projection weights")
    args = ap.parse_args()

    G = Graph(args.onnx)
    T = {}  # gguf name -> float32 array (torch layout)

    # Conv1d stem
    convs = G.nodes("Conv")
    assert len(convs) == 2, f"expected 2 Conv nodes, got {len(convs)}"
    for c in convs:
        i = {"encoder.conv1.weight": 1, "encoder.conv2.weight": 2}[G.root(c.input[1])]
        T[f"enc.conv{i}.weight"] = G.const(c.input[1])
        T[f"enc.conv{i}.bias"] = G.const(c.input[2])
    assert T["enc.conv1.weight"].shape == (D, N_MEL, 3) and T["enc.conv2.weight"].shape == (D, D, 3)

    # Positional embedding: the only [400, 384] constant
    pos = [n for n in list(G.init) if G.init[n].shape == (N_CTX, D)]
    assert len(pos) == 1, pos
    T["enc.pos_emb"] = G.init[pos[0]]

    # MatMuls with constant weights. Node order is not reliable (ORT's optimizer reorders), so identify each
    # projection by the name of the bias added after it; k_proj has no bias and is the bias-free sibling of q_proj.
    bias_map = {"self_attn.q_proj": "attn.q", "self_attn.v_proj": "attn.v", "self_attn.out_proj": "attn.o", "fc1": "fc1", "fc2": "fc2"}
    mms = [(n, G.const(n.input[1])) for n in G.nodes("MatMul")]
    mms = [(n, w) for n, w in mms if w is not None]
    assert len(mms) == 6 * N_LAYER + 2, f"expected {6*N_LAYER+2} weight MatMuls, got {len(mms)}"
    unbiased = []
    for n, w in mms:
        b, bname = G.bias_after(n.output[0])
        if b is None:
            unbiased.append((n, w))
            continue
        m = re.match(r"encoder\.layers\.(\d+)\.(.+)\.bias", bname or "")
        if m and m.group(2) in bias_map:
            key = f"enc.{m.group(1)}.{bias_map[m.group(2)]}"
        elif bname and bname.startswith("pool_attention.") and bname.endswith(".bias"):
            key = "pool." + bname.split(".")[1]
        else:
            raise SystemExit(f"cannot identify MatMul {n.name} (bias {bname})")
        assert key + ".weight" not in T, key
        T[key + ".weight"] = np.ascontiguousarray(w.T)  # ONNX stores [in, out]; ggml wants torch [out, in]
        T[key + ".bias"] = b
    q_inputs = {}
    for n, w in mms:
        b, bname = G.bias_after(n.output[0])
        m = re.match(r"encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias", bname or "")
        if m:
            q_inputs[n.input[0]] = int(m.group(1))
    assert len(unbiased) == N_LAYER, f"expected {N_LAYER} bias-free (k_proj) MatMuls, got {len(unbiased)}"
    for n, w in unbiased:
        l = q_inputs[n.input[0]]  # same input tensor as that layer's q_proj
        T[f"enc.{l}.attn.k.weight"] = np.ascontiguousarray(w.T)
    for l in range(N_LAYER):
        for s in ("attn.q", "attn.k", "attn.v", "attn.o", "fc1", "fc2"):
            assert f"enc.{l}.{s}.weight" in T, f"missing enc.{l}.{s}"
        assert T[f"enc.{l}.fc1.weight"].shape == (D_FF, D)
    assert T["pool.0.weight"].shape == (256, D) and T["pool.2.weight"].shape == (1, 256)

    # LayerNorms, identified by the (named) gamma initializer
    ln_map = {"self_attn_layer_norm": "attn_ln", "final_layer_norm": "ffn_ln"}
    lns = G.nodes("LayerNormalization")
    assert len(lns) == 2 * N_LAYER + 2, f"expected {2*N_LAYER+2} LayerNorm, got {len(lns)}"
    for n in lns:
        r = G.root(n.input[1])
        m = re.match(r"encoder\.layers\.(\d+)\.(\w+)\.weight", r or "")
        if m:
            key = f"enc.{m.group(1)}.{ln_map[m.group(2)]}"
        elif r == "encoder.layer_norm.weight":
            key = "enc.ln_post"
        elif r == "classifier.1.weight":
            key = "cls.ln"
        else:
            raise SystemExit(f"cannot identify LayerNorm {n.name} ({r})")
        T[key + ".weight"] = G.const(n.input[1])
        T[key + ".bias"] = G.const(n.input[2])

    # Classifier Gemms: 384->256, 256->64, 64->1
    gemms = G.nodes("Gemm")
    assert len(gemms) == 3, len(gemms)
    for n in gemms:
        r = G.root(n.input[1]) or ""
        m = re.match(r"classifier\.(\d)\.weight", r)
        assert m, f"cannot identify Gemm {n.name} ({r})"
        name = "cls." + m.group(1)
        w = G.const(n.input[1])
        transB = next((a.i for a in n.attribute if a.name == "transB"), 0)
        T[name + ".weight"] = w if transB else w.T
        T[name + ".bias"] = G.const(n.input[2])
    assert T["cls.0.weight"].shape == (256, D) and T["cls.6.weight"].shape == (1, 64)

    T["mel_filters"] = slaney_mel_filters().T.copy()  # (80, 201)

    w = gguf.GGUFWriter(args.out, "smart-turn")
    w.add_string("general.name", "smart-turn-v3")
    w.add_uint32("smart_turn.n_layer", N_LAYER)
    w.add_uint32("smart_turn.d_model", D)
    w.add_uint32("smart_turn.n_head", N_HEAD)
    w.add_uint32("smart_turn.n_ctx", N_CTX)
    w.add_uint32("smart_turn.n_mel", N_MEL)
    big = {f"enc.{l}.{s}.weight" for l in range(N_LAYER) for s in ("attn.q", "attn.k", "attn.v", "attn.o", "fc1", "fc2")}
    nbytes = 0
    for name, arr in T.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        if name in big and args.type == "f16":
            arr = arr.astype(np.float16)
            w.add_tensor(name, arr)
        elif name in big and args.type == "q8_0":
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, q, raw_shape=q.shape, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            arr = q
        else:
            w.add_tensor(name, arr)
        nbytes += arr.nbytes
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {len(T)} tensors, {nbytes/1e6:.2f} MB of tensor data ({args.type} projections)")


if __name__ == "__main__":
    main()
