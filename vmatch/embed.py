"""Sentence embeddings with all-MiniLM-L6-v2 (ONNX), no PyTorch needed."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import BertWordPieceTokenizer

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


class Embedder:
    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 256):
        self.tok = BertWordPieceTokenizer(str(model_dir / "vocab.txt"), lowercase=True)
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok.encode_batch(list(texts[i:i + batch_size]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]                      # (b, t, 384)
            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)    # mean pooling
            out.append(v / np.linalg.norm(v, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)
