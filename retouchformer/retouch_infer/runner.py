# retouch_infer/runner.py
import os
import glob
import traceback
import torch

from .loader import load_model
from .preprocess import read_image_to_tensor
from .postprocess import save_tensor_normed
from .tiling import pad_to_multiple, unpad


class RetouchFormerRunner:
    """
    仅推理用的封装：
    - 默认把输入 resize 到 512x512（与原 img_retouching.py / wildDataset --size 512 对齐）
    - 可选 multiple：不 resize，而是 pad 到倍数（注意：该模型多尺度对齐敏感，multiple 方案不一定稳）
    - debug=True 时会打印：设备信息、输入范围/shape、encoder/conv_* 输出 shape、pred shape、保存路径等
    """

    def __init__(
        self,
        model_name: str = "RetouchFormer",
        ckpt_dir: str = "release_model",
        epoch: str = "best",
        device: str = None,
        multiple: int = None,
        size: int = 512,
        debug: bool = False,
    ):
        # 1) 加载模型与权重
        print(f"[runner.__init__] loading model='{model_name}', ckpt_dir='{ckpt_dir}', epoch='{epoch}'")
        self.model, self.device = load_model(model_name, ckpt_dir, epoch, device)
        print(f"[runner.__init__] model loaded on device: {self.device}")

        # 2) 输入尺寸策略：size 和 multiple 二选一（默认使用 size=512）
        self.multiple = multiple
        self.size = size
        self.debug = debug

        if self.multiple is not None and self.size is not None:
            print("[runner.__init__] both 'multiple' and 'size' provided; 'size' takes precedence.")
            self.multiple = None

        print(f"[runner.__init__] infer config -> size={self.size}, multiple={self.multiple}, debug={self.debug}")

        # 3) 注册 forward hooks（仅调试打印，不修改计算图）
        self._hooks = []
        self._register_hooks()

    # -----------------------------
    # Hook 注册 / 移除
    # -----------------------------
    def _register_hooks(self):
        """
        注册 forward hooks：打印 encoder / conv_512 / conv_256 / conv_128 输出形状。
        """
        def _enc_hook(module, inp, out):
            try:
                if isinstance(out, (list, tuple)):
                    shapes = [tuple(t.shape) for t in out]
                    print(f"[hook.encoder] decoder_noise shapes: {shapes}")
                else:
                    print(f"[hook.encoder] out shape: {tuple(out.shape)}")
            except Exception as e:
                print(f"[hook.encoder] print failed: {e}")

        def _make_hook(name):
            def hook(module, inp, out):
                try:
                    shape = tuple(out.shape)
                except Exception:
                    shape = "N/A"
                print(f"[hook.{name}] out shape={shape}")
            return hook

        # encoder
        try:
            if hasattr(self.model, "encoder"):
                h = self.model.encoder.register_forward_hook(_enc_hook)
                self._hooks.append(h)
                if self.debug:
                    print("[hook.register] encoder hook registered.")
            else:
                print("[hook.register] encoder not found; skip.")
        except Exception as e:
            print(f"[hook.register] encoder hook failed: {e}")

        # conv_512 / conv_256 / conv_128
        for name in ("conv_512", "conv_256", "conv_128"):
            if hasattr(self.model, name):
                try:
                    m = getattr(self.model, name)
                    h = m.register_forward_hook(_make_hook(name))
                    self._hooks.append(h)
                    if self.debug:
                        print(f"[hook.register] {name} hook registered.")
                except Exception as e:
                    print(f"[hook.register] {name} hook failed: {e}")
            else:
                if self.debug:
                    print(f"[hook.register] {name} not found; skip.")

    def _remove_hooks(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        if self.debug:
            print("[hook.remove] all hooks removed.")

    # -----------------------------
    # 单张图片推理
    # -----------------------------
    @torch.no_grad()
    def infer_path(self, in_path: str, out_path: str):
        print(f"[infer_path] input='{in_path}' -> output='{out_path}'")
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"[infer_path] input path does not exist: {in_path}")

        try:
            # 1) 读图 + 归一化 + 尺寸策略
            if self.size is not None:
                print(f"[infer_path] resize input to size={self.size}")
            else:
                print(f"[infer_path] keep original size; multiple pad={self.multiple}")

            t, _ = read_image_to_tensor(in_path, resize_to=self.size, debug=self.debug)
            t = t.to(self.device)

            # === 新增：让输入 dtype 与模型一致 + 打印定位 ===
            try:
                model_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float32
            if t.dtype != model_dtype:
                if self.debug:
                    print(f"[infer_path] cast input from {t.dtype} to model dtype {model_dtype}")
                t = t.to(dtype=model_dtype)
            else:
                if self.debug:
                    print(f"[infer_path] input dtype matches model dtype: {t.dtype}")

            ph = pw = 0
            if self.size is None and self.multiple:
                t, ph, pw = pad_to_multiple(t, self.multiple)
                print(f"[infer_path] after pad: shape={tuple(t.shape)}, ph={ph}, pw={pw}")

            # 2) 前向
            print("[infer_path] forward ...")
            out = self.model(t)

            # 统一解析返回（兼容 tuple / dict / tensor）
            if isinstance(out, (list, tuple)):
                pred = out[0]
                extra = out[1] if len(out) > 1 else None
            elif isinstance(out, dict):
                pred = out.get("pred", None) or out.get("result", None)
                extra = out
            else:
                pred = out
                extra = None

            if pred is None:
                raise RuntimeError("[infer_path] model output parsing failed: pred is None")

            if self.size is None and self.multiple:
                pred = unpad(pred, ph, pw)
                if self.debug:
                    print(f"[infer_path] after unpad: pred shape={tuple(pred.shape)}")

            if self.debug:
                try:
                    print(f"[infer_path] pred shape={tuple(pred.shape)}")
                except Exception:
                    pass
                if isinstance(extra, (list, tuple)):
                    print(f"[infer_path] extra list length={len(extra)}")
                elif isinstance(extra, dict):
                    print(f"[infer_path] extra dict keys={list(extra.keys())}")
                else:
                    print(f"[infer_path] extra type={type(extra).__name__}")

            # 3) 保存（与原脚本保持一致的 [-1,1] → save_image(normalize=True, value_range=(-1,1))）
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_tensor_normed(pred, out_path)
            print(f"[infer_path] saved: {out_path}")

            return out_path

        except Exception as e:
            print("[infer_path] EXCEPTION raised during inference:")
            traceback.print_exc()
            raise e
        finally:
            # 单图结束即移除 hook，避免多次注册叠加
            self._remove_hooks()

    # -----------------------------
    # 文件夹批量推理
    # -----------------------------
    @torch.no_grad()
    def infer_folder(self, in_dir: str, out_dir: str):
        print(f"[infer_folder] input_dir='{in_dir}' -> output_dir='{out_dir}'")
        if not os.path.isdir(in_dir):
            raise NotADirectoryError(f"[infer_folder] input_dir is not a directory: {in_dir}")

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths = [p for e in exts for p in glob.glob(os.path.join(in_dir, e))]
        print(f"[infer_folder] found {len(paths)} images.")
        os.makedirs(out_dir, exist_ok=True)

        results = []
        for idx, p in enumerate(sorted(paths), 1):
            name = os.path.splitext(os.path.basename(p))[0]
            out_p = os.path.join(out_dir, f"{name}_out.png")
            print(f"[infer_folder] [{idx}/{len(paths)}] {p} -> {out_p}")
            try:
                self.infer_path(p, out_p)
                results.append(out_p)
            except Exception as e:
                print(f"[infer_folder] failed on '{p}': {e}")
        print(f"[infer_folder] done. succeeded={len(results)} / total={len(paths)}")
        return results
