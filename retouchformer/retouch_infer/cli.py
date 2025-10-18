import argparse
from .runner import RetouchFormerRunner

def main():
    ap = argparse.ArgumentParser("RetouchFormer Inference")
    ap.add_argument("--model", default="RetouchFormer")
    ap.add_argument("--ckpt",  default="release_model")
    ap.add_argument("--epoch", default="best")
    ap.add_argument("--input", required=True, help="单图路径或文件夹")
    ap.add_argument("--output", default="results")
    ap.add_argument("--multiple", type=int, default=None, help="可选：输入尺寸倍数(8/16/32)，与 size 二选一")
    ap.add_argument("--size", type=int, default=512, help="推理时将输入resize到此尺寸(默认512，与multiple二选一)")
    ap.add_argument("--debug", action="store_true", help="打印中间形状与范围")
    args = ap.parse_args()

    runner = RetouchFormerRunner(model_name=args.model,
                                 ckpt_dir=args.ckpt,
                                 epoch=args.epoch,
                                 multiple=args.multiple,
                                 size=args.size,
                                 debug=args.debug)

    import os
    if os.path.isdir(args.input):
        runner.infer_folder(args.input, args.output)
    else:
        out = args.output
        if os.path.isdir(out):
            import os.path as osp
            name = osp.splitext(osp.basename(args.input))[0]
            out = osp.join(args.output, f"{name}_out.png")
        runner.infer_path(args.input, out)

if __name__ == "__main__":
    main()
