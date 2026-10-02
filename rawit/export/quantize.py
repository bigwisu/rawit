# Copyright 2025 Wisu Suntoyo — Apache-2.0
"""INT8 dynamic quantization for the Rawit ONNX model.

Usage:
    python -m rawit.export.quantize \\
        --input rawit.onnx \\
        --output rawit-int8.onnx
"""

import argparse
import logging

_log = logging.getLogger("rawit.quantize")


def quantize(input_path: str, output_path: str):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime is required for quantization. "
            "Install with: pip install rawit[onnx]"
        )

    _log.info("Quantizing %s → %s (INT8 dynamic)", input_path, output_path)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        optimize_model=True,
    )
    _log.info("INT8 model saved: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="INT8 dynamic quantization for Rawit ONNX")
    parser.add_argument("--input", required=True, help="Input ONNX model path.")
    parser.add_argument("--output", required=True, help="Output INT8 ONNX model path.")
    args = parser.parse_args()
    quantize(args.input, args.output)


if __name__ == "__main__":
    main()
