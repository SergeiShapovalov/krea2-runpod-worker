import json
import traceback


def main() -> None:
    diagnostics = {}

    try:
        import comfy_kitchen

        diagnostics["comfy_kitchen_file"] = getattr(comfy_kitchen, "__file__", None)
        diagnostics["comfy_kitchen_backends"] = comfy_kitchen.list_backends()
    except Exception:
        diagnostics["comfy_kitchen_error"] = traceback.format_exc()

    try:
        from comfy_kitchen.tensor import get_layout_class

        diagnostics["comfy_kitchen_tensor_layout"] = str(get_layout_class("TensorCoreFP8Layout"))
    except Exception:
        diagnostics["comfy_kitchen_tensor_error"] = traceback.format_exc()

    try:
        import comfy.quant_ops as quant_ops

        diagnostics["comfy_quant_ops"] = {
            "_CK_AVAILABLE": getattr(quant_ops, "_CK_AVAILABLE", None),
            "layout": str(quant_ops.get_layout_class("TensorCoreFP8Layout")),
        }
    except Exception:
        diagnostics["comfy_quant_ops_error"] = traceback.format_exc()

    print("krea2-build-diagnostics=" + json.dumps(diagnostics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
