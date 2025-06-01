import onnx

ruta_modelo = "models/arcfaceresnet100-8.onnx"
modelo = onnx.load(ruta_modelo)

print("===== OPSSET_IMPORT =====")
for imp in modelo.opset_import:
    print(f"  dominio='{imp.domain}'  versión={imp.version}")

print("\n===== INPUTS =====")
for inp in modelo.graph.input:
    name    = inp.name
    ttype   = inp.type.tensor_type
    shape   = [d.dim_value for d in ttype.shape.dim]
    dtype   = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[ttype.elem_type]
    print(f"  · {name} : dtype={dtype}, shape={shape}")
