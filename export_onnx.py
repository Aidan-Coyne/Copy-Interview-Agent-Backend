# export_onnx.py
import os
import torch
from transformers import AutoModel, AutoTokenizer

# 1) Model identifier
MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# 2) Where to write the ONNX file so your code will find it:
OUTPUT_DIR  = os.path.join("keyword_extraction", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "paraphrase-MiniLM-L3-v2.onnx")

# 3) Load model & tokenizer
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModel.from_pretrained(MODEL_ID)
model.eval()

# 4) Dummy input
dummy_text = "This is a dummy input for ONNX export."
inputs     = tokenizer(dummy_text, return_tensors="pt")

# 5) Export with opset_version=14 to include scaled_dot_product_attention
print(f"Exporting to ONNX at {OUTPUT_PATH} …")
with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":         {0: "batch", 1: "sequence"},
            "attention_mask":    {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
    )

print("✅ ONNX export complete.")
