# CPU Model Training on Another Computer

The dataset stays on the training computer. This repository does not need a copy
of the images or labels to prepare the low-spec runtime.

Install the normal training dependencies on the GPU computer, then train with:

```powershell
python scripts/train_cpu_model.py --data D:\datasets\dnf\dataset.yaml --device 0
```

The dataset YAML must preserve the existing class order:

```text
people, door, monster, brand, menu, article, purple_card, hero, elite, boss-n, boss-m
```

Export each candidate after training. Start with 416, then compare 320 and 512
only when the validation and CPU benchmark results justify it:

```powershell
python scripts/export_cpu_onnx.py `
  --weights runs\cpu_training\yolo11n_640\weights\best.pt `
  --output models\cpu\best_dxc_yolo11n_416_fp32.onnx --imgsz 416
```

Copy both the `.onnx` file and adjacent `.json` metadata file to the low-spec
computer. Then install `requirements-cpu.txt` and run the benchmark against
captured game images.

At startup, `python main.py` checks the available CUDA GPUs. A GPU with at least
4 GiB of total VRAM uses `config/settings.yaml`; all other machines use
`config/settings.cpu.yaml`. The largest GPU is selected when more than one GPU is
available. Use `-c` to choose a configuration explicitly, or `-d cpu` / `-d cuda`
to override the automatically selected device profile.

If the low-spec model is not present, the program reports the expected path
`models/cpu/best_dxc_yolo11n_416_fp32.onnx` and exits during detector startup.

Do not copy the full dataset to the runtime computer. Keep train, validation, and
test scenes separate by recording session so adjacent video frames do not cross
sets.
