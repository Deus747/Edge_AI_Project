# Dataset Preprocessing

This folder contains the IDD annotation conversion notebook:

```text
idd_polygon_to_masks_preprocessing.ipynb
```

The notebook converts raw IDD polygon JSON annotations into model-ready masks for semantic segmentation, instance segmentation, color visualization, and optional panoptic-style outputs.

## What the Notebook Does

The raw IDD dataset stores labels as polygons in JSON files under `gtFine`. A segmentation model cannot train directly from those polygon files; it needs dense pixel masks. The notebook rasterizes each polygon into an image-sized mask.

The main conversion function is:

```python
run_pipeline(
    datadir,
    out_basedir,
    encoding,
    do_semantic,
    do_instance,
    do_color,
    do_panoptic,
)
```

Supported label encodings include:

| Encoding | Purpose |
| --- | --- |
| `level1Id` | Coarse semantic labels, 7 classes. |
| `level2Id` | Medium semantic labels, 16 classes. |
| `level3Id` | Fine semantic labels, 26 classes. |
| `id`, `csId`, `csTrainId`, `level4Id`, `unifiedId` | Alternate IDD/Cityscapes-compatible encodings. |

The generated outputs can include:

| Output | Description |
| --- | --- |
| Semantic masks | Single-channel PNG masks where each pixel stores a class ID. |
| Instance masks | PNG masks using `class_id * 1000 + instance_index`. |
| Color masks | Visual RGB/RGBA previews of the labels. |
| Panoptic outputs | Panoptic-style PNG and JSON metadata. |

## Expected Raw IDD Layout

Set `DATADIR` in the notebook to the root of the IDD segmentation dataset. A typical raw layout is:

```text
IDD-Segmentation/
  leftImg8bit/
    train/
    val/
    test/
  gtFine/
    train/
    val/
    test/
```

The exact city/sequence subfolders can remain as provided by IDD. The notebook walks the annotation tree and finds the matching images.

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Open:

```text
idd_polygon_to_masks_preprocessing.ipynb
```

## How to Run

1. Edit `DATADIR` to point to the raw IDD dataset root.
2. Edit `OUT_BASEDIR` to the output folder where converted masks should be written.
3. Choose the label encoding. For the main deployed semantic model, use `level2Id`.
4. Choose which outputs to generate.
5. Run the notebook cells.

Example:

```python
DATADIR = r"/path/to/IDD-Segmentation"
OUT_BASEDIR = r"/path/to/IDD-Segmentation/LabelLevel2Id"

run_pipeline(
    datadir=DATADIR,
    out_basedir=OUT_BASEDIR,
    encoding="level2Id",
    do_semantic=True,
    do_instance=True,
    do_color=True,
    do_panoptic=False,
)
```

## Output for Semantic Training

After conversion, arrange or copy outputs into the structure expected by the training code:

```text
IDDL2/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

The same pattern applies for `IDDL1` and `IDDL3`.

Semantic masks should use:

```text
0..C-1 = valid class IDs
255    = ignore label
```

## Notes

- Unknown labels are skipped.
- Labels ending in `group` are normalized before lookup.
- Instance masks use `class_id * 1000 + instance_index`.
- Background or ignored pixels are encoded as `255` for semantic masks.
- Panoptic export is optional and not required for the semantic training scripts.

