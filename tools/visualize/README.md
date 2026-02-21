MapView — geo-locate-ml

Interactive map visualization tool for dataset exploration and validation.

Main script:

tools/visualize/mapview.py

This tool generates an interactive mapview.html file from indexed dataset files.

📦 Requirements

From the project root:

pip install folium pandas pyarrow

Optional (H3 layer support):

pip install h3
🗺️ Generate the Map

From the repository root:

python tools/visualize/mapview.py build --out mapview.html

This generates:

mapview.html

Open it:

open mapview.html        # macOS
xdg-open mapview.html    # Linux
🔬 Add Validation Layers (Errors / Diagnostics)
python tools/visualize/mapview.py build \
  --add-val-layers \
  --labels-json data/index/labels.json \
  --out mapview.html

Optional flags:

--far-heatmap     # Heatmap of far errors
--h3              # H3 aggregation layer (requires h3)
📤 Export Splits from Parquet
python tools/visualize/mapview.py export-splits \
  --parquet data/dataset.parquet \
  --out data/index/splits.jsonl
✅ Check Split Coverage
python tools/visualize/mapview.py check-coverage
📁 Expected Data Structure

The script expects the following structure (relative to repo root):

data/
  index/
    images.jsonl
    splits.jsonl          # optional
    labels.json           # required for validation layers
💡 Notes

Paths are always resolved relative to the repository root.

mapview.html is a static file and can be shared or hosted anywhere.

No external services required — fully local visualization.