# doc-processor-api (service #2 — Document Processing)

Converts a raw financial-report PDF into **independent, retrieval-ready elements** using a
deep-learning layout + OCR + table-structure pipeline (PaddleOCR PP-Structure: PicoDet layout
detection, PP-OCR text detection/recognition, SLANet table structure recognition).

Every element carries: `document_id`, `page` (1-based), `section`, `content_type`
(`text` | `heading` | `table`), `content` (string, or `{headers, rows}` for tables),
a normalized `bbox`, and a unique id (`chunk_id`). Elements are independent (no parent/child in Phase 1).

## Endpoints
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | – | `{"status":"ok"}` |
| POST | `/process` | multipart/form-data: `file` (PDF, required), `document_id` (optional), `dpi` (optional, default 200) | `{document_id, num_pages, elements[]}` |

`document_id` defaults to the uploaded filename without `.pdf` (for TAT-DQA that is the
dataset's `doc.uid`), falling back to sha1 of the bytes.

## Element example
```json
{
  "chunk_id": "d5d739657785641f4689be3d234e0a8f_p1_b3",
  "document_id": "d5d739657785641f4689be3d234e0a8f",
  "page": [1],
  "section": "Notes to the Company financial statements (continued)",
  "content_type": "table",
  "content": {"headers": ["", "2019 m", "", "2018 m"],
              "rows": [["Cost:", "", "", ""], ["1 April", "", "91,905", "91,902"]]},
  "bbox": [0.057, 0.252, 0.885, 0.510],
  "metadata": {"source": "paddleocr_ppstructure", "block_type": "table",
               "bbox_unit": "normalized", "render_dpi": 200, "page_image_size": [1654, 2339]}
}
```
`bbox` is `[x0, y0, x1, y1]` as fractions of the page image; multiply by the rendered page
width/height to draw it. Field names match the retrieval-api `Chunk` model, so
`Chunk(**element)` works directly. Table cells are kept as strings (e.g. `"91,905"`).

## Setup (Linux / WSL, Python 3.10 required — Paddle does not support newer Pythons)
```bash
sudo apt install -y libgomp1                       # runtime lib needed by paddlepaddle
micromamba create -y -n ledger-docproc python=3.10 pip
micromamba run -n ledger-docproc python -m pip install -r services/doc-processor-api/requirements.txt
```
(Any Python 3.10 venv works; micromamba is just what we use.)

## Run

> Note: Paddle/PaddleOCR requires Python 3.10. We typically run inside a micromamba env named `ledger-docproc`.

### Option A (recommended): via micromamba
From the repo root:

```bash
micromamba run -n ledger-docproc uvicorn app.main:app \
  --app-dir services/doc-processor-api \
  --host 0.0.0.0 --port 8001
```
Models (~35 MB) are downloaded automatically on first start to ~/.paddleocr.


## Test
```bash
curl -s http://localhost:8001/health
curl -s -F "file=@/path/to/d5d739657785641f4689be3d234e0a8f.pdf" http://localhost:8001/process > out.json
```
### Option B (run env binaries directly if micromamba run is broken)

```bash
cd ~/MIA-Project-Team_6
ENV="$HOME/.local/share/mamba/envs/ledger-docproc"
export PYTHONPATH="services/doc-processor-api"

# optional: stabilize on low-memory machines
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

$ENV/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```
## Known limitations (v0.1)
- The default PubLayNet layout model has limited recall on borderless financial tables
  (adjacent tables may be merged or missed). Being measured against TAT-DQA ground truth
  and improved in a follow-up PR.
- CPU inference, roughly 5–15 s per page.

## Full-page OCR safety net (default ON)
This service appends one extra element per page:

- `chunk_id: <document_id>_p<page>_fullocr`
- `content_type: "text"`
- `bbox: null`
- `metadata.full_page_ocr = true`

This is a recall safety net: if the layout model misses a table region, the numbers still appear in `fullocr` text and can be retrieved/cited.
Downstream retrieval can optionally filter these out (e.g. only index them for BM25).

## Multi-page stability: DPI cap (for low-memory machines)
Some multi-page PDFs can crash/kill the process at high DPI on low-memory machines.
To keep the service stable:

- If `num_pages > 1`, the processor caps DPI to `DOC_PROCESSOR_MAX_DPI_MULTIPAGE` (default `120`).
- The effective DPI is recorded in each element's metadata:
  - `requested_dpi`
  - `render_dpi`
  - `dpi_capped`

Override on stronger machines:
```bash
export DOC_PROCESSOR_MAX_DPI_MULTIPAGE=200
```

## Heading promotion heuristic
Some headings are sometimes detected as plain `text` blocks by the layout model (common in financial PDFs).
We apply a lightweight post-processing heuristic (regex + bbox height vs median text height) to promote such blocks to:

- `content_type: "heading"`
- `metadata.heading_promoted: true`

This improves `section` assignment for downstream retrieval and citations.

---

## MinerU 2.5 Engine & Side-by-Side Benchmarking

In addition to PaddleOCR, this service provides high-speed native PDF parsing and layout extraction powered by **MinerU 2.5**.

### Advantages of MinerU 2.5:
- **~18x Faster**: Processes pages in ~0.2s vs ~15–40s on CPU.
- **Superior Financial Table Extraction**: Isolates distinct tables without merging adjacent financial data.
- **Zero Heavy Weights Required**: Starts up instantly without requiring multi-hundred megabyte model downloads.
- **Full Backward Compatibility**: Emits the identical `(document_id, num_pages, elements)` schema required by downstream `retrieval-api` and `orchestrator`.

### Selecting the Processor Engine in the Pipeline:
By default in this branch, `doc-processor-api` runs with MinerU (`DOC_PROCESSOR_ENGINE=mineru`). To toggle engines:
```bash
# Run with MinerU (default)
DOC_PROCESSOR_ENGINE=mineru uvicorn app.main:app --app-dir services/doc-processor-api --host 0.0.0.0 --port 8001

# Run with PaddleOCR (legacy)
DOC_PROCESSOR_ENGINE=paddle uvicorn app.main:app --app-dir services/doc-processor-api --host 0.0.0.0 --port 8001
```

Or pass `engine="mineru"` or `engine="paddle"` dynamically in the multipart form body to `/process`.

### Testing & Verification Scripts:

1. **Verify Parity (Ensuring Parallel Processing Matches Sequential 1:1)**:
   ```bash
   python services/doc-processor-api/test_mineru_vs_paddle.py --verify-parity
   ```

2. **Run Concurrent Side-by-Side Visual Comparison**:
   ```bash
   python services/doc-processor-api/compare_visual.py --parallel --paddle-pages 1
   ```
   Renders color-coded bounding boxes and stitched side-by-side comparison images into `services/doc-processor-api/visual_output/`.

3. **Test with Any Custom PDF**:
   ```bash
   python services/doc-processor-api/test_mineru_vs_paddle.py --pdf "path/to/report.pdf" --parallel
   python services/doc-processor-api/compare_visual.py --pdf "path/to/report.pdf"
   ```

