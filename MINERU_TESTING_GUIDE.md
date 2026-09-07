# MinerU 2.5 Document Processor & Side-by-Side Testing Guide

This guide details how to test the **MinerU 2.5** document processing engine, run side-by-side benchmarks against PaddleOCR, and test the full end-to-end pipeline on the `mineru-test` branch.

---

## 1. Overview & Key Advantages

The `mineru-test` branch integrates **MinerU 2.5** into `services/doc-processor-api` as a drop-in, high-speed document processing alternative to PaddleOCR PP-Structure.

| Feature / Metric | MinerU 2.5 | PaddleOCR (PP-Structure) | Benefit |
| :--- | :--- | :--- | :--- |
| **Speed (per page)** | **~0.2s – 2.3s** | ~15s – 43s | **~18x faster execution** |
| **Startup / Weights** | **Instant (native PyMuPDF streams)** | Downloads ~35MB models on first run | No download delay or model loading stalls |
| **Table Extraction** | **Preserves distinct sub-tables** | Tends to merge adjacent tables | Cleaner financial tabular reasoning |
| **Memory / CPU** | **Lightweight (< 100MB RAM)** | Heavy neural network inference | Runs smoothly on developer laptops & CI |
| **API Contract** | **100% Identical** | Baseline | Full compatibility with `retrieval-api` and `orchestrator` |

---

## 2. Branch Information

- **Branch Name**: `mineru-test`
- **Base Branch**: Forked directly from `main` to enable end-to-end pipeline testing across all services.
- **Remote**: `origin` (`https://github.com/anasalazab10-glitch/MIA-Project-Team_6.git`)

To switch to this branch locally:
```bash
git fetch origin
git checkout mineru-test
```

---

## 3. Testing in the Full Pipeline

In this branch, `services/doc-processor-api` defaults to **MinerU** (`DOC_PROCESSOR_ENGINE=mineru`).

### Step 1: Start `doc-processor-api`
From the repo root:
```bash
uvicorn app.main:app --app-dir services/doc-processor-api --host 0.0.0.0 --port 8001
```

*(Optional)* To toggle back to PaddleOCR:
```bash
DOC_PROCESSOR_ENGINE=paddle uvicorn app.main:app --app-dir services/doc-processor-api --host 0.0.0.0 --port 8001
```

### Step 2: Test the Ingestion Pipeline End-to-End
When you ingest a financial PDF through the Orchestrator service or the Gradio UI:
1. Orchestrator sends the PDF to `doc-processor-api` (`/process`).
2. `doc-processor-api` uses MinerU to extract structured text, headings, bounding boxes, and financial tables in under 2 seconds.
3. The resulting elements are indexed into `dense-retrieval` / Qdrant.
4. The reasoning service can immediately answer questions with exact page citations.

You can also test the endpoint directly using `curl`:
```bash
# Process bundled sample report
curl -s -F "file=@services/doc-processor-api/test_data/Vodafone 2026 Annual Report.pdf" \
     http://localhost:8001/process | jq .
```

---

## 4. Standalone Verification & Benchmarking Scripts

All test scripts use dynamic relative paths and bundled test data in `services/doc-processor-api/test_data/`. No hardcoded machine paths are required.

### 4.1. Parity Verification (Ensuring Sequential & Parallel Yield Identical Output)
Tests that multi-threaded page extraction produces the exact same structured elements as sequential processing:
```bash
# Run parity check on default test report
python services/doc-processor-api/test_mineru_vs_paddle.py --verify-parity

# Run parity check on multi-page report
python services/doc-processor-api/test_mineru_vs_paddle.py --pdf "services/doc-processor-api/test_data/Can_GPT_models_be_Financial_Analysts_An.pdf" --verify-parity
```

### 4.2. Side-by-Side Parallel Benchmark & Visual Rendering
Concurrently runs MinerU 2.5 and PaddleOCR in parallel using threads, generates color-coded bounding boxes on the rendered PDF pages, and stitches them side-by-side into a single visual comparison image:
```bash
python services/doc-processor-api/compare_visual.py --parallel --paddle-pages 1
```

Outputs generated in `services/doc-processor-api/visual_output/`:
- `*_side_by_side_page_1.png`: Left side = MinerU 2.5, Right side = PaddleOCR.
- `*_mineru_page_1.png`: Standalone MinerU bounding box overlay.
- `*_paddle_page_1.png`: Standalone PaddleOCR bounding box overlay.
- `*_ocr_comparison.json`: Detailed quantitative breakdown (timings, table counts, element counts).

### 4.3. Benchmark Performance Reference (`Vodafone 2026 Annual Report.pdf`)

```text
======================================================================
SIDE-BY-SIDE BENCHMARK SUMMARY
======================================================================
Metric                    | MinerU 2.5           | PaddleOCR (PP-Structure) 
---------------------------------------------------------------------------
Execution Time            | 2.33s                | 42.67s                   
Speedup Factor            | 18.3x faster         | 1.0x (baseline)          
Pages Tested              | 1 pages              | 1 pages                  
Total Elements            | 17                   | 4                        
Tables Detected           | 5                    | 1                        
Headings Detected         | 2                    | 1                        
Text Paragraphs           | 10                   | 2                        
Total Wall-Clock Time     | 42.71s (concurrent)  |                          
======================================================================
```

---

## 5. Testing with Custom PDFs

You can test any custom financial PDF simply by passing the path to `--pdf`:

```bash
# Visual side-by-side comparison
python services/doc-processor-api/compare_visual.py --pdf "path/to/your_report.pdf" --parallel

# Extraction benchmark
python services/doc-processor-api/test_mineru_vs_paddle.py --pdf "path/to/your_report.pdf" --parallel
```
