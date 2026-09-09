import json
import requests
from datetime import datetime


EVAL_URL = "http://localhost:8005/run_benchmark"

OUTPUT_FILE = "benchmark_100_raw.json"


def main():
    print("Starting 100-question benchmark...")
    print("-" * 60)

    payload = {
        "limit": 100,
        "debug": True
    }

    start_time = datetime.now()

    try:
        response = requests.post(
            EVAL_URL,
            json=payload,
            timeout=3600
        )

        response.raise_for_status()

        results = response.json()

    except requests.RequestException as exc:
        print(f"ERROR: Benchmark request failed: {exc}")
        return

    end_time = datetime.now()

    # Save the exact response from the Eval Service
    output = {
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("100-question benchmark finished.")
    print("=" * 60)
    print(f"Raw results saved to: {OUTPUT_FILE}")
    print(f"Started:  {start_time}")
    print(f"Finished: {end_time}")
    print(
        f"Duration: "
        f"{(end_time - start_time).total_seconds():.2f} seconds"
    )


if __name__ == "__main__":
    main()