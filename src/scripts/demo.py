from __future__ import annotations
import subprocess
from pathlib import Path

from .paths import p

def main():
    subprocess.run(["python", "-m", "src.run"], check=True)
    for i in range(5):
        print(f"\n--- DEMO PREDICT {i+1}/5 ---")
        subprocess.run(["python", "-m", "src.predict", "--ensemble"], check=True)

    latest = p("runs", "latest")
    demo_report = latest / "DEMO_REPORT.md"
    demo_report.write_text(
        "# DEMO REPORT\n\n"
        "- Ran: python -m src.run\n"
        "- Then: 5x python -m src.predict --ensemble\n",
        encoding="utf-8",
    )
    print("\nWrote:", demo_report)

if __name__ == "__main__":
    main()
