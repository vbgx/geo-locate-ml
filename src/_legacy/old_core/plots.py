from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics_csv(metrics_csv: str, out_base: str) -> dict:
    df = pd.read_csv(metrics_csv)

    out_base = Path(out_base)
    loss_png = out_base.with_name(out_base.stem + "_loss.png")
    acc_png = out_base.with_name(out_base.stem + "_valacc.png")

    # Loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"])
    plt.xlabel("epoch")
    plt.ylabel("train_loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(loss_png, dpi=160)
    plt.close()

    # Val acc
    plt.figure()
    plt.plot(df["epoch"], df["val_acc"])
    plt.xlabel("epoch")
    plt.ylabel("val_acc")
    plt.title("Validation accuracy")
    plt.tight_layout()
    plt.savefig(acc_png, dpi=160)
    plt.close()

    return {"loss_png": str(loss_png), "valacc_png": str(acc_png)}
