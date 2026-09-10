"""plot_sem_loss.py

Grafica la evolucion por epoca de la perdida semantica total y del
coeficiente lambda(t) a partir del CSV sem_loss_log.csv volcado durante
el entrenamiento.

Uso:
    python plot_sem_loss.py [ruta_al_csv]
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sem_loss_log.csv")
WARMUP_END = 30
RAMP_END = 80


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df.groupby("epoch").mean(numeric_only=True).reset_index()


def plot_terms(df: pd.DataFrame, out_path: Path) -> None:
    terms = [
        ("sem_loss", r"$\mathcal{L}_{\mathrm{sem}}$ total"),
        ("lambda",   r"$\lambda(t)$ (coeficiente)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, (col, title) in zip(axes, terms):
        if col not in df.columns:
            ax.set_title(f"{title} (columna ausente)")
            continue
        ax.plot(df["epoch"], df[col], color="tab:blue", linewidth=1.5)
        ax.axvline(WARMUP_END, color="gray", linestyle="--", linewidth=0.8,
                   label=f"warmup ({WARMUP_END})")
        ax.axvline(RAMP_END, color="black", linestyle="--", linewidth=0.8,
                   label=f"ramp end ({RAMP_END})")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("valor")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Evolucion de la perdida semantica y su coeficiente",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafica guardada en {out_path}")


def main() -> None:
    if not CSV_PATH.exists():
        sys.exit(f"No existe {CSV_PATH}. Pasa la ruta como argumento.")
    df = load_and_aggregate(CSV_PATH)
    print(f"Epocas cargadas: {df['epoch'].min()}..{df['epoch'].max()}")
    print(df[["epoch", "sem_loss", "lambda"]].describe().round(4))
    out_path = CSV_PATH.with_name(CSV_PATH.stem + "_evolucion.png")
    plot_terms(df, out_path)
    plt.show()


if __name__ == "__main__":
    main()