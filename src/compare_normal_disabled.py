import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_trc(path):
    with open(path) as f:
        lines = f.readlines()

    markers = [m.strip().split(":")[-1] for m in lines[3].split("\t")[2:] if m.strip() != ""]

    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    df = df.dropna(axis=1, how="all")

    cols = ["Frame", "Time"]
    for m in markers:
        cols += [f"{m}_X", f"{m}_Y", f"{m}_Z"]

    df = df.iloc[:, :len(cols)]
    df.columns = cols

    return df


def simulate_disabled(df, factor=0.4):
    df2 = df.copy()

    for side in ["R"]:  # right arm
        SHO = f"{side}SHO"
        ELB = f"{side}ELB"
        WRA = f"{side}WRA"

        for axis in ["X","Y","Z"]:
            s = df[f"{SHO}_{axis}"]
            e = df[f"{ELB}_{axis}"]
            w = df[f"{WRA}_{axis}"]

            df2[f"{ELB}_{axis}"] = s + factor * (e - s)
            df2[f"{WRA}_{axis}"] = s + factor * (w - s)

    return df2


def plot_comparison(df, df_disabled):
    fig = plt.figure(figsize=(12,6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    markers = ["RSHO","RELB","RWRA","C7","CLAV"]

    for m in markers:
        ax1.plot(df[f"{m}_X"], df[f"{m}_Y"], df[f"{m}_Z"], label=m)
        ax2.plot(df_disabled[f"{m}_X"], df_disabled[f"{m}_Y"], df_disabled[f"{m}_Z"], label=m)

    ax1.set_title("Mouvement normal")
    ax2.set_title("Mouvement simulé (épaule limitée)")

    plt.legend()
    plt.show()


def main():
    path = os.path.join("data","viconRecord.TRC")

    df = load_trc(path)
    df_disabled = simulate_disabled(df)

    plot_comparison(df, df_disabled)


if __name__ == "__main__":
    main()