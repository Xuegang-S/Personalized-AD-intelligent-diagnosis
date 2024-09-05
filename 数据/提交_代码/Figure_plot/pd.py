import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

plt.rcParams['figure.figsize'] = (18.0, 22.0)
plt.rc('font', family='Times New Roman', size=18)


def print_interval(path):
    df = pd.read_excel(path, sheet_name=0)
    q1 = df["Score"].quantile(1/3.)
    q2 = df["Score"].quantile(2/3.)
    q_min = df["Score"].min()
    q_max = df["Score"].max()
    print(f"q1:{q_min} - {q1}")
    print(f"q2:{q1} - {q2}")
    print(f"q1:{q2} - {q_max}")


def plot_mean_weighted_curve(path, sheet, data=None):
    y_label_list = ["mm3"] * 6 + ["mm"] * 16 + ["mm2"] * 2
    y_ticks_list = [
        # hippocampus
        np.linspace(3000, 5000, 5),
        np.linspace(3000, 5000, 5),
        # amygdala
        np.linspace(1000, 2200, 5),
        np.linspace(1000, 2200, 5),
        # Inf_Lat_Vent
        np.linspace(50, 1250, 5),
        np.linspace(50, 1250, 5),
        # entorhil
        # np.linspace(2.5, 4.0, 5),
        # np.linspace(2.5, 4.0, 5),
        [2.50, 2.87, 3.25, 4.00],
        [2.50, 2.87, 3.25, 4.00],
        # middletemporal
        # np.linspace(2.5, 3.25, 5),
        # np.linspace(2.5, 3.25, 5),
        [2.50, 2.68, 2.87, 3.06, 3.25],
        [2.50, 2.68, 2.87, 3.06, 3.25],
        # inferiortemporal
        np.linspace(2.5, 3.5, 5),
        np.linspace(2.5, 3.5, 5),
        # fusiform
        # np.linspace(2.5, 3.25, 5),
        # np.linspace(2.5, 3.25, 5),
        [2.5, 2.68, 2.87, 3.06, 3.25],
        [2.5, 2.68, 2.87, 3.06, 3.25],
        # superiortemporal
        np.linspace(2.5, 3.5, 5),
        np.linspace(2.5, 3.5, 5),
        # inferiorparietal
        # np.linspace(2.25, 3.0, 5),
        # np.linspace(2.25, 3.0, 5),
        [2.25, 2.43, 2.62, 2.81, 3.00],
        [2.25, 2.43, 2.62, 2.81, 3.00],
        # precuneus
        # np.linspace(2.25, 3.0, 5),
        # np.linspace(2.25, 3.0, 5),
        [2.25, 2.43, 2.62, 2.81, 3.00],
        [2.25, 2.43, 2.62, 2.81, 3.00],
        # caudalmiddlefrontal
        # np.linspace(2.5, 3.25, 5),
        # np.linspace(2.5, 3.25, 5),
        [2.5, 2.68, 2.87, 3.06, 3.25],
        [2.5, 2.68, 2.87, 3.06, 3.25],
        # inferiortemporal
        np.arange(2500, 5000, 500),
        np.arange(2500, 5000, 500)
    ]

    df = pd.read_excel(path, sheet_name=sheet)
    y_lbl = df.columns.tolist()[0]
    EIS_temp = {"val": [], "ratio": []}
    q1 = df["PRS"].quantile(1 / 3.)
    q2 = df["PRS"].quantile(2 / 3.)
    df1 = df[df["PRS"] <= q1]
    df2 = df[(q1 < df["PRS"]) & (df["PRS"] <= q2)]
    df3 = df[q2 < df["PRS"]]
    for prs_idx, (sub_df, color) in enumerate(zip((df1, df2, df3), ('blue', 'green', 'red'))):
        sub_df_prs = sub_df.groupby(["Age"]).mean()
        y_prs = sub_df_prs[y_lbl].tolist()
        x_prs = sub_df_prs.index.tolist()
        y_prs = gaussian_filter1d(y_prs, sigma=4)
        plt.plot(x_prs, y_prs, color, linewidth=2.5)

        q1 = sub_df["EIS"].quantile(1 / 3.)
        q2 = sub_df["EIS"].quantile(2 / 3.)
        sub_df1 = sub_df[sub_df["EIS"] <= q1]
        sub_df2 = sub_df[(q1 < sub_df["EIS"]) & (sub_df["EIS"] <= q2)]
        sub_df3 = sub_df[q2 < sub_df["EIS"]]
        x_cor = [43, 55, 68]
        y_cor = []
        for ssub_df in [sub_df1, sub_df3]:
            ssub_df_eis = ssub_df.groupby(["Age"]).mean()
            y_eis = ssub_df_eis[y_lbl].tolist()
            x_eis = ssub_df_eis.index.tolist()
            y_eis = gaussian_filter1d(y_eis, sigma=5)
            y_cor.append(np.array(y_eis)[[2, 14, 27]])
            plt.plot(x_eis, y_eis, linestyle='--', c=color, alpha=0.5, linewidth=1.5)

        y_min = np.min(y_cor, axis=0)
        y_max = np.max(y_cor, axis=0)
        # difference_text(x_cor, y_min, y_max, np.array(y_prs)[[2, 14, 27]], sheet, prs_idx, EIS_temp)

    label_font_size = 18
    plt.xlabel("Age", fontsize=label_font_size)
    # if sheet <= 5:
    #     plt.ylabel("Volume")
    # elif sheet >= 22:
    #     plt.ylabel("Area")
    # else:
    #     plt.ylabel("Mean thickness")
    plt.ylabel(y_label_list[sheet], fontsize=label_font_size)
    plt.yticks(y_ticks_list[sheet])
    plt.ylim(min(y_ticks_list[sheet]), max(y_ticks_list[sheet]))

    plt.title(y_lbl + '\n', fontsize=label_font_size)
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.tight_layout()

    # prs difference
    temp = []
    df2_prs = df2.groupby(["Age"]).mean()
    df2y_prs = df2_prs[y_lbl].tolist()
    df2y_prs = gaussian_filter1d(df2y_prs, sigma=5)
    prs = np.array(df2y_prs)[[2, 14, 27]]
    for sub_df in (df1, df3):
        sub_df_prs = sub_df.groupby(["Age"]).mean()
        y_prs = sub_df_prs[y_lbl].tolist()
        y_prs = gaussian_filter1d(y_prs, sigma=5)
        temp.append(np.array(y_prs)[[2, 14, 27]])
    y_min = np.min(temp, axis=0)
    y_max = np.max(temp, axis=0)
    PRS_temp = {"val": [], "ratio": []}
    for i in range(3):
        diff = y_max[i] - y_min[i]
        relative_diff = diff / 2 / prs[i] * 100
        PRS_temp["val"].append(diff)
        PRS_temp["ratio"].append(relative_diff)

    # if data is not None:
    #     data[y_lbl] = {
    #         "prs_diff": sum(PRS_temp["val"]) / len(PRS_temp["val"]),
    #         "prs_ratio": sum(PRS_temp["ratio"]) / len(PRS_temp["ratio"]),
    #         "eis": sum(EIS_temp["val"]) / len(EIS_temp["val"]),
    #         "eis_ratio":sum(EIS_temp["ratio"]) / len(EIS_temp["ratio"])
    #     }


def plot_separate_curve(path, sex=None):
    df = pd.read_excel(path, sheet_name=0)
    if sex:
        df = df[df["Sex"] == sex]
    y_lbl = df.columns.tolist()[0]

    q1 = df["PRS"].quantile(1 / 3.)
    q2 = df["PRS"].quantile(2 / 3.)
    df1 = df[df["PRS"] <= q1]
    df2 = df[(q1 < df["PRS"]) & (df["PRS"] <= q2)]
    df3 = df[q2 < df["PRS"]]
    x_cor = [43, 55, 68]
    y_cor = []
    for sub_df in [df1, df2, df3]:
        sub_df = sub_df.groupby(["Age"]).mean()
        y = sub_df[y_lbl].tolist()
        x = sub_df.index.tolist()
        y = gaussian_filter1d(y, sigma=5)
        y_cor.append(np.array(y)[[2, 14, 27]])
        plt.plot(x, y, 'r', alpha=0.5, linewidth=1.5)
    y_min = np.min(y_cor, axis=0)
    y_max = np.max(y_cor, axis=0)
    difference_text(x_cor, y_min, y_max, loc="upper")

    q1 = df["EIS"].quantile(1 / 3.)
    q2 = df["EIS"].quantile(2 / 3.)
    df1 = df[df["PRS"] <= q1]
    df2 = df[(q1 < df["EIS"]) & (df["EIS"] <= q2)]
    df3 = df[q2 < df["EIS"]]
    x_cor = [43, 55, 68]
    y_cor = []
    for sub_df in [df1, df2, df3]:
        sub_df = sub_df.groupby(["Age"]).mean()
        y = sub_df[y_lbl].tolist()
        x = sub_df.index.tolist()
        y = gaussian_filter1d(y, sigma=5)
        y_cor.append(np.array(y)[[2, 14, 27]])
        plt.plot(x, y, 'g', alpha=0.5, linewidth=1.5)
    y_min = np.min(y_cor, axis=0)
    y_max = np.max(y_cor, axis=0)
    difference_text(x_cor, y_min, y_max, loc="upper")

    plt.xlabel("Age")
    plt.ylabel(y_lbl)
    if sex == 0:
        plt.title("Female")
    elif sex == 1:
        plt.title("Male")
    else:
        plt.title("All")
    plt.tight_layout()
    plt.show()


def linear_regression(path):
    df = pd.read_excel(path, sheet_name=0)
    y_lbl = df.columns.tolist()[0]

    PRS = df["PRS"]
    Age = df["Age"]
    EIS = df["EIS"]
    volume = df[y_lbl]
    plt.scatter(EIS, volume, marker='x')
    x = np.linspace(EIS.min(), EIS.max(), 500)
    y = 5681.251833 * x + 3542.214005
    plt.plot(x, y, 'r')
    plt.ylabel(y_lbl)
    plt.xlabel("EIS")
    plt.text(0.1, 2200, "y=5681.25x+3542.21")
    plt.tight_layout()
    plt.show()


def mean_std_line(path):
    df = pd.read_excel(path, sheet_name=0)
    y_lbl = df.columns.tolist()[0]

    df_gp = df.groupby(["Age"])
    age = df_gp.indices.keys()
    mean = df_gp[y_lbl].mean().tolist()
    std = df_gp[y_lbl].std().tolist()
    upper = np.array(mean) + 2 * np.array(std)
    lower = np.array(mean) - 2 * np.array(std)

    plt.scatter(df["Age"], df[y_lbl], marker='x')
    plt.plot(age, gaussian_filter1d(upper, sigma=5), 'r', linewidth=2.5)
    plt.plot(age, gaussian_filter1d(lower, sigma=5), 'r', linewidth=2.5)

    plt.xlabel("Age")
    plt.ylabel(y_lbl)
    plt.tight_layout()
    plt.show()


def difference_text(x, y_min, y_max, prs, sheet, prs_idx, data):
    for i in range(3):
        plt.plot((x[i]-0.5, x[i]+0.5), (y_min[i], y_min[i]), 'k', linewidth=1)
        plt.plot((x[i]-0.5, x[i]+0.5), (y_max[i], y_max[i]), 'k', linewidth=1)
        plt.plot((x[i], x[i]), (y_min[i], y_max[i]), 'k', linewidth=1)

        diff = y_max[i] - y_min[i]
        relative_diff = diff / 2 / prs[i] * 100
        data["val"].append(diff)
        data["ratio"].append(relative_diff)
        diff_text = "{:.1e}\n({:.1f}%)".format(diff, relative_diff)
        diff_text = diff_text.replace('e', 'E')
        if sheet == 4:
            if prs_idx == 0:
                if i == 0:
                    plt.text(x[i], prs[i] + 450, diff_text, ha="center", va="center", size=15, c="blue")
                else:
                    plt.text(x[i], y_min[i] - 120, diff_text, ha="center", va="center", size=15)
            elif prs_idx == 1:
                if i == 0:
                    plt.text(x[i], prs[i] + 650, diff_text, ha="center", va="center", size=15, c="green")
                else:
                    plt.text(x[i], prs[i], diff_text, ha="center", va="center", size=15)
            elif prs_idx == 2:
                if i == 0:
                    plt.text(x[i], prs[i] + 700, diff_text, ha="center", va="center", size=15, c="red")
                else:
                    plt.text(x[i], prs[i], diff_text, ha="center", va="center", size=15)
        elif sheet == 5:
            if prs_idx == 0:
                if i == 0:
                    plt.text(x[i], prs[i] + 450, diff_text, ha="center", va="center", size=15, c="blue")
                else:
                    plt.text(x[i], y_min[i] - 120, diff_text, ha="center", va="center", size=15)
            elif prs_idx == 1:
                if i == 0:
                    plt.text(x[i], prs[i] + 570, diff_text, ha="center", va="center", size=15, c="green")
                else:
                    plt.text(x[i], prs[i], diff_text, ha="center", va="center", size=15)
            elif prs_idx == 2:
                if i == 0:
                    plt.text(x[i], prs[i] + 600, diff_text, ha="center", va="center", size=15, c="red")
                else:
                    plt.text(x[i], prs[i], diff_text, ha="center", va="center", size=15)
        else:
            plt.text(x[i], prs[i], diff_text, ha="center", va="center", size=15)


def json2excel(path):
    with open(path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    # 将 DataFrame 写入 Excel 文件
    df.to_excel('output.xlsx')


if __name__ == "__main__":
    path = r"D:\Download\result\res\score\24brainSectionScore.xlsx"
    data = {}
    for i in range(24):
        plt.subplot(6, 4, i + 1)
        plot_mean_weighted_curve(path, i, data)
    plt.tight_layout()
    plt.savefig('./score_no_text.jpg', dpi=300)
    plt.show()
    # with open('./data.json', "w") as f:
    #     json.dump(data, f, indent=4)
    # json2excel('./data.json')
