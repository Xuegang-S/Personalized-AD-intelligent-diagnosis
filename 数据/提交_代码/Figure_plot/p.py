from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import json

with open("./Hippocampus left single para.json") as f:
    stat_data = json.load(f)

cat_lbl = ["Socioeconomic\nStatus", "Lifestyles", "Local\nEnvironment", "Medical\nHistory", "Physical\nMeasures", "Psychosocial\nFactors"]

color = {
    "Socioeconomic status": '#ef7d00',
    "Lifestyles": '#16a39b',
    "Local environment": '#619a13',
    "Medical history": '#f85e78',
    "Physical measures": '#fa55d8',
    "Psychosocial factors": '#ac79fc'
}

a = 0.05
cor_a = a/73
space = 5
x_ord = 1
category_label_ticks = []
plt.rcParams['figure.figsize'] = (20.2, 12.8)
plt.rc('font', family='Times New Roman', size=20)
del matplotlib.font_manager.weight_dict['roman']
for category, var in stat_data.items():
    start = x_ord
    for lbl, val in var.items():
        log_p = np.log10(1 / val['p'])
        plt.scatter(x_ord, log_p, c=color[category], alpha=0.8, zorder=2, s=155)
        # if log_p > np.log10(1 / a):
        #     print(lbl)
        #     plt.text(x_ord, log_p+0.1, lbl, ha='center')
        x_ord += 1
    category_label_ticks.append(np.mean([start, x_ord]))
    x_ord += space

plt.axhline(y=np.log10(1 / a), ls='-.', c='k', linewidth=3.0, alpha=0.5, zorder=1)
plt.axhline(y=np.log10(1 / cor_a), ls='--', c='k', linewidth=3.0, zorder=1)
plt.xticks(category_label_ticks, cat_lbl)
# plt.xticks(rotation=45)
plt.grid(axis='y', zorder=1, linewidth=2.0)
plt.ylabel('-log10(P)')
# plt.ylim((0, 55))
plt.gca().set_facecolor('#ffffff')
plt.tight_layout()
plt.savefig('./single factor text.jpg', dpi=600)
plt.show()
