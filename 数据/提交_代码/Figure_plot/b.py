from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import json

with open("b2.json") as f:
    stat_data = json.load(f)

# temp_data = sorted(stat_data.items(), key=lambda kv: (kv[1]["STD B"], kv[0]))
# stat_data = dict(temp_data)
# print(stat_data)

plt.rcParams['figure.figsize'] = (16, 20)
plt.rc('font', family='Times New Roman', size=30)
del matplotlib.font_manager.weight_dict['roman']

a = 0.05
cor_a = 0.05/73
y_labels = []
y_ord = 1
for _, var in stat_data.items():
    for lbl, para in var.items():
        y_labels.append(lbl)
        if para['p'] < cor_a:
            color = 'r'
        elif para['p'] < a:
            color = 'g'
        else:
            color = 'gray'
        plt.scatter(para["std_b"], y_ord, color=color, zorder=3, s=300)
        plt.plot([para["lower"], para["upper"]], [y_ord, y_ord], color=color, linewidth=2.0, zorder=3)
        y_ord += 1

plt.axvline(x=0, ls='--', c='k', linewidth=2.0, zorder=2)
plt.yticks(np.arange(1, y_ord), y_labels)
plt.ylim((0, y_ord))
plt.grid(zorder=1)
plt.gca().set_facecolor('#ffffff')
plt.tight_layout()

plt.savefig('b2.jpg', dpi=600)
# plt.show()
