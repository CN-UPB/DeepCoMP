# dashboard test with matplotlib: https://matplotlib.org/stable/tutorials/intermediate/gridspec.html

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from deepcomp.env.entities.map import Map


map = Map(250, 250)

fig = plt.figure(constrained_layout=True, figsize=map.dashboard_figsize)
# fig.suptitle('DeepCoMP Dashboard')
# fig.subplots_adjust(top=0.88)
gs = fig.add_gridspec(4, 3)

ax1 = fig.add_subplot(gs[:, :2])
ax1.set_title('Map View')

ax_text = fig.add_subplot(gs[0, 2])
ax_text.axis('off')
# info text: https://matplotlib.org/stable/gallery/recipes/placing_text_boxes.html
text_str = '\n'.join((
    "Agent: DeepCoMP",
    "Time: 4",
    "Current Total Data Rate: 25 GB/s",
    "Current Total QoE: 8",
    "Avg. Total QoE: 9"
))
text_box = dict(boxstyle='round', facecolor='lightgrey', alpha=0.5)
# ax_text.text(-0.1, 1, text_str, fontsize=13, verticalalignment='top', bbox=text_box)
# or table: https://matplotlib.org/stable/api/table_api.html
text_table = [
    ['Agent', 'DeepCoMP'],
    ['Time', '4'],
    ['Curr. Total Rate', '25 GB/s'],
    ['Curr. Total QoE', '8'],
    ['Avg. Total QoE', '9']
]
table = ax_text.table(cellText=text_table, cellLoc='left', edges='open', loc='upper center')
table.auto_set_font_size(False)
table.set_fontsize(12)

ax2 = fig.add_subplot(gs[1, 2])
ax2.set_title('Global Stats')

ax3 = fig.add_subplot(gs[2, 2])
ax3.set_title('UE 1 Stats')
ax4 = fig.add_subplot(gs[3, 2], sharex=ax3)
ax4.set_title('UE 2 Stats')
# ax5 = fig.add_subplot(gs[4, 2], sharex=ax4)
# ax5.set_title('UE 3 Stats')

plt.show()