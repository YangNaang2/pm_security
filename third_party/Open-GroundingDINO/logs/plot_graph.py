import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.use('Agg')

epochs = list(range(15))
train_loss = [43.6547, 38.7042, 36.6193, 34.5717, 32.1582, 31.6135, 30.9209, 30.7558, 30.2526, 30.1406, 30.0898, 30.2461, 30.1864, 30.5137, 30.1799]
map_50_95 = [0.202, 0.229, 0.191, 0.255, 0.284, 0.242, 0.295, 0.239, 0.284, 0.296, 0.290, 0.297, 0.300, 0.288, 0.293]
map_50 = [0.407, 0.459, 0.389, 0.463, 0.563, 0.509, 0.564, 0.504, 0.566, 0.579, 0.571, 0.581, 0.581, 0.572, 0.580]

sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(11, 6))

color_loss = '#E63946'
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Training Loss', color=color_loss, fontsize=12, fontweight='bold')
line1 = ax1.plot(epochs, train_loss, color=color_loss, marker='o', linewidth=2.5, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_xticks(epochs)

ax2 = ax1.twinx()
color_map50 = '#1D3557'
color_map95 = '#457B9D'
ax2.set_ylabel('Validation mAP', color='#1D3557', fontsize=12, fontweight='bold')
line2 = ax2.plot(epochs, map_50, color=color_map50, marker='s', linestyle='--', linewidth=2, label='mAP @ IoU=0.50')
line3 = ax2.plot(epochs, map_50_95, color=color_map95, marker='^', linestyle=':', linewidth=2, label='mAP @ IoU=0.50:0.95')
ax2.tick_params(axis='y', labelcolor=color_map50)

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', edgecolor='none')

plt.title('Open-GroundingDINO Training Progress (Total Time: 1h 10m)', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

output_path = './training_progress_graph.png'
plt.savefig(output_path, dpi=300)
print(f"🎉 graph saved: {output_path}")