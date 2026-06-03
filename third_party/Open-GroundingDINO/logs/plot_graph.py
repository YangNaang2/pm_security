import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('Agg')

log_file_path = 'train_open_gdino.log'

epochs = []
train_loss = []
map_50 = []
map_50_95 = []

with open(log_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    if "Averaged stats:" in line and "loss:" in line:
        parts = line.split("loss:")
        if len(parts) > 1:
            loss_part = parts[1].split("(")[1].split(")")[0]
            train_loss.append(float(loss_part.strip()))
            
    if "Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]" in line:
        val = float(line.split('=')[-1].strip())
        map_50_95.append(val)
        
    elif "Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]" in line:
        val = float(line.split('=')[-1].strip())
        map_50.append(val)

num_epochs = len(train_loss)
map_50 = map_50[:num_epochs]
map_50_95 = map_50_95[:num_epochs]

epochs = list(range(num_epochs))

print(f"📊 로그 분석 완료! (Epoch 수: {len(epochs)}, Loss: {len(train_loss)}개, mAP50: {len(map_50)}개, mAP95: {len(map_50_95)}개)")

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

plt.title('Open-GroundingDINO Training', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

output_path = './train_graph.png'
plt.savefig(output_path, dpi=300)
print(f"🎉 graph saved: {output_path}")