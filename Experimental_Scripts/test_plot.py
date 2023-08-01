
import matplotlib.pyplot as plt
import numpy as np

species = ("AveragePooling2D", "MAXPooling2D", "MINPooling2D")
penguin_means = {
    'MNIST': (97.35, 95.43, 98.98),
    'MRI-BATCH(15)': (67.79, 77.83, 60.50),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accurasy')
ax.set_title('Gaussian Input: Trained Network Performance')
ax.set_xticks(x + width, species)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 100)

plt.show()
