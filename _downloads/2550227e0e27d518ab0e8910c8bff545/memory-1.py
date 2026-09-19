import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from tme.memory import compute_schedule

mask = np.zeros((256, 256), dtype=bool)
mask[32:96, 48:112] = True      # 64×64 region
mask[128:192, 176:240] = True   # 64×64 region
mask[96:128, 128:192] = True    # 32×64 region
mask[192:224, 64:128] = True    # 32×64 region

y, x = np.ogrid[:256, :256]
circle1 = (x - 224)**2 + (y - 64)**2 <= 22**2
circle2 = (x - 160)**2 + (y - 208)**2 <= 18**2
mask[circle1] = True
mask[circle2] = True

# Compute schedules for both modes
boxes_uniform, schedule = compute_schedule(
   shape=mask.shape,
   mode="uniform",
   max_memory=1e10,
   max_workers=4,
   matching_method="CC",
)

boxes_masked, schedule = compute_schedule(
   shape=mask.shape,
   mode="subdivide",
   mask=mask,
   min_box_size=16,
   padding=(8, 8),
   max_memory=1e10,
   max_workers=1,
   matching_method="CC",
   verbose=True,
   min_improvement=1.0,
   n_sat=64,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
colors_uniform = plt.cm.Set3(np.linspace(0, 1, len(boxes_uniform)))
colors_masked = plt.cm.Set3(np.linspace(0, 1, len(boxes_masked)))

# Uniform mode
ax1.imshow(mask, cmap="gray", alpha=0.3, origin="upper",
    extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])
for i, box in enumerate(boxes_uniform):
   y, x = box
   rect = patches.Rectangle(
       (x.start -0.5, y.start - 0.5),
       x.stop - x.start, y.stop - y.start,
       linewidth=2.5, edgecolor=colors_uniform[i],
       facecolor=colors_uniform[i], alpha=0.25
   )
   ax1.add_patch(rect)
ax1.set_title("Uniform", fontsize=14)
ax1.axis("off")

# Masked mode
ax2.imshow(mask, cmap="gray", alpha=0.3, origin="upper",
    extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])
for i, box in enumerate(boxes_masked):
   y, x = box
   rect = patches.Rectangle(
       (x.start - 0.5, y.start - 0.5),
       x.stop - x.start, y.stop - y.start,
       linewidth=2.5, edgecolor=colors_masked[i],
       facecolor=colors_masked[i], alpha=0.25
   )
   ax2.add_patch(rect)
ax2.set_title("Masked", fontsize=14)
ax2.axis("off")

plt.tight_layout()
plt.show()