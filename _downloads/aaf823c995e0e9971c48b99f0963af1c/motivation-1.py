import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tme import Density
from tme.cli import match_template

target = Density.from_file("../../_static/examples/preprocessing_target.png").data
template = Density.from_file("../../_static/examples/preprocessing_template.png").data

result = match_template(target, template)[0]
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.imshow(result)
ax.set_title('Template Matching Score', color='#0a7d91')

square_size = max(template.shape)
rect = patches.Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size, edgecolor='red', facecolor='none')
ax.add_patch(rect)

plt.tight_layout()
plt.show()