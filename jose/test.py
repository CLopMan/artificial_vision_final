from neural_style import *
import time
import os

matplotlib.rcParams['figure.dpi'] = 300

root = os.path.join(os.path.dirname(os.getcwd()), "cesar", "neural_style", "images")
content = Image(path=os.path.join(root, "photorealistic", "style", "tar3.png"))
#style   = Image(path=os.path.join(root, "manga", "city-1.jpg"))
style   = Image(path=os.path.join("cosas", "trails-1.jpg"))

steps = 10
step = 4
output, matrix = apply_and_reverse_matrix(content, style, steps, step)

stamp = str(time.time()).replace(".", "-")
show_matrix(matrix, steps, step)
matplotlib.pyplot.savefig(os.path.join("results", "%s-matrix.png" % stamp))
output.show()
matplotlib.pyplot.savefig(os.path.join("results", "%s-output.png" % stamp))
matplotlib.pyplot.show()
