from neural_style import *
import time
import os
import sys

def main (style_path, steps, step):
    matplotlib.rcParams['figure.dpi'] = 300

    root = os.path.join(os.path.dirname(os.getcwd()), "cesar", "neural_style", "images")
    content = Image(path=os.path.join(root, "photorealistic", "style", "tar3.png"))
    style = Image(path=style_path)
    #style   = Image(path=os.path.join(root, "manga", "city-1.jpg"))
    #style   = Image(path=os.path.join("cosas", "trails-1.jpg"))

    output, matrix = apply_and_reverse_matrix(content, style, steps, step)

    stamp = str(time.time()).replace(".", "-")
    stamp += "-" + os.path.basename(style_path).split(".")[0]
    show_matrix(matrix, steps, step)
    matplotlib.pyplot.savefig(os.path.join("results", "%s-matrix.png" % stamp))
    output.show()
    matplotlib.pyplot.savefig(os.path.join("results", "%s-output.png" % stamp))
    matplotlib.pyplot.show()

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
