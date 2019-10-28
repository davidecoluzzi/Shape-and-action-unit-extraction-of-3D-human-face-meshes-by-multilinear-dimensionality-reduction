# GRAPHICAL USER INTERFACE

# import of the libraries
import random
import matplotlib
import tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import operator
import numpy as np
from tensorly import fold
import tensorly.tenalg.n_mode_product as n_mode

# Open the files with the mean face, the maximum and the minimum for each unit and the core
meanFace = np.genfromtxt(".\MeanFace.csv", delimiter=',')
maxmin = np.genfromtxt(".\MaxMin.csv", delimiter=',')
core1 = np.genfromtxt(".\Core.csv", delimiter=',')
core = fold(core1, 1, [4041, 8, 8])
meanVec = np.array([])

if __name__ == '__main__':
    x = meanFace[0, :]
    y = meanFace[1, :]

    # save the mean face in an array
    for i in range(0, 1347):
        for j in range(0, 3):
            meanVec = np.append(meanVec, meanFace[j, i])

    # set the interface
    root = Tk.Tk()
    root.wm_title("Multilinear PCA")
    fig = plt.Figure()
    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    ax=fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.4, top=0.99)

    # plot the mean face
    ax.plot(x, y, 'b.')

    # set 16 sliders in the range between the minimum and maximum values previously stored
    ax_Sh1 = fig.add_axes([0.12, 0.3, 0.3, 0.02])
    s_Sh1 = Slider(ax_Sh1, 'SH - w2,1', maxmin[0][1], maxmin[0][0], valinit=0)

    ax_Sh2 = fig.add_axes([0.12, 0.26, 0.3, 0.02])
    s_Sh2 = Slider(ax_Sh2, 'SH - w2,2', maxmin[0][3], maxmin[0][2], valinit=0)

    ax_Sh3 = fig.add_axes([0.12, 0.22, 0.3, 0.02])
    s_Sh3 = Slider(ax_Sh3, 'SH - w2,3', maxmin[0][5], maxmin[0][4], valinit=0)

    ax_Sh4 = fig.add_axes([0.12, 0.18, 0.3, 0.02])
    s_Sh4 = Slider(ax_Sh4, 'SH - w2,4', maxmin[0][7], maxmin[0][6], valinit=0)

    ax_Sh5 = fig.add_axes([0.12, 0.14, 0.3, 0.02])
    s_Sh5 = Slider(ax_Sh5, 'SH - w2,5', maxmin[0][9], maxmin[0][8], valinit=0)

    ax_Sh6 = fig.add_axes([0.12, 0.10, 0.3, 0.02])
    s_Sh6 = Slider(ax_Sh6, 'SH - w2,6', maxmin[0][11], maxmin[0][10], valinit=0)

    ax_Sh7 = fig.add_axes([0.12, 0.06, 0.3, 0.02])
    s_Sh7 = Slider(ax_Sh7, 'SH - w2,7', maxmin[0][13], maxmin[0][12], valinit=0)

    ax_Sh8 = fig.add_axes([0.12, 0.02, 0.3, 0.02])
    s_Sh8 = Slider(ax_Sh8, 'SH - w2,8', maxmin[0][15], maxmin[0][14], valinit=0)


    ax_Ex1 = fig.add_axes([0.62, 0.3, 0.3, 0.02])
    s_Ex1 = Slider(ax_Ex1, 'EX - w3,1', maxmin[1][1], maxmin[1][0], valinit=0)

    ax_Ex2 = fig.add_axes([0.62, 0.26, 0.3, 0.02])
    s_Ex2 = Slider(ax_Ex2, 'EX - w3,2', maxmin[1][3], maxmin[1][2], valinit=0)

    ax_Ex3 = fig.add_axes([0.62, 0.22, 0.3, 0.02])
    s_Ex3 = Slider(ax_Ex3, 'EX - w3,3', maxmin[1][5], maxmin[1][4], valinit=0)

    ax_Ex4 = fig.add_axes([0.62, 0.18, 0.3, 0.02])
    s_Ex4 = Slider(ax_Ex4, 'EX - w3,4', maxmin[1][7], maxmin[1][6], valinit=0)

    ax_Ex5 = fig.add_axes([0.62, 0.14, 0.3, 0.02])
    s_Ex5 = Slider(ax_Ex5, 'EX - w3,5', maxmin[1][9], maxmin[1][8], valinit=0)

    ax_Ex6 = fig.add_axes([0.62, 0.10, 0.3, 0.02])
    s_Ex6 = Slider(ax_Ex6, 'EX - w3,6', maxmin[1][11], maxmin[1][10], valinit=0)

    ax_Ex7 = fig.add_axes([0.62, 0.06, 0.3, 0.02])
    s_Ex7 = Slider(ax_Ex7, 'EX - w3,7', maxmin[1][13], maxmin[1][12], valinit=0)

    ax_Ex8 = fig.add_axes([0.62, 0.02, 0.3, 0.02])
    s_Ex8 = Slider(ax_Ex8, 'EX - w3,8', maxmin[1][15], maxmin[1][14], valinit=0)


def update(val):
    ax.clear()
    # save the selected values by means of the sliders in 2 arrays of coefficients
    coeffSh = np.array([s_Sh1.val, s_Sh2.val, s_Sh3.val, s_Sh4.val, s_Sh5.val, s_Sh6.val, s_Sh7.val, s_Sh8.val])[np.newaxis]
    coeffEx = np.array([s_Ex1.val, s_Ex2.val, s_Ex3.val, s_Ex4.val, s_Ex5.val, s_Ex6.val, s_Ex7.val, s_Ex8.val])[np.newaxis]

    print (coeffSh)
    print (coeffEx)

    prod2 = (n_mode.mode_dot(core, coeffSh, 1)) # 2-mode product
    prod3 = (n_mode.mode_dot(prod2, coeffEx, 2)) # 3-mode product

    prodFac = np.zeros(4041)
    for i in range(0, 4041):
        prodFac[i] = prod3[i][0][0]

    faccia = meanVec + prodFac  # reconstruction of the face adding the result of the n-mode products to the mean face

    # reshape the obtained face in a matrix of 3x1347
    faccia1 = np.zeros((3, 1347))
    x = 0
    for i in range(0, 4041):
        y = operator.mod(i, 3)
        if (y == 0 and i != 0):
            x = x + 1;
        faccia1[y, x] = faccia[i]

    x = faccia1[0, :]
    y = faccia1[1, :]


    ax.plot(x, y, 'b.')
    fig.canvas.draw_idle()


s_Sh1.on_changed(update)
s_Sh2.on_changed(update)
s_Sh3.on_changed(update)
s_Sh4.on_changed(update)
s_Sh5.on_changed(update)
s_Sh6.on_changed(update)
s_Sh7.on_changed(update)
s_Sh8.on_changed(update)

s_Ex1.on_changed(update)
s_Ex2.on_changed(update)
s_Ex3.on_changed(update)
s_Ex4.on_changed(update)
s_Ex5.on_changed(update)
s_Ex6.on_changed(update)
s_Ex7.on_changed(update)
s_Ex8.on_changed(update)

Tk.mainloop()