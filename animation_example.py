"""
=========================
Simple animation examples
=========================

Two animations where the first is a random walk plot and
the second is an image animation.
"""
#
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
#
# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,
#
# ###############################################################################
#
# fig1 = plt.figure()
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)
#
# # To save the animation, use the command: line_ani.save('lines.mp4')
#
# ###############################################################################
#
# fig2 = plt.figure()
#
# x = np.arange(-9, 10)
# y = np.arange(-9, 10).reshape(-1, 1)
# base = np.hypot(x, y)
# ims = []
# for add in np.arange(15):
#     ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))
#
# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
#                                    blit=True)

####

# TODO: use this to save a html5 replay when running the simulation (not training)


def my_plot3():
    my_fig = plt.figure()
    plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)], figure=my_fig)
    plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)], figure=my_fig)
    return my_fig

def my_plot2():
    fig, ax = plt.subplots()
    ax.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    ax.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    plt.show()
    return ax

def my_plot():
    # fig, ax = plt.subplots()
    # my_fig = plt.figure()
    patch = []
    patch.extend(plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)]))
    patch.extend(plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)]))
    # plt.show()
    return patch

def my_plot4():
    p0, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    p1, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    # important: no plt.show()!
    return [p0, p1]   # return a list of the new plots

# fig, ax = plt.subplots()    # fig and axes created once
fig = plt.figure()
ims = []
for _ in range(5):
    patch = my_plot()
    ims.append(patch)
    # ax = my_plot2()
    # ims.append((ax,))
    # my_fig = my_plot3()
    # ims.append((my_fig,))
print(ims)
ani = animation.ArtistAnimation(fig, ims, repeat=False)

# To save this second animation with some metadata, use the following command:
# ani.save('im.mp4', metadata={'artist':'Guido'})
html = ani.to_html5_video()
with open('replay.html', 'w') as f:
    f.write(html)
#
# plt.show()
