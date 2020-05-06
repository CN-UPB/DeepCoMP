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

fig3 = plt.figure()

ims = []
for _ in range(10):
    im, = plt.plot([random.randrange(10), random.randrange(10)], [random.randrange(10), random.randrange(10)])
    ims.append((im,))
ani = animation.ArtistAnimation(fig3, ims)

# To save this second animation with some metadata, use the following command:
# ani.save('im.mp4', metadata={'artist':'Guido'})
html = ani.to_html5_video()
with open('replay.html', 'w') as f:
    f.write(html)

plt.show()
