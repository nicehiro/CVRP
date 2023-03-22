import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import animation

mpl.rcParams[
    "animation.ffmpeg_path"
] = r"D:\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"

# 创建一个画布和一个子图
fig, ax = plt.subplots()


# 定义一个函数来生成要绘制的数据
def generate_data():
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    return x, y


# 定义一个函数来更新图形
def update(frame):
    x, y = generate_data()
    ax.clear()
    ax.plot(x, y)
    ax.set_title("Frame {}".format(frame))


# 创建一个动画对象
ani = FuncAnimation(fig, update, frames=10, repeat=False)


f = r"C:\Users\zhihui\Desktop\imation.mp4"
writermp4 = animation.FFMpegWriter(fps=60)
ani.save(f, writer=writermp4)
# 显示动画

plt.show()
