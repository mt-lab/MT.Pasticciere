import tkinter as tk
from tkinter import messagebox as mb
from tkinter import Toplevel
from tkinter import filedialog
import os
import dxf2gcode
import scanner
import logging

window = tk.Tk()
logging.basicConfig(filename="pasticciere.log", level=logging.INFO)
# Перечень функций


def mask():
    os.system('python3 manmask.py')


def mancompare():
    os.system('python3 mancompare.py')
    with open("mancompare.txt", "r") as check:
        checkstr = check.readline()
    if checkstr == "no":
        mb.showerror("Внимание!", "Печать с браком!")


def gcoder():
    pathToPLY = filedialog.askopenfilename(title="Облако точек")
    pathToDxf = filedialog.askopenfilename(title="Файл dxf с заданием")
    dxf2gcode.dxf2gcode(pathToDxf, pathToPLY)


def gcodesetdiag():

    def gcodeset():
        gcodesettings = open("accur.txt", "w")
        gcodesettings.write(stepform.get())
        gcodesettings.close
        gcodesetwin.destroy()

    gcodesetwin = Toplevel(window)
    gcodesetwin.title("Настройка gcode")
    gcodesetwin.minsize(width=180, height=100)
    steplabel = tk.Label(gcodesetwin, text="Размер шага")
    steplabel.grid(row=0, column=0)
    stepform = tk.Entry(gcodesetwin)
    stepform.grid(row=1, column=0)
    stepbutton = tk.Button(gcodesetwin, text="Задать", command=gcodeset)
    stepbutton.grid(row=2, column=0)


def pointcloud():
    pathToVideo = filedialog.askopenfilename(title="Видео для рельефа")
    scanner.scan(pathToVideo)


# def scannet():
#    clients = 0
#    for a in IPNetwork('192.168.0.0/28').iterhosts():
#        clients = clients + 1

# Конец перечня функций

# Начало отрисовки интерфейса

# scannet = Button(text="Сканировать сеть", command = scannet)
# clients = 5
# Блок генератора интерфейса для одного устройства


lname = tk.Label(window, height=1, text="Номер устройства")
lname.grid(row=1, column=0)
lstat = tk.Label(window, text="Статус")
lstat.grid(row=1, column=1)
ltask = tk.Label(window, text="Задание")
ltask.grid(row=1, column=2)
ref = tk.Button(text="Обновить")
ref.grid(row=1, column=3)
cl_v = tk.Label(window, text="Версия клиента")
cl_v.grid(row=1, column=4)
home = tk.Button(text="Домой")
home.grid(row=1, column=5)
camshot = tk.Button(text="Снимок")
camshot.grid(row=1, column=6)
manmask = tk.Button(text="Маска", command=mask)
manmask.grid(row=1, column=7)
fullstop = tk.Button(text="СТОП")
fullstop.grid(row=1, column=8)
mancompare = tk.Button(text="Сравнить", command=mancompare)
mancompare.grid(row=1, column=9)
gcoderb = tk.Button(window, text="Генерация gcode", command=gcoder)
gcoderb.grid(row=1, column=10)
gcodesetb = tk.Button(window, text="Параметры gcode", command=gcodesetdiag)
gcodesetb.grid(row=1, column=11)
pointcloudb = tk.Button(window, text="Опознание рельефа", command=pointcloud)
pointcloudb.grid(row=1, column=12)

# Конец блока генератора

# Конец отрисовки интерфейса

# Параметры окна
window['bg'] = 'gray22'
icon = tk.Image("photo", file="icon.gif")
window.tk.call('wm', 'iconphoto', window._w, icon)
window.title('MT.Pasticciere 0.1 (Black Badger)')
window.geometry("1280x400+10+10")
menu = tk.Menu(window)
menu.add_command(label='Обновить')
window.config(menu=menu)
window.mainloop()
