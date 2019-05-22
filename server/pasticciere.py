import tkinter as tk          # для рисования графики
from tkinter import messagebox as mb
from tkinter import Toplevel
from tkinter import filedialog
# import subprocess           # для запуска подпроцессов
# import sys
import os
# import server
# from ipaddr import IPv4Address, IPNetwork
# Начало программы
import dxf2gcode
from scanner import scan
import configparser


def get_config(path):
    """
    Выбираем файл настроек
    """

    config = configparser.ConfigParser()
    config.read(path)
    return config


def update_setting(path, section, setting, value):
    """
    Обновляем параметр в настройках
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)


def get_setting(path, section, setting):
    """
    Выводим значение из настроек
    """
    config = get_config(path)
    value = config.get(section, setting)
    msg = "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value
    )
    print(msg)
    return value


path = "settings.ini"


window = tk.Tk()

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
    update_setting(path, "GCoder", "pointcloudpath", pathToPLY)
    update_setting(path, "GCoder", "dxfpath", pathToDxf)
    dxf2gcode.dxf2gcode(get_setting(path, "GCoder", "dxfpath"), get_setting(path, "GCoder", "pointcloudpath"))


def gcodesetdiag():

    def gcodeset():
        gcodesettings = open("accur.txt", "w")
        gcodesettings.write(stepform.get())
        update_setting(path, "GCoder", "accuracy", stepform.get())
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
    pathToVideo = filedialog.askopenfilename(title="Видео для задания рельефа")
    update_setting(path, "GCoder", "videoforpointcloud", pathToVideo)
    scan(get_setting(path, "GCoder", "videoforpointcloud"))


def getstatus():
    statuswin = Toplevel(window)
    statuswin.title("Текущие параметры")
    statuswin.minsize(width=400, height=400)
    steplabel = tk.Label(statuswin, text="Размер шага:")
    steplabel.grid(row=0, column=0)
    stepvalue = tk.Label(statuswin, text=get_setting(path, "GCoder", "accuracy"))
    stepvalue.grid(row=1, column=0)
    stepbutton = tk.Button(statuswin, text="Задать", command=statuswin.destroy)
    stepbutton.grid(row=2, column=0)


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
lstat = tk.Button(window, text="Статус", command=getstatus)
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
