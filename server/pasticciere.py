import tkinter as tk          # для рисования графики
from tkinter import filedialog
# import subprocess           # для запуска подпроцессов
# import sys
import os
# import server
# from ipaddr import IPv4Address, IPNetwork
# Начало программы

window = tk.Tk()
window.config(bg="white")

# Перечень функций


def mask():
    os.system('python3 manmask.py')


def mancompare():
    os.system('python3 mancompare.py')


def gcoder():
    file = tk.filedialog.askopenfilename()


# def scannet():
#    clients = 0
#    for a in IPNetwork('192.168.0.0/28').iterhosts():
#        clients = clients + 1

# Конец перечня функций

# Начало отрисовки интерфейса

# scannet = Button(text="Сканировать сеть", command = scannet)
# clients = 5
# Блок генератора интерфейса для одного устройства


lname = tk.Label(window, text="Номер устройства")
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
gcoderb = tk.Button(window, text="Загрузить gcode", command=gcoder)
gcoderb.grid(row=1, column=10)


# Конец блока генератора

# Конец отрисовки интерфейса

# Параметры окна

window.title('MT.Pasticciere ver 0.012 (Athletic Alcoholic)')
window.geometry("800x400+10+10")
menu = tk.Menu(window)
menu.add_command(label='Обновить')
window.config(menu=menu)
window.mainloop()
