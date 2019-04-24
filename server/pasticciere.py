from tkinter import *       # для рисования графики
import subprocess           # для запуска подпроцессов
import sys
import os

# Начало программы

window = Tk()
window.config(bg="white")

# Перечень функций

def mask():
    os.system('python3 manmask.py')

def mancompare():
    os.system('python3 mancompare.py')

# Конец перечня функций

# Начало отрисовки интерфейса

lname = Label(window, text= "Номер устройства")
lname.grid(row=0, column=0)
lstat = Label(window, text="Статус")
lstat.grid(row=0, column=1)
ltask = Label(window, text="Задание")
ltask.grid(row=0, column=2)
ref = Button(text="Обновить")
ref.grid(row=0, column=3)
cl_v = Label(window, text="Версия клиента")
cl_v.grid(row=0, column=4)
home = Button(text="Домой")
home.grid(row=0, column=5)
camshot = Button(text="Снимок")
camshot.grid(row=0, column=6)
manmask = Button(text="Маска", command = mask)
manmask.grid(row=0, column=7)
fullstop = Button(text="СТОП")
fullstop.grid(row=0, column=8)
mancompare = Button(text="Сравнить", command = mancompare)
mancompare.grid(row=0, column=9)

# Конец отрисовки интерфейса

# Параметры окна

window.title('MT.Pasticciere ver 0.011 (Obossanaya Ogloblya)')
window.geometry("800x400+10+10")
window.mainloop()
