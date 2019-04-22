from tkinter import *
import subprocess

# Начало отрисовки интерфейса
window=Tk()
window.config(bg="white")


def mask(event):
    cmd = "python ~/Repos/MT.Pasticciere/server/polzunok_threshold.py"
    PIPE = subprocess.PIPE
    p = subprocess.Popen(cmd, shell = True)


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
selfmask = Button(text="Маска")
selfmask.bind("<Button-1>", mask)
selfmask.grid(row=0, column=7)


window.title('MT.Pasticciere ver 0.01 (Beshenaya Lopata)')
window.geometry("600x400+10+10")
window.mainloop()
# Конец отрисовки интерфейса
