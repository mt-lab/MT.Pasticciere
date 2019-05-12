import tkinter as tk
import os
window = tk.Tk()
window.title("Скрипт настройки печатающего устройства")
window.geometry('400x80')
warn = tk.Label(window, text="!!!ВНИМАНИЕ!!!")
warn.grid(column=0, row=0)
warn1 = tk.Label(window, text="Закройте окно, если не понимаете, что делаете!")
warn1.grid(column=0, row=1)
ipdial = tk.Label(window, text="IP этой машины:")
ipdial.grid(column=0, row=2)
ip = tk.Entry(window, width=10)
ip.grid(column=0, row=3)


def set():
    os.system('echo "nodhcp" >> /etc/dhcpd.conf')
    os.system('echo -e "interface eth0 \nstatic ip_address= + ip.get()" >> /etc/dhcpd.conf')
    os.system('echo -e "static routers=192.168.0.1static \ndomain_name_servers=192.168.0.1" >> /etc/dhcpd.conf')


btn = tk.Button(window, text="Задать", command=set)
btn.grid(column=0, row=4)
window.mainloop()
