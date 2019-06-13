import tkinter as tk          # для рисования графики
from tkinter import messagebox as mb
from tkinter import Toplevel
from tkinter import filedialog
from scanner import scan
import configparser
import logging
import otk
import dxf2gcode
import paramiko
import time

logger = logging.getLogger("pasticciere")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("pasticciere.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("Запуск программы")


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


def getOtk():
    host = get_setting(path, "network", "ip1")
    port = 22
    sshUsername1 = get_setting(path, "network", "user1")
    sshPassword1 = get_setting(path, "network", "pass1")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    channel = client.get_transport().open_session()
    channel.get_pty()
    channel.settimeout(5)
    client.exec_command('ffmpeg -y -f video4linux2 -s hd720 -i /dev/video0 \
                        -vframes 1 -f image2 otk.jpg')
    time.sleep(1)
    channel.close()
    client.close()
    transport = paramiko.Transport((host, port))
    transport.connect(username=sshUsername1, password=sshPassword1)
    sftp = paramiko.SFTPClient.from_transport(transport)
    remotepath = 'otk.jpg'
    localpath = 'otk.jpg'
    sftp.get(remotepath, localpath)
    sftp.put(localpath, remotepath)
    sftp.close()
    transport.close()


def getScan():
    host = get_setting(path, "network", "ip1")
    port = 22
    sshUsername1 = get_setting(path, "network", "user1")
    sshPassword1 = get_setting(path, "network", "pass1")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    channel = client.get_transport().open_session()
    channel.get_pty()
    channel.settimeout(5)
    client.exec_command('ffmpeg -y -f video4linux2 -r 15 -s 640x480 \
                        -i /dev/video2 -t 00:00:20 -vcodec mpeg4 \
                        -y scanner.mp4')

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    console = client.invoke_shell()
    print(console)
    console.keep_this = client

    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(2)
    console.send('home\n')
    time.sleep(2)
    console.send('load scanner.gcode\n')
    time.sleep(1)
    console.send('print\n')
    time.sleep(7)
    console.send('exit\n')
    time.sleep(17)
    channel.close()
    client.close()
    transport = paramiko.Transport((host, port))
    transport.connect(username=sshUsername1, password=sshPassword1)
    sftp = paramiko.SFTPClient.from_transport(transport)
    remotepath = 'scanner.mp4'
    localpath = 'scanner.mp4'
    sftp.get(remotepath, localpath)
    sftp.put(localpath, remotepath)
    sftp.close()
    transport.close()


def mancomparing():
    thresh = get_setting(path, "OTK", "threshlevel")
    otk.mancompare(thresh)
    with open("mancompare.txt", "r") as check:
        checkstr = check.readline()
    if checkstr == "no":
        mb.showerror("Внимание!", "Печать с браком!")
        logger.info("Печать с браком!")


def gcoder():
    pathToPLY = filedialog.askopenfilename(title="Облако точек")
    pathToDxf = filedialog.askopenfilename(title="Файл dxf с заданием")
    update_setting(path, "GCoder", "pointcloudpath", pathToPLY)
    update_setting(path, "GCoder", "dxfpath", pathToDxf)
    dxf = get_setting(path, "GCoder", "dxfpath")
    ply = get_setting(path, "GCoder", "pointcloudpath")
    logger.info("Начало генерации g-code")
    dxf2gcode.dxf2gcode(dxf, ply)
    logger.info("Gcode получен")


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
    stepvalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                     "accuracy"))
    stepvalue.grid(row=1, column=0)
    videolabel = tk.Label(statuswin, text="Путь к файлу видео:")
    videolabel.grid(row=2, column=0)
    videovalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                      "videoforpointcloud"))
    videovalue.grid(row=3, column=0)
    cloudlabel = tk.Label(statuswin, text="Путь к облаку точек:")
    cloudlabel.grid(row=4, column=0)
    cloudvalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                      "pointcloudpath"))
    cloudvalue.grid(row=5, column=0)
    dxflabel = tk.Label(statuswin, text="Путь к dxf-файлу:")
    dxflabel.grid(row=6, column=0)
    dxfvalue = tk.Label(statuswin, text=get_setting(path, "GCoder", "dxfpath"))
    dxfvalue.grid(row=7, column=0)
    stepbutton = tk.Button(statuswin, text="Ok", command=statuswin.destroy)
    stepbutton.grid(row=8, column=0)


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
camshot = tk.Button(text="Снимок", command=getOtk)
camshot.grid(row=1, column=6)
camshot = tk.Button(text="Скан", command=getScan)
camshot.grid(row=1, column=7)
manmask = tk.Button(text="Маска", command=otk.manmask)
manmask.grid(row=1, column=8)
fullstop = tk.Button(text="СТОП")
fullstop.grid(row=1, column=9)
mancomparing = tk.Button(text="Сравнить", command=mancomparing)
mancomparing.grid(row=1, column=10)
gcoderb = tk.Button(window, text="Генерация gcode", command=gcoder)
gcoderb.grid(row=1, column=11)
gcodesetb = tk.Button(window, text="Параметры gcode", command=gcodesetdiag)
gcodesetb.grid(row=1, column=12)
pointcloudb = tk.Button(window, text="Опознание рельефа", command=pointcloud)
pointcloudb.grid(row=1, column=13)


# Конец отрисовки интерфейса

# Параметры окна
window['bg'] = 'gray22'
icon = tk.Image("photo", file="icon.gif")
window.tk.call('wm', 'iconphoto', window._w, icon)
window.title('MT.Pasticciere v.0.2 (Cool Cactus)')
window.geometry("1280x400+10+10")
menu = tk.Menu(window)
menu.add_command(label='Обновить')
window.config(menu=menu)
window.mainloop()
