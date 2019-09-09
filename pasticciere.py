"""
pasticcere.py
Author: Dmitry K of MT.lab
mailto: dmitry@kuprianov.su

Главный файл программы, отвечает за отрисовку интерфейса программы и содержит
функции для передачи данных и команд по сети.
"""

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
    Выбор файла настроек

    path - путь к файлу
    """
    config = configparser.ConfigParser()
    config.read(path)
    return config


def update_setting(path, section, setting, value):
    """
    Обновление параметра в файле настроек

    path - путь к файлу настроек
    section - название секции
    setting - название параметра в секции
    value - значение параметра
    """
    config = get_config(path)
    config.set(section, setting, value)
    with open(path, "w") as config_file:
        config.write(config_file)


def get_setting(path, section, setting):
    """
    Выводим значение из настроек

    path - путь к файлу настроек
    section - название секции
    setting - название параметра в секции
    """
    config = get_config(path)
    value = config.get(section, setting)
    msg = "{section} {setting} is {value}".format(
        section=section, setting=setting, value=value
    )
    logger.info(msg)
    return value


def getFile(host, port, name, password, file):
    """
    Забирает файл с удалённого устройства не меняя имени файла

    host - ip-адрес устройства
    port - порт для соединения с устройством
    name - имя пользователя ssh
    password - пароль пользователя ssh
    file - имя файла на удалённом устройстве
    """
    transport = paramiko.Transport((host, port))
    transport.connect(username=name, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    remotepath = file
    localpath = file
    sftp.get(remotepath, localpath)
    sftp.close()
    transport.close()


def sendFile(host, port, name, password, file):
    """
    Забирает файл с удалённого устройства не меняя имени файла

    host - ip-адрес устройства
    port - порт для соединения с устройством
    name - имя пользователя ssh
    password - пароль пользователя ssh
    file - имя файла на удалённом устройстве
    """
    transport = paramiko.Transport((host, port))
    transport.connect(username=name, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    remotepath = file
    localpath = file
    sftp.put(remotepath, localpath)
    sftp.close()
    transport.close()


path = "settings.ini"


window = tk.Tk()


# Перечень функций

def getHome():
    """
    Отправка бункера в дом
    """
    host = get_setting(path, "network", "ip1")
    port = 22
    sshUsername1 = get_setting(path, "network", "user1")
    sshPassword1 = get_setting(path, "network", "pass1")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    console = client.invoke_shell()
    console.keep_this = client
    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(2)
    console.send('home\n')
    print("Подана команда домой")
    logger.info("Подана команда домой")
    time.sleep(1)
    console.send('exit\n')
    client.close()


def getOtk():
    """
    Фотографирование и получение файла для проведения контроля качества
    """
    print("Запрос на получение снимка")
    logger.info("Запрос на получение снимка")
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
    client.exec_command(r'ffmpeg -y -f video4linux2 -s hd720 -i /dev/video0 \
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
    print("Снимок получен")
    logger.info("Снимок получен")


def stop():
    """
    Аварийная остановка устройства (тестовый функционал, скорей всего работать
    не будет)
    """
    host = get_setting(path, "network", "ip1")
    port = 22
    sshUsername1 = get_setting(path, "network", "user1")
    sshPassword1 = get_setting(path, "network", "pass1")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    console = client.invoke_shell()
    console.keep_this = client
    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(2)
    console.send('M112\n')
    time.sleep(1)
    console.send('exit\n')
    client.close()


def getScan():
    """
    Проведение сканирования на удалённом устройстве и получение видеофайла
    """
    logger.info("Начало сканирования")
    print("Начало сканирования")
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
    client.exec_command(r'v4l2-ctl -d /dev/video2 \
                         --set-ctrl=white_balance_temperature_auto=0')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=focus_auto=0')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=9')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=brightness=30')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=contrast=7')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=saturation=0')
    client.exec_command(r'v4l2-ctl -d /dev/video2 \
                         --set-ctrl=white_balance_temperature=6166')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=exposure_auto=1')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=exposure_absolute=9')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-ctrl=sharpness=0')
    client.exec_command('v4l2-ctl -d /dev/video2 -p 30')
    client.exec_command('v4l2-ctl -d /dev/video2 --set-fmt-video=width=640,height=480')

    console = client.invoke_shell()
    console.keep_this = client
    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(1)
    console.send('home\n')
    time.sleep(1)
    console.send('G0 F2000 Z33\n')
    time.sleep(1)
    console.send('G0 X150 Y90.5\n')
    time.sleep(5)
    client.exec_command(r'ffmpeg -y -f video4linux2 -r 30 -video_size 640x480 \
                        -i /dev/video2 -t 00:00:30 -vcodec mpeg4 \
                        -y scanner.mp4')
    time.sleep(3)
    console.send('G0 F300 X286\n')
    time.sleep(20)
    console.send('G0 F2000\n')
    console.send('home\n')
    console.send('exit\n')
    time.sleep(15)
    channel.close()
    client.close()
    getFile(host, port, sshUsername1, sshPassword1, 'scanner.mp4')
    print("Получен файл с камеры")
    logger.info("Получен файл с камеры")


def getPrinted():
    print("Отправка задания на печать")
    logger.info("Отправка задания на печать")
    host = get_setting(path, "network", "ip1")
    port = 22
    sshUsername1 = get_setting(path, "network", "user1")
    sshPassword1 = get_setting(path, "network", "pass1")
    sendFile(host, port, sshUsername1, sshPassword1, "cookie.gcode")
    time.sleep(5)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=sshUsername1, password=sshPassword1,
                   port=port)
    channel = client.get_transport().open_session()
    channel.get_pty()
    channel.settimeout(5)
    console = client.invoke_shell()
    console.keep_this = client
    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(1)
    console.send('home\n')
    time.sleep(5)
    console.send('load cookie.gcode\n')
    time.sleep(1)
    console.send('print\n')
    time.sleep(60)
    console.send('exit\n')
    channel.close()
    client.close()
    print("Печать выполнена")
    logger.info("Печать выполнена")


def mancomparing():
    """
    Сравнение изображения печенья с эталоном (плохая функция надо переписать)
    """
    check = otk.mancompare(get_setting(path, "OTK", "threshlevel"))
    if check == "no":
        mb.showerror("Внимание!", "Печать с браком!")
        logger.error("Печать с браком!")


def gcoder():
    """
    Функция генерации gcode из заданных файлов облака точек и dxf
    Вызывает два диалоговых окна для задания путей к этим файлам
    """
    pathToPLY = filedialog.askopenfilename(title="Облако точек")
    pathToDxf = filedialog.askopenfilename(title="Файл dxf с заданием")
    update_setting(path, "GCoder", "pointcloudpath", pathToPLY)
    update_setting(path, "GCoder", "dxfpath", pathToDxf)
    dxf = get_setting(path, "GCoder", "dxfpath")
    ply = get_setting(path, "GCoder", "pointcloudpath")
    window.update()
    logger.info("Начало генерации g-code")
    dxf2gcode.dxf2gcode(dxf, ply)
    logger.info("Gcode получен")


def gcodesetdiag():
    """
    Задание точности построения gcode (минимальной длины линии)
    Содержит вложенную функцию записи значения в конфигурационный файл
    """
    def gcodeset():
        update_setting(path, "GCoder", "accuracy", stepform.get())
        gcodesetwin.destroy()

    gcodesetwin = Toplevel(window)
    gcodesetwin.title("Настройка gcode")
    gcodesetwin.minsize(width=180, height=100)
    steplabel = tk.Label(gcodesetwin, text="Размер шага (мм)")
    steplabel.grid(row=0, column=0)
    stepform = tk.Entry(gcodesetwin)
    stepform.grid(row=1, column=0)
    stepbutton = tk.Button(gcodesetwin, text="Задать", command=gcodeset)
    stepbutton.grid(row=2, column=0)


def addressCutter(address):
    cuttedAddress = address[-15:]
    return cuttedAddress


def pointcloud():
    """
    Вызывается диалоговое окно с указанием пути к видео для генерации облака
    """
    pathToVideo = filedialog.askopenfilename(title="Видео для задания рельефа")
    update_setting(path, "GCoder", "videoforpointcloud", pathToVideo)
    scan(get_setting(path, "GCoder", "videoforpointcloud"))


def getstatus():
    """
    Отрисовка окна статуса, где выводятся параметры из файла настроек
    """
    statuswin = Toplevel(window)
    statuswin.title("Текущие параметры")
    statuswin.minsize(width=400, height=400)
    steplabel = tk.Label(statuswin, text="Размер шага:")
    steplabel.grid(row=0, column=0)
    stepvalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                     "accuracy")+" мм")
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


lname = tk.Label(window, height=1, text=get_setting(path, "network", "ip1"))
lname.grid(row=1, column=0)
lstat = tk.Button(window, text="Статус", command=getstatus)
lstat.grid(row=1, column=1)
ltask = tk.Label(window, text=".." + addressCutter
                 (get_setting(path, "GCoder", "dxfpath")))
ltask.grid(row=1, column=2)
home = tk.Button(text="Домой", command=getHome)
home.grid(row=1, column=3)
camshot = tk.Button(text="Снимок", command=getOtk)
camshot.grid(row=1, column=4)
camshot = tk.Button(text="Скан", command=getScan)
camshot.grid(row=1, column=5)
getMask = tk.Button(text="Маска", command=otk.getMask)
getMask.grid(row=1, column=6)
fullstop = tk.Button(text="СТОП", command=stop)
fullstop.grid(row=1, column=7)
mancomparing = tk.Button(text="Сравнить", command=mancomparing)
mancomparing.grid(row=1, column=8)
gcoderb = tk.Button(window, text="Генерация gcode", command=gcoder)
gcoderb.grid(row=1, column=9)
gcodesetb = tk.Button(window, text="Параметры gcode", command=gcodesetdiag)
gcodesetb.grid(row=1, column=10)
pointcloudb = tk.Button(window, text="Опознание рельефа", command=pointcloud)
pointcloudb.grid(row=1, column=11)
getprinted = tk.Button(window, text="Печать", command=getPrinted)
getprinted.grid(row=1, column=12)


# Конец отрисовки интерфейса

# Параметры окна
window['bg'] = 'gray22'
icon = tk.Image("photo", file="icon.gif")
window.tk.call('wm', 'iconphoto', window._w, icon)
window.title('MT.Pasticciere v.0.2 (Cool Cactus)')
window.geometry("1280x400+10+10")
window.mainloop()
