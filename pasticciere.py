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
from threading import Thread
import configparser
import logging
import otk
import dxf2gcode
import paramiko
import time
import os

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
    client.exec_command(r'v4l2-ctl -d /dev/scanner \
                         --set-ctrl=white_balance_temperature_auto=0')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=focus_auto=0')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=focus_absolute=9')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=brightness=30')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=contrast=10')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=saturation=0')
    client.exec_command(r'v4l2-ctl -d /dev/scanner \
                         --set-ctrl=white_balance_temperature=2800')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=exposure_auto=1')
    client.exec_command(r'v4l2-ctl -d /dev/scanner \
                         --set-ctrl=exposure_absolute=156')
    client.exec_command('v4l2-ctl -d /dev/scanner --set-ctrl=sharpness=50')
    client.exec_command('v4l2-ctl -d /dev/scanner -p 30')
    client.exec_command(r'v4l2-ctl -d /dev/video2 \
                        --set-fmt-video=width=640,height=480')

    console = client.invoke_shell()
    console.keep_this = client
    console.send('pronsole\n')
    time.sleep(1)
    console.send('connect\n')
    time.sleep(1)
    console.send('home\n')
    time.sleep(1)
    console.send('G0 F2000 Z46\n')
    time.sleep(2)
    console.send('G0 X200 Y112\n')
    time.sleep(10)
    client.exec_command(r'ffmpeg -y -f video4linux2 -r 30 -video_size 640x480 \
                        -i /dev/scanner -t 00:00:43 -vcodec mpeg4 \
                        -y scanner.mp4')
    time.sleep(3)
    console.send('G0 F300 X0\n')
    time.sleep(30)
    console.send('G0 F2000\n')
    time.sleep(3)
    console.send('G28 Z\n')
    time.sleep(3)
    console.send('home\n')
    console.send('exit\n')
    time.sleep(15)
    channel.close()
    client.close()
    getFile(host, port, sshUsername1, sshPassword1, 'scanner.mp4')
    print("Получен файл с камеры")
    logger.info("Получен файл с камеры")


def getPrinted():

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
    stopwin = Toplevel(window)
    stopwin.title("Остановка")
    stopwin.minsize(width=100, height=100)
    stopButton = tk.Button(stopwin, text="СТОП", command=stop)
    stopButton.grid(row=0, column=0)
    stepbutton = tk.Button(stopwin, text="Закрыть", command=stopwin.destroy)
    stepbutton.grid(row=20, column=0)
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

    def accuracySet():
        update_setting(path, "GCoder", "accuracy", accuracyForm.get())
        gcodesetwin.destroy()

    def extrusionMultiplexSet():
        update_setting(path, 'GCoder', 'slice_step', extrusionMultiplexForm.get())
        gcodesetwin.destroy()

    def sliceStepSet():
        update_setting(path, 'GCoder', 'slice_step', sliceStepForm.get())
        gcodesetwin.destroy()

    def zoffsetSet():
        update_setting(path, "GCoder", "z_offset", zoffsetForm.get())
        gcodesetwin.destroy()

    def extcoeffSet():
        update_setting(path, "GCoder", "extrusion_coefficient",
                       extcoeffForm.get())
        gcodesetwin.destroy()

    def retractSet():
        update_setting(path, "GCoder", "retract_amount", retractForm.get())
        gcodesetwin.destroy()

    def p0Set():
        update_setting(path, "GCoder", "p0", p0Form.get())
        gcodesetwin.destroy()

    def p1Set():
        update_setting(path, "GCoder", "p1", p1Form.get())
        gcodesetwin.destroy()

    def p2Set():
        update_setting(path, "GCoder", "p2", p2Form.get())
        gcodesetwin.destroy()

    gcodesetwin = Toplevel(window)
    gcodesetwin.title("Настройка gcode")
    gcodesetwin.minsize(width=170, height=500)

    accuracyLabel = tk.Label(gcodesetwin, text="Точность (мм):")
    accuracyLabel.grid(row=0, column=0)
    accuracyForm = tk.Entry(gcodesetwin)
    accuracyForm.grid(row=1, column=0)
    accuracyButton = tk.Button(gcodesetwin, text="Задать", command=accuracySet)
    accuracyButton.grid(row=1, column=1)

    zoffsetLabel = tk.Label(gcodesetwin, text="Смещение по Z (мм):")
    zoffsetLabel.grid(row=2, column=0)
    zoffsetForm = tk.Entry(gcodesetwin)
    zoffsetForm.grid(row=3, column=0)
    zoffsetButton = tk.Button(gcodesetwin, text="Задать", command=zoffsetSet)
    zoffsetButton.grid(row=3, column=1)

    extcoeffLabel = tk.Label(gcodesetwin, text="Коэффициент экструзии:")
    extcoeffLabel.grid(row=4, column=0)
    extcoeffForm = tk.Entry(gcodesetwin)
    extcoeffForm.grid(row=5, column=0)
    extcoeffButton = tk.Button(gcodesetwin, text="Задать", command=extcoeffSet)
    extcoeffButton.grid(row=5, column=1)

    retractLabel = tk.Label(gcodesetwin, text="Подсос:")
    retractLabel.grid(row=6, column=0)
    retractForm = tk.Entry(gcodesetwin)
    retractForm.grid(row=7, column=0)
    retractButton = tk.Button(gcodesetwin, text="Задать", command=retractSet)
    retractButton.grid(row=7, column=1)

    p0Label = tk.Label(gcodesetwin, text="P0:")
    p0Label.grid(row=8, column=0)
    p0Form = tk.Entry(gcodesetwin)
    p0Form.grid(row=9, column=0)
    p0Button = tk.Button(gcodesetwin, text="Задать", command=p0Set)
    p0Button.grid(row=9, column=1)

    p1Label = tk.Label(gcodesetwin, text="P1:")
    p1Label.grid(row=10, column=0)
    p1Form = tk.Entry(gcodesetwin)
    p1Form.grid(row=11, column=0)
    p1Button = tk.Button(gcodesetwin, text="Задать", command=p1Set)
    p1Button.grid(row=11, column=1)

    p2Label = tk.Label(gcodesetwin, text="P2:")
    p2Label.grid(row=12, column=0)
    p2Form = tk.Entry(gcodesetwin)
    p2Form.grid(row=13, column=0)
    p2Button = tk.Button(gcodesetwin, text="Задать", command=p2Set)
    p2Button.grid(row=13, column=1)

    sliceStepLabel = tk.Label(gcodesetwin, text="Размер шага (мм):")
    sliceStepLabel.grid(row=14, column=0)
    sliceStepForm = tk.Entry(gcodesetwin)
    sliceStepForm.grid(row=15, column=0)
    sliceStepButton = tk.Button(gcodesetwin, text="Задать", command=sliceStepSet)
    sliceStepButton.grid(row=15, column=1)

    extrusionMultiplexLabel = tk.Label(gcodesetwin, text="Множитель экструзии:")
    extrusionMultiplexLabel.grid(row=14, column=0)
    extrusionMultiplexForm = tk.Entry(gcodesetwin)
    extrusionMultiplexForm.grid(row=15, column=0)
    extrusionMultiplexButton = tk.Button(gcodesetwin, text="Задать", command=extrusionMultiplexSet)
    extrusionMultiplexButton.grid(row=15, column=1)


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
    stepLabel = tk.Label(statuswin, text="Точность:")
    stepLabel.grid(row=0, column=0)
    stepValue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                     "accuracy") + " мм")
    stepValue.grid(row=1, column=0)

    zoffsetLabel = tk.Label(statuswin, text="Смещение по Z:")
    zoffsetLabel.grid(row=2, column=0)
    zoffsetValue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                        "z_offset") + " мм")
    zoffsetValue.grid(row=3, column=0)

    excoefLabel = tk.Label(statuswin, text="Коэффициент экструзии:")
    excoefLabel.grid(row=4, column=0)
    excoefValue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                       "extrusion_coefficient"))
    excoefValue.grid(row=5, column=0)

    retractLabel = tk.Label(statuswin, text="Подсос:")
    retractLabel.grid(row=6, column=0)
    retractValue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                        "retract_amount"))
    retractValue.grid(row=7, column=0)

    p0Label = tk.Label(statuswin, text="P0:")
    p0Label.grid(row=8, column=0)
    p0Value = tk.Label(statuswin, text=get_setting(path, "GCoder", "p0"))
    p0Value.grid(row=9, column=0)

    p1Label = tk.Label(statuswin, text="P1:")
    p1Label.grid(row=10, column=0)
    p1Value = tk.Label(statuswin, text=get_setting(path, "GCoder", "p1"))
    p1Value.grid(row=11, column=0)

    p2Label = tk.Label(statuswin, text="P2:")
    p2Label.grid(row=12, column=0)
    p2Value = tk.Label(statuswin, text=get_setting(path, "GCoder", "p2"))
    p2Value.grid(row=13, column=0)

    sliceStepLabel = tk.Label(statuswin, text="Размер шага (мм):")
    sliceStepLabel.grid(row=14, column=0)
    sliceStepValue = tk.Label(statuswin, text=get_setting(path, "GCoder", "slice_step"))
    sliceStepValue.grid(row=15, column=0)

    extrusionMultiplexLabel = tk.Label(statuswin, text="Множитель экструзии:")
    extrusionMultiplexLabel.grid(row=16, column=0)
    extrusionMultiplexValue = tk.Label(statuswin, text=get_setting(path, "GCoder", "extrusion_multiplex"))
    extrusionMultiplexValue.grid(row=17, column=0)

    videolabel = tk.Label(statuswin, text="Путь к файлу видео:")
    videolabel.grid(row=18, column=0)
    videovalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                      "videoforpointcloud"))
    videovalue.grid(row=19, column=0)

    cloudlabel = tk.Label(statuswin, text="Путь к облаку точек:")
    cloudlabel.grid(row=20, column=0)
    cloudvalue = tk.Label(statuswin, text=get_setting(path, "GCoder",
                                                      "pointcloudpath"))
    cloudvalue.grid(row=21, column=0)

    dxflabel = tk.Label(statuswin, text="Путь к dxf-файлу:")
    dxflabel.grid(row=22, column=0)
    dxfvalue = tk.Label(statuswin, text=get_setting(path, "GCoder", "dxfpath"))
    dxfvalue.grid(row=23, column=0)

    stepbutton = tk.Button(statuswin, text="Ok", command=statuswin.destroy)
    stepbutton.grid(row=24, column=0)


def fixIni():
    osCommandString = "notepad.exe settings.ini"
    os.system(osCommandString)


def fixGcode():
    osCommandString = "notepad.exe cookie.gcode"
    os.system(osCommandString)


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
#fullstop = tk.Button(text="СТОП", command=stop)
#fullstop.grid(row=1, column=7)
mancomparing = tk.Button(text="Сравнить", command=mancomparing)
mancomparing.grid(row=1, column=8)
gcoderb = tk.Button(window, text="Генерация gcode", command=gcoder)
gcoderb.grid(row=1, column=9)
gcodesetb = tk.Button(window, text="Параметры gcode", command=gcodesetdiag)
gcodesetb.grid(row=1, column=10)
pointcloudb = tk.Button(window, text="Опознание рельефа", command=pointcloud)
pointcloudb.grid(row=1, column=11)
getprinted = tk.Button(window, text="Печать",
                       command=lambda: Thread(target=getPrinted).start())
getprinted.grid(row=1, column=12)
fixIni = tk.Button(window, text="Правка конфига", command=fixIni)
fixIni.grid(row=1, column=13)
fixGcode = tk.Button(window, text="Правка gcode", command=fixGcode)
fixGcode.grid(row=1, column=14)


# Конец отрисовки интерфейса

# Параметры окна
window['bg'] = 'gray22'
icon = tk.Image("photo", file="icon.gif")
window.tk.call('wm', 'iconphoto', window._w, icon)
window.title('MT.Pasticciere v.0.2 (Cool Cactus)')
window.geometry("1280x400+10+10")
window.mainloop()
