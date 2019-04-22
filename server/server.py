import socket
import sys

HOST = '169.254.222.55'  # адрес ноута в проводной локальной сети
# HOST = '127.0.0.1'
PORT = 65432        # Порт из примера, сойдёт любой аналогичный

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    #while True:
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        f = open('dummy' + ".avi", 'wb')  # открываем в двоичном виде
        while True:
            data = conn.recv(1024)
            while (data):
                f.write(data)   # запись файла
                data = conn.recv(1024)
        f.close()
        conn.close()
    s.close()
