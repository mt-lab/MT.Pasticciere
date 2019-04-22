import socket
import sys

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    f = open("pointcloud.avi", "rb")
    data = f.read(1024)
    while (data):
        s.send(data)
        data = f.read(1024)
    s.close()
