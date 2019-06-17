# MT.Pasticciere

## Требования к системе:
* numpy
* opencv-python
* ezdxf
* pygcode
* imutils
* paramiko
* matplotlib

---

## Требования к оформлению кода:

* Переменные и функции:
    * Называются camelCaseом
    * Названия не дублируются
    * В начале функции содержится комментарий о её назначении и вводимых в неё параметрах

* Пример:
```python
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
    sftp.put(localpath, remotepath)
    sftp.close()
    transport.close()
```


* Логгинг функций:
    * Необходимо перечислить функции (процедуры), в которых следует замерять время или собирать дебаг информацию в логе
    * Здесь будет пример как это сделать

* Файл настроек:
    * Здесь необходимо привести список параметров для файла настроек. ВСЁ, что вводится пользователем в интерфейсе необходимо хранить в этом файле. ОБЯЗАТЕЛЬНО указывать единицы измерения у параметров, если таковые присутствуют, ибо подобные вещи должны находить отражение в интерфейсе программы.
