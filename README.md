# MT.Pasticciere

Теперь работает **только на python 3.7+**

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
* В начале каждого файла необходимо указать информацию об атворе, функции, которые он содержит и задачи, которые он выполняет в программе
    * Пример:
        ```python
        """
        pasticcere.py
        Author: Dmitry K of MT.lab

        Главный файл программы, отвечает за отрисовку интерфейса программы и содержит
        функции для передачи данных и команд по сети.
        """
        ```

* Переменные и функции:
    * Называются camelCaseом и своим названием составляют представление о своём назначении
    * Названия переменных и функций не дублируются
    * В начале функции содержится комментарий о её назначении и вводимых в неё параметрах с типами
        * Пример:
            ```python
            def getFile(host, port, name, password, file):
              """
              Забирает файл с удалённого устройства не меняя имени файла

              host (str) - ip-адрес устройства
              port (int) - порт для соединения с устройством
              name (str)- имя пользователя ssh
              password (str) - пароль пользователя ssh
              file (str) - имя файла на удалённом устройстве
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
    * Пример:
  ```python
  import logging
  module_logger = logging.getLogger("pasticciere.**%ИМЯ МОДУЛЯ%**")
  def add(x, y):
      """
      Docstring
      """
      logger = logging.getLogger("pasticciere.**%ИМЯ МОДУЛЯ%**.add")
      logger.info("**%ТЕКСТ СООБЩЕНИЯ В ЛОГ%**")
      return x + y
  ```

* Файл настроек:
    * Здесь необходимо привести список параметров для файла настроек. ВСЁ, что вводится пользователем в интерфейсе необходимо хранить в этом файле. ОБЯЗАТЕЛЬНО указывать единицы измерения у параметров, если таковые присутствуют, ибо подобные вещи должны находить отражение в интерфейсе программы.
