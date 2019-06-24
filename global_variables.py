import configparser

# TODO: рефактор и комментарии

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


config_path = 'settings.ini'
X, Y, Z = 0, 1, 2
hsvLowerBoundString = get_setting(config_path, 'Scanner', 'hsv_min')[1:-1]
hsvUpperBoundString = get_setting(config_path, 'Scanner', 'hsv_max')[1:-1]
hsvLowerBound = [int(hsvLowerBoundString.split(', ')[i]) for i in range(len(hsvLowerBoundString.split(', ')))]
hsvUpperBound = [int(hsvUpperBoundString.split(', ')[i]) for i in range(len(hsvUpperBoundString.split(', ')))]
accuracy = float(get_setting(config_path, 'GCoder', 'accuracy'))
step = float(get_setting(config_path, 'GCoder', 'slice_step'))
DXF_PATH = get_setting(config_path, 'GCoder', 'dxfpath')
PCD_PATH = get_setting(config_path, 'GCoder', 'pointcloudpath')
VID_PATH = get_setting(config_path, 'GCoder', 'videoforpointcloud')
Z_max = 30
