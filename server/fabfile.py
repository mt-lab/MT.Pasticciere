from fabric.api import *
env.hosts['169.254.222.55']

def startCheese():
    run("cheese")
