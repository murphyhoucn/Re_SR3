import time

def get_current_time():
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    return current_time