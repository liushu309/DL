## 显示设备
  使用tensorflow显示计算机中可用的设备
  
    import os
    from tensorflow.python.client import device_lib
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
    print(os.environ.keys())

    if __name__ == "__main__":
        print(device_lib.list_local_devices())
