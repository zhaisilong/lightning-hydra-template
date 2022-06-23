import threading


def _show(num):
    print(num)

if __name__ == '__main__':
    for i in range(10):
        t = threading.Thread(target=_show, args=(i,))
        t.start()

