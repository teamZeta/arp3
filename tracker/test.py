from multiprocessing import Process, Pipe
import time
def neki():
    return 2


def f(conn, a):
    time.sleep(2)
    conn.send([42+a + neki(), None, 'hello'])
    conn.close()


def g(conn):
    time.sleep(2)
    conn.send([42, None, 'hellobbbb'])
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,2))
    p2 = Process(target=g, args=(parent_conn,))
    p.start()
    p2.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"

    print(child_conn.recv())
    #time.sleep(2)
    print("dobil")
    p.join()
    p2.join()