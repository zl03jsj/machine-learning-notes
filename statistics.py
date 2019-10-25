import json
import requests
import sys, getopt
import time
import datetime
import os


def do_call(method):
    global rpc_url

    try:
        return requests.post(
            rpc_url, data=json.dumps(method), headers=headers,
            auth=(user_name, password)).json()
    except Exception as e:
        print('do call failed, with err :', e)
        sys.exit(2)


def get_blockhash(num):
    method = methods["blockhash"]
    method["params"] = [num]

    result = do_call(method)

    if result['error'] is not None:
        return ""
    return result["result"]


def get_block(block_id):
    method = methods["block"]
    method["params"] = [block_id]

    result = do_call(method)
    if result['error'] is not None:
        print(result['error'])
        return None

    return result["result"]


def get_block_count():
    method = methods["blockcount"]
    result = do_call(method)
    if result['error'] is not None:
        return ""
    return result["result"]


methods = {
    "blockhash": {
        "method": "getblockhash",
        "params": [1],
        "jsonrpc": "2.0",
        "id": 0,
    },
    "block": {
        "method": "getblock",
        "params": [],
        "jsonrpc": "2.0",
        "id": 0,
    },
    "blockcount": {
        "method": "getblockcount",
        "params": [],
        "jsonrpc": "2.0",
        "id": 0,
    }
}

rpc_url = ''
user_name = ''
password = ''
start_block_num = 0
count = 0

headers = {'content-type': 'application/json'}


def usage():
    print("""
    usage: 
    statistic.py 
        -r(--rpc)       <rpc-server-url>
        -f(--from)      <from-block-num>
        -c(--count)     <block-count> 
        -u(--user)      <rpc-user-name>
        -p(--password)  <rpc-password> 
    
    for example:
    python statistic.py -r http://127.0.0.1:18443/ -h -u zengl -p 123456 -f 100 -c 100
    """)

def pars_args(argv):
    global rpc_url, user_name, password, start_block_num, count

    try:
        opts, args = getopt.getopt(argv, "hr:f:c:u:p:",
                                   ["rpc=", "from=", "count=", "user=", "password="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-r", "--rpc"):
            rpc_url = arg
        elif opt in ("-f", "--from"):
            start_block_num = int(arg)
        elif opt in ("-c", "--count"):
            count = int(arg)
        elif opt in ("-u", "--user"):
            user_name = arg
        elif opt in ("-p", "--password"):
            password = arg
    if rpc_url == "" or user_name == "" or password == "" or start_block_num < 0 or count <= 0:
        usage()
        sys.exit(2)


def main(argv):
    global rpc_url, user_name, password, start_block_num, count
    pars_args(argv)

    print("rpc-url={0}, user={1}, password={2}, start_block_num={3}, block_count={4}".
          format(rpc_url, user_name, password, start_block_num, count))

    block_count = get_block_count()
    lasttime = 0
    time_diff = list()

    f = open('statistic.txt', 'w')

    for i in range(start_block_num, start_block_num + count):
        block_id = get_blockhash(i)
        result = get_block(block_id)

        if result is None:
            break

        block_time = result["time"]
        timestruct = time.localtime(block_time)

        message = "block : %6d   block-time: (%s)" % ( \
            i, time.strftime('%Y-%m-%d '
                             '%H:%M:%S',
                             timestruct))

        if i > start_block_num:
            time_diff.append(block_time - lasttime)
            message += "    time-diff : %4d(s)" % (time_diff[-1])
        lasttime = block_time

        f.write(message + "\n")

        print(message)

    message = (
            """
            |______________>>>>>>the statistic result:<<<<<<____________
            |from : %d, block_count : %d, average time : %d(second)
            |-----------------------------------------------------------
            """ % (count, start_block_num, sum(time_diff) / (count - 1)))

    print(message)

    f.write(message)
    f.close()

    print("write to file:%s" % os.path.realpath(f.name))


if __name__ == "__main__":
    main(sys.argv[1:])
