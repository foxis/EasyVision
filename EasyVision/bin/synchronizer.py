import Pyro4
from EasyVision.processors.synchronization import PyroSynchronizerObject
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Remote multiple processor stack Synchronizer")
    parser.add_argument("name", help="Remote name of Synchronizer")
    parser.add_argument("-N", "--stream-number", default=2, type=int, help="Number of streams to synchronize")
    parser.add_argument("-n", "--nameserver", default="", help="hostname of the name server (default: empty)")
    parser.add_argument("-H", "--host", default="localhost", help="Hostname of the server (default: localhost)")
    parser.add_argument("-p", "--port", default=0, help="Port of the server (default: 0)")
    parser.add_argument("-t", "--timeout", default=30, help="Timeout for synchronization in seconds (default: 30s)")

    args = parser.parse_args()

    with Pyro4.Daemon(host=args.host, port=args.port) as daemon:
        if args.nameserver:
            ns = Pyro4.locateNS(host=args.nameserver)
        else:
            ns = Pyro4.locateNS()

        sync = PyroSynchronizerObject(args.N, timeout=args.timeout)
        uri = daemon.register(sync)
        ns.register(args.name, uri)

        daemon.requestLoop()


if __name__ == "__main__":
    main()
