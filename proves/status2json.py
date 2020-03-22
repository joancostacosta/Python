#!/usr/bin/env python

import re
import json

STATUS_FILE_PATH = "/var/cache/nagios3/status.dat"

def read_status():
    hosts = {}
    services = {}

    fh = open(STATUS_FILE_PATH)
    status_raw = fh.read()
    pattern = re.compile('([\w]+)\s+\{([\S\s]*?)\}',re.DOTALL)
    matches = pattern.findall(status_raw)
    for def_type, data in matches:
        lines = [line.strip() for line in data.split("\n")]
        pairs = [line.split("=", 1) for line in lines if line != '']
        data = dict(pairs)

        if def_type == "servicestatus":
            services[data['service_description']] = data
            if 'host_name' in data:
                hosts[data['host_name']]['services'].append(data)

        if def_type == "hoststatus":
            data['services'] = []
            hosts[data['host_name']] = data
    return {
        'hosts': hosts,
        'services': services,
    }

if __name__ == "__main__":
    data = read_status()

    print ("Content-Type: application/json\n") # 1 \n because print does it's own
    print (json.dumps(data['hosts'], sort_keys=True, indent=4))
