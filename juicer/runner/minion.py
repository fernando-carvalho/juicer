import argparse
import logging
import urlparse

import redis
import yaml
from juicer.spark.spark_minion import SparkMinion

logging.basicConfig(
    format=('[%(levelname)s] %(asctime)s,%(msecs)05.1f '
            '(%(funcName)s) %(message)s'),
    datefmt='%H:%M:%S')
log = logging.getLogger()
log.setLevel(logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Config file", required=True)
    parser.add_argument("-j", "--job_id", help="Job id", type=int,
                        required=True)
    parser.add_argument("-t", "--type", help="Processing technology type",
                        required=False, default="SPARK")
    args = parser.parse_args()

    with open(args.config) as config_file:
        juicer_config = yaml.load(config_file.read())

    parsed_url = urlparse.urlparse(
        juicer_config['juicer']['servers']['redis_url'])
    redis_conn = redis.StrictRedis(host=parsed_url.hostname,
                                   port=parsed_url.port)
    if args.type == 'SPARK':
        log.info('Starting Juicer Spark Minion')
        server = SparkMinion(redis_conn, args.job_id, juicer_config)
        server.process()
    else:
        raise ValueError(
            "{type} is not supported (yet!)".format(type=args.type))