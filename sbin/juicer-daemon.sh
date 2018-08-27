#!/usr/bin/env bash

# This script controls the juicer server daemon initialization, status reporting
# and termination
# TODO: rotate logs

REDIS_PORT=${REDIS_PORT:-6379}
REDIS_PORT_6379_TCP_ADDR=${REDIS_PORT_6379_TCP_ADDR:-redis}
REDIS=${REDIS_PORT_6379_TCP_ADDR}:${REDIS_PORT}


usage="Usage: juicer-daemon.sh (start|docker|stop|status)"

# this sript requires the command parameter
if [ $# -le 0 ]; then
  echo $usage
  exit 1
fi

# parameter option
cmd_option=$1

# set juicer_home to juicer directory root, without ./sbin
export JUICER_HOME=${JUICER_HOME:-$(cd `dirname $0`/.. && pwd)}
echo "JUICER_HOME=${JUICER_HOME}"

# get log directory
export JUICER_LOG_DIR=${JUICER_LOG_DIR:-${JUICER_HOME}/logs}

# get pid directory
export JUICER_PID_DIR=${JUICER_PID_DIR:-/var/run/}

# ensure directory exists
mkdir -p ${JUICER_PID_DIR} ${JUICER_LOG_DIR}

# log and pid files
log=${JUICER_LOG_DIR}/juicer-server-${USER}-${HOSTNAME}.out
pid=${JUICER_PID_DIR}/juicer-server-${USER}.pid

wait_for_it=`dirname $0`/wait-for-it.sh

case $cmd_option in

   (start)
      # set python path
      PYTHONPATH=${JUICER_HOME}:${PYTHONPATH} nohup -- \
        python ${JUICER_HOME}/juicer/runner/server.py \
         -c ${JUICER_HOME}/conf/juicer-config.yaml \
         >> $log 2>&1 < /dev/null &
      juicer_server_pid=$!
      # persist the pid
      echo $juicer_server_pid > $pid

      echo "Juicer server started, logging to $log (pid=$juicer_server_pid)"
      ;;

   (docker)
      trap "$0 stop" SIGINT SIGTERM

      $wait_for_it -t 60 ${REDIS} || exit 1

      # set python path
      PYTHONPATH=${JUICER_HOME}:${PYTHONPATH} \
        python ${JUICER_HOME}/juicer/runner/server.py \
         -c ${JUICER_HOME}/conf/juicer-config.yaml &
      juicer_server_pid=$!

      # persist the pid
      echo $juicer_server_pid > $pid

      echo "Juicer server started, logging to $log (pid=$juicer_server_pid)"
      wait
      ;;

   (stop)
      if [ -f $pid ]; then
         TARGET_ID=$(cat $pid)
         if [[ $(ps -p ${TARGET_ID} -o comm=) =~ "python" ]]; then
            echo "stopping juicer server, user=${USER}, hostname=${HOSTNAME}"
            kill -SIGTERM ${TARGET_ID} && rm -f ${pid}
         else
            echo "no juicer server to stop"
         fi
      else
         echo "no juicer server to stop"
      fi
      ;;

   (status)
      if [ -f $pid ]; then
         TARGET_ID=$(cat $pid)
         if [[ $(ps -p ${TARGET_ID} -o comm=) =~ "python" ]]; then
            echo "juicer server is running (pid=${TARGET_ID})"
            exit 0
         else
            echo "$pid file is present (pid=${TARGET_ID}) but juicer server not running"
            exit 1
         fi
      else
         echo juicer server not running.
         exit 2
      fi
      ;;

   (*)
      echo $usage
      exit 1
      ;;
esac
