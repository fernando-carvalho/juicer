FROM ubuntu:16.04
MAINTAINER Vinicius Dias <viniciusvdias@dcc.ufmg.br>

ENV SPARK_HOME /usr/local/spark
ENV JUICER_HOME /usr/local/juicer
ENV PYTHONPATH $PYTHONPATH:$JUICER_HOME:$SPARK_HOME/python

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv E56151BF \
  && echo deb http://repos.mesosphere.io/ubuntu trusty main > /etc/apt/sources.list.d/mesosphere.list \

  && apt-get update && apt-get install -y --no-install-recommends \
      python-pip \
      python-tk \
      openjdk-8-jdk \
      curl \
      locales \
      mesos \
  && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
  && locale-gen \ 
  && update-locale LANG=en_US.UTF-8 \
  && echo "LANG=en_US.UTF-8" >> /etc/default/locale \
  && echo "LANGUAGE=en_US.UTF-8" >> /etc/default/locale \
  && echo "LC_ALL=en_US.UTF-8" >> /etc/default/locale \
  && rm -rf /var/lib/apt/lists/*

ENV SPARK_HADOOP_PKG spark-2.2.0-bin-hadoop2.6
ENV SPARK_HADOOP_URL http://www-eu.apache.org/dist/spark/spark-2.2.0/${SPARK_HADOOP_PKG}.tgz
RUN curl -s ${SPARK_HADOOP_URL} | tar -xz -C /usr/local/  \
  && mv /usr/local/$SPARK_HADOOP_PKG $SPARK_HOME

WORKDIR $JUICER_HOME
COPY requirements.txt $JUICER_HOME

RUN pip install -r $JUICER_HOME/requirements.txt

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

COPY . $JUICER_HOME
RUN pybabel compile -d $JUICER_HOME/juicer/i18n/locales

CMD ["/usr/local/juicer/sbin/juicer-daemon.sh", "startf"]
