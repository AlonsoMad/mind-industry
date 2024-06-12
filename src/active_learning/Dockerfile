FROM jupyter/base-notebook

RUN pip install --no-cache-dir notebook

USER root

COPY . /home/jovyan/work/active_learning

COPY bash_script_nb_eval/audrey.sh /usr/local/bin/audrey.sh
COPY bash_script_nb_eval/valerie.sh /usr/local/bin/valerie.sh

RUN chmod +x /usr/local/bin/audrey.sh
RUN chmod +x /usr/local/bin/valerie.sh

USER jovyan

WORKDIR /home/jovyan/work/active_learning

RUN pip install -r requirements.txt

EXPOSE 8888