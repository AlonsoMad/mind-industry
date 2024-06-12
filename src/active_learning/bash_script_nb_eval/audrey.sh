#!/bin/bash
# Start Jupyter Notebook and open the specific notebook
jupyter notebook --NotebookApp.default_url="/notebooks_eval/audrey.ipynb" --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
