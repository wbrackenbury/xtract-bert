FROM python:3.6

COPY xtract_bert_main.py txt_xtract.py utils.py requirements.txt /

RUN pip install -U nltk
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
