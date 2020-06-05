FROM python:3.6

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM" -O google_news_vec.bin.gz && rm -rf /tmp/cookies.txt

#RUN git clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
#RUN cd word2vec-GoogleNews-vectors && tar -xvf GoogleNews-vectors-negative300.bin.gz && cd

COPY xtract_bert_main.py txt_xtract.py utils.py requirements.txt /

RUN pip install -U nltk
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
