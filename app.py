

from flask import Flask, request
import requests
import pickle
import time

from xtract_bert_main import extract_from_path, load_model

app = Flask(__name__)


@app.route('/process_bert', methods=['POST'])
def process_bert():

    # TODO: connect this
    r = request.json

    path = "/app/example_items/ex1.txt"
    model = "bert"

    # TODO: remove scale testing.
    mdata = []
    for i in range(1, 5):
        ta = time.time()
        p = extract_from_path(path=path, model_choice=model, model_items=bert_model_items)
        mdata.append(p)
        # w2v should be ~8 ms
        # bert should be ~1-2 s
        print(p)
        tb = time.time()

        print(f"Total prediction time: {tb-ta}")
    return pickle.dumps(mdata)


if __name__ == '__main__':
    t0 = time.time()
    print("...loading w2v!")
    w2v_model_items = load_model("w2v")

    t1 = time.time()
    print(f"Word2Vec model loaded in {t1 - t0} seconds!")

    bert_model_items = load_model("bert")
    # w2v should be ~9 seconds
    # bert should be ~90-120 seconds
    t2 = time.time()
    print(f"BERT model loaded in {t2 - t1} seconds!")
    print(f"Total time to load models: {t2 - t0}")

    app.run(debug=True, host='0.0.0.0')


