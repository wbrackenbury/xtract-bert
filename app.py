

from flask import Flask, request
from xtract_sdk.downloaders.google_drive import GoogleDriveDownloader
import requests
import pickle
import time

from xtract_bert_main import extract_from_path, load_model

app = Flask(__name__)

# TODO: unassume that files must be from Google Drive. 
@app.route('/process_bert', methods=['POST', 'GET'])
def process_bert():

    print("Received new request!")
    r = request.data
    data = json.loads(r)

    creds = event["gdrive"]
    file_id = event["file_id"]
    is_gdoc = event["is_gdoc"]
    extension = event["extension"]

    if extension.lower() == "pdf":
        mimeType = "application/pdf"
    else:
        mimeType = "text/plain"

    t_download_0 = time.time()
    try:
        downloader = GoogleDriveDownloader(auth_creds=creds)
        if is_gdoc:
            downloader.fetch(fid=file_id, download_type="export", mimeType=mimeType)
        else:
            downloader.fetch(fid=file_id, download_type="media")

    except Exception as e:
        return e
    tb = time.time()

    for filepath in downloader.success_files:
        try:
            new_mdata = xtract_keyword_main.extract_keyword(filepath, pdf=True if extension.lower() == 'pdf' else False)
        except Exception as e:
            return e
    t_download_1 = time.time()
    print(f"Total download time: {t_download_0 - t_download_1}")

    mdata = {}
    
    ta = time.time()
    bert_mdata = extract_from_path(path=path, model_choice="bert", model_items=bert_model_items)
    w2v_mdata = extract_from_path(path=path, model_choice="w2v", model_items=w2v_model_items)    
    # w2v should be ~8 ms
    # bert should be ~1-2 s
    tb = time.time()

    mdata["w2v"] = w2v_mdata
    mdata["bert"] = bert_mdata

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

    app.run(debug=True, use_reloader=False, host='0.0.0.0')


