from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pathlib import Path

app = Flask(__name__)
_ARTIFACTS_PATH = Path('/app/artifacts')
_MODEL_PATH = _ARTIFACTS_PATH / 'distilbert-base-nli-stsb-mean-tokens'


class Sentence2Embedding(object):

    # Class attribute model
    model = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = SentenceTransformer(str(_MODEL_PATH))
        return cls.model

    @classmethod
    def run(cls, text):
        """Compute embeddings for a given sentence"""
        model = cls.get_model()
        return model.encode([text])[0]


@app.route('/ping', methods=['GET'])
def ping():
    health = Sentence2Embedding.get_model() is not None
    status = 200 if health else 404
    return jsonify(response='Ping', status=status)


@app.route('/sentence-transformers', methods=['POST'])
def sentence_transformers():
    text = request.json['text']
    embedding = Sentence2Embedding.run(text)
    return jsonify({'embedding': embedding.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
