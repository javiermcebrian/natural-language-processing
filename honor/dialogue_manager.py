from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
import requests

from utils import *

_ARTIFACTS_PATH = Path('artifacts')


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(str(_ARTIFACTS_PATH / paths['WORD_EMBEDDINGS']))
        self.thread_embeddings_folder = _ARTIFACTS_PATH / paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        thread_ids, thread_embeddings = unpickle_file(str(self.thread_embeddings_folder / "{}.pkl".format(tag_name)))
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = np.expand_dims(question_to_vec(question, self.word_embeddings, self.embeddings_dim), axis=0)
        best_thread = pairwise_distances_argmin(X=question_vec, Y=thread_embeddings, axis=1)[0]
        
        return thread_ids[best_thread]


class QARanker(object):
    def __init__(self, paths):
        self.embeddings_path = _ARTIFACTS_PATH / paths['QA_EMBEDDINGS']

    def set_endpoint(self, ip: str, port: str):
        self.endpoint = f'http://{ip}:{port}/sentence-transformers'
        self.ping = f'http://{ip}:{port}/ping'
        self._warmup_service()

    def _warmup_service(self):
        r = requests.get(url=self.ping)
        status = r.json()['status']
        if status != 200:
            raise Exception(f'Service Unavailable: status {status}')

    def _post_service(self, text: str):
        r = requests.post(url=self.endpoint, json={'text': text})
        return np.array(r.json()['embedding'])

    def get_best_answer(self, question: str):
        # Load sentence embeddings into memory
        answers, q_embeddings = unpickle_file(str(self.embeddings_path))
        # Get question vector
        question_vec = np.expand_dims(self._post_service(text=question), axis=0)
        # Get best answer ID
        best_answer_id = pairwise_distances_argmin(X=question_vec, Y=q_embeddings, axis=1)[0]
        # Get the answer
        return answers[best_answer_id]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(str(_ARTIFACTS_PATH / paths['INTENT_RECOGNIZER']))
        self.tfidf_vectorizer = unpickle_file(str(_ARTIFACTS_PATH / paths['TFIDF_VECTORIZER']))

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(str(_ARTIFACTS_PATH / paths['TAG_CLASSIFIER']))
        self.thread_ranker = ThreadRanker(paths)

        # Chit-chat
        self.qa_ranker = QARanker(paths)

    def create_chitchat_bot(self, ip: str, port: str):
        self.qa_ranker.set_endpoint(ip=ip, port=port)
       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:
        if intent == 'dialogue':
            # Launch dialogue model and sentence embeddings to get best answer. Then release the memory used.
            return self.qa_ranker.get_best_answer(question=prepared_question)
        
        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question=prepared_question, tag_name=tag)
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)
