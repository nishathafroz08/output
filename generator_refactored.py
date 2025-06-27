# quiz_core.py
import os
import json
import random
import spacy
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import HTTPException

# Load config
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("Missing config.json. App cannot start.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

SUPPORTED_GOALS = config.get("retrieval_supported_goals", [])
SUPPORTED_DIFFICULTIES = config.get("supported_difficulties", [])
MAX_NUM_QUESTIONS = config.get("max_questions", 10)
SPACY_MODEL_PATH = config.get("spacy_model", "en_core_web_sm")

class TopicExtractor:
    def __init__(self, model=SPACY_MODEL_PATH):
        self.nlp = spacy.load(model)
        self.relevant_entity_types = [
            "ORG", "PRODUCT", "GPE", "LOC", "PERSON",
            "WORK_OF_ART", "EVENT", "NORP", "FAC"
        ]

    def extract_topics(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in self.relevant_entity_types]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
        keywords = [
            token.lemma_ for token in doc
            if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and len(token.text) > 2
        ]
        return list(set([topic.lower() for topic in (entities + noun_phrases + keywords) if len(topic) > 2]))

class QuestionMatcher:
    def __init__(self, question_bank, model=SPACY_MODEL_PATH):
        if not question_bank:
            raise ValueError("Question bank is empty.")
        self.question_bank = question_bank
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.corpus = [q.get('context', '') + ' ' + q.get('question', '') for q in question_bank]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.nlp = spacy.load(model)
        self.question_topics = {
            i: {
                'topic': q.get('topic', '').lower(),
                'goal': q.get('goal', '').lower(),
                'context_topics': self._extract_context_topics(q.get('context', ''))
            } for i, q in enumerate(question_bank)
        }

    def _extract_context_topics(self, context):
        doc = self.nlp(context)
        return list(set([
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN"]
        ]))

    def match_questions(self, topics: List[str], max_questions=10, difficulty=None, question_type=None):
        max_questions = min(max_questions, MAX_NUM_QUESTIONS)
        topic_text = ' '.join(topics)
        topic_vector = self.vectorizer.transform([topic_text])
        scores = cosine_similarity(topic_vector, self.tfidf_matrix)[0]
        question_scores = [(i, scores[i]) for i in range(len(self.question_bank))]

        for i, (idx, score) in enumerate(question_scores):
            meta = self.question_topics[idx]
            if any(
                topic in meta['topic'] or
                topic in meta['goal'] or
                topic in meta['context_topics']
                for topic in topics
            ):
                question_scores[i] = (idx, score + 0.2)

        question_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in question_scores:
            q = self.question_bank[idx]
            if difficulty and q.get('difficulty') != difficulty:
                continue
            if question_type and q.get('type') != question_type:
                continue
            q_copy = q.copy()
            q_copy['similarity_score'] = float(score)
            results.append(q_copy)
            if len(results) >= max_questions:
                break
        return results

class QuizGenerator:
    def __init__(self, matcher: QuestionMatcher):
        self.matcher = matcher

    def generate_quiz(self, topics, num_questions, difficulty=None, question_types=None, randomize=True):
        matched_questions = self.matcher.match_questions(
            topics=topics,
            max_questions=num_questions,
            difficulty=difficulty
        )

        if question_types:
            matched_questions = [q for q in matched_questions if q["type"] in question_types]

        if randomize:
            random.shuffle(matched_questions)

        return {"questions": matched_questions[:num_questions]}

def load_question_bank(goal):
    domain_files = config.get("domain_files", {})
    filename = domain_files.get(goal)
    if not filename:
        return []

    path = os.path.join("data", filename)
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_retrieval_questions(request):
    if request.goal not in SUPPORTED_GOALS or request.difficulty not in SUPPORTED_DIFFICULTIES:
        raise HTTPException(status_code=400, detail="Unsupported goal or difficulty")

    question_bank = load_question_bank(goal=request.goal)
    if not question_bank:
        raise HTTPException(status_code=404, detail="No questions found for this goal")

    sample_text = random.choice([q.get("context", "") for q in question_bank if "context" in q]) or request.goal

    try:
        extractor = TopicExtractor()
        topics = extractor.extract_topics(sample_text)
    except Exception:
        topics = [request.goal.lower()]

    matcher = QuestionMatcher(question_bank)
    quiz_gen = QuizGenerator(matcher)

    quiz = quiz_gen.generate_quiz(
        topics=topics,
        num_questions=min(request.num_questions, MAX_NUM_QUESTIONS),
        difficulty=request.difficulty,
        question_types=["mcq", "short_answer"],
        randomize=True
    )

    return {
        "goal": request.goal,
        "difficulty": request.difficulty,
        "questions": quiz["questions"]
    }
