import pickle
from io import BytesIO
import re
from os import environ

from requests import get
from keras_preprocessing.sequence import pad_sequences
from stop_words import get_stop_words
from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())
BUCKET_URL = environ.get('BUCKET_URL')
response = get(f'{BUCKET_URL}/model.pkl')
model = pickle.load(BytesIO(response.content))

response = get(f'{BUCKET_URL}/scaler.pkl')
scaler = pickle.load(BytesIO(response.content))

response = get(f'{BUCKET_URL}/tokenizer.pkl')
tokenizer = pickle.load(BytesIO(response.content))

russian_stop_words = set(get_stop_words('ru'))
max_seq_length = 260

# Функция для очистки текста
def clean_text(text):
    text = re.sub(r'\s+|[^\w]+|\d+', ' ', text).lower() # Да, подобная чистка произайдёт и при токенизации после, но сейчас она нужна для лучшей очистки от стоп-слов
    text = ' '.join([word for word in text.split() if word not in russian_stop_words])  # Удаление стоп-слов
    return text

def make_prediction(post_text: str) -> float:
    # Для нового поста повторяем все действия, проведённые с данными для обучения
    # Очистка текса
    post_text_cleaned = clean_text(post_text)
    # Приведение его в векторный вид заданоЙ длинныы
    post_text_seq = tokenizer.texts_to_sequences([post_text_cleaned])
    post_text_pad = pad_sequences(post_text_seq, maxlen=max_seq_length)
    # Само предсказание
    predicted_comments_scaled = model.predict(post_text_pad, verbose=0)
    # Рескейл предсказания и возврат ответаы
    predicted_comments = scaler.inverse_transform(predicted_comments_scaled)[0][0]
    return predicted_comments if predicted_comments > 0 else 0

if __name__ == "__main__":
    print(make_prediction(''))