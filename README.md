audio_search/
├── database/              # База отпечатков, модели, FAISS индекс
├── music_library/         # Локальные треки
├── samples/               # Семплы для поиска
├── templates/             # HTML шаблоны
├── static/                # CSS/JS
├── uploads/               # Загруженные файлы
├── index.py              # Индексация библиотеки
├── search.py             # Поиск по семплу
├── train.py              # Обучение CNN модели
├── cnn_model.py          # Свёрточная нейросеть
├── augmentation.py       # Аугментация данных
├── vector_index.py       # FAISS векторный поиск
├── hot_reload.py         # Горячее обновление базы
├── app.py                # FastAPI веб-интерфейс
└── requirements.txt
