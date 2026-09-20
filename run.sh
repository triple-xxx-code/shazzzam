# 1. Установка зависимостей
pip install -r requirements.txt
pip check

# 2. Поместите музыку в music_library/

# 3. Обучение CNN модели (опционально, но рекомендуется)
python train.py

# 4. Индексация библиотеки
python index.py

# 5. Запуск веб-сервера
python app.py

# вырезать фрагмент, начиная с 10-й секунды, длиной 30 секунд
# ffmpeg -ss 00:00:10 -i input.mp3 -t 30 -c copy output.mp3
