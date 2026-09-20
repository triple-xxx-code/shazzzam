import os
import time
import threading
from pathlib import Path
from datetime import datetime
import joblib

class HotReloader:
    """Горячее обновление базы без переиндексации"""
    
    def __init__(self, music_folder="music_library", db_path="database", 
                 check_interval=30, vector_index=None):
        self.music_folder = Path(music_folder)
        self.db_path = Path(db_path)
        self.check_interval = check_interval
        self.vector_index = vector_index
        self.running = False
        self.thread = None
        
        # Хранилище состояния файлов
        self.file_state_path = self.db_path / "file_state.pkl"
        self.file_state = self._load_file_state()
        
        # Очередь изменений
        self.changes = {'added': [], 'modified': [], 'removed': []}
    
    def _load_file_state(self):
        """Загрузка состояния файлов"""
        if self.file_state_path.exists():
            return joblib.load(self.file_state_path)
        return {}
    
    def _save_file_state(self):
        """Сохранение состояния файлов"""
        joblib.dump(self.file_state, self.file_state_path)
    
    def scan_folder(self):
        """Сканирование папки и выявление изменений"""
        extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        current_files = {}
        
        for audio_file in self.music_folder.rglob("*"):
            if audio_file.suffix.lower() not in extensions:
                continue
            
            stat = audio_file.stat()
            track_id = f"{audio_file.parent.name}/{audio_file.name}"
            current_files[track_id] = {
                'path': str(audio_file.absolute()),
                'mtime': stat.st_mtime,
                'size': stat.st_size
            }
        
        # Находим изменения
        current_ids = set(current_files.keys())
        old_ids = set(self.file_state.keys())
        
        added = current_ids - old_ids
        removed = old_ids - current_ids
        
        # Проверяем изменённые файлы
        modified = set()
        for track_id in current_ids & old_ids:
            if current_files[track_id]['mtime'] != self.file_state[track_id]['mtime']:
                modified.add(track_id)
        
        self.changes = {
            'added': list(added),
            'modified': list(modified),
            'removed': list(removed)
        }
        
        return current_files, self.changes
    
    def apply_changes(self, indexer):
        """Применение изменений к базе"""
        current_files, changes = self.scan_folder()
        
        if not any(changes.values()):
            return 0
        
        print(f"\n[{datetime.now()}] Обнаружены изменения:")
        
        # Удаляем удалённые треки
        if changes['removed']:
            print(f"  Удалено: {len(changes['removed'])}")
            for track_id in changes['removed']:
                indexer.fingerprints.pop(track_id, None)
                indexer.metadata.pop(track_id, None)
                if self.vector_index:
                    self.vector_index.remove([track_id])
        
        # Добавляем новые треки
        if changes['added']:
            print(f"  Добавлено: {len(changes['added'])}")
            for track_id in changes['added']:
                file_info = current_files[track_id]
                try:
                    fingerprint = indexer.extract_fingerprint(file_info['path'])
                    indexer.fingerprints[track_id] = fingerprint
                    indexer.metadata[track_id] = {
                        'path': file_info['path'],
                        'size': file_info['size'],
                        'indexed_at': datetime.now().isoformat()
                    }
                    
                    # Добавляем в векторный индекс (если есть CNN)
                    if self.vector_index:
                        from cnn_model import CNNModel
                        cnn = CNNModel(db_path=str(self.db_path))
                        embedding = cnn.get_embedding(file_info['path'])
                        self.vector_index.add_vectors(
                            [track_id], [embedding],
                            {track_id: {'path': file_info['path']}}
                        )
                except Exception as e:
                    print(f"  Ошибка при добавлении {track_id}: {e}")
        
        # Обновляем изменённые
        if changes['modified']:
            print(f"  Изменено: {len(changes['modified'])}")
            for track_id in changes['modified']:
                file_info = current_files[track_id]
                try:
                    fingerprint = indexer.extract_fingerprint(file_info['path'])
                    indexer.fingerprints[track_id] = fingerprint
                    indexer.metadata[track_id]['indexed_at'] = datetime.now().isoformat()
                    
                    if self.vector_index:
                        # Удаляем старый и добавляем новый
                        self.vector_index.remove([track_id])
                        from cnn_model import CNNModel
                        cnn = CNNModel(db_path=str(self.db_path))
                        embedding = cnn.get_embedding(file_info['path'])
                        self.vector_index.add_vectors(
                            [track_id], [embedding],
                            {track_id: {'path': file_info['path']}}
                        )
                except Exception as e:
                    print(f"  Ошибка при обновлении {track_id}: {e}")
        
        # Обновляем состояние и сохраняем
        self.file_state = current_files
        self._save_file_state()
        indexer.save_database()
        if self.vector_index:
            self.vector_index.save()
        
        total = sum(len(v) for v in changes.values())
        return total
    
    def start_watching(self, indexer):
        """Запуск фонового мониторинга"""
        self.running = True
        
        def watch_loop():
            while self.running:
                try:
                    changes = self.apply_changes(indexer)
                    if changes > 0:
                        print(f"Горячее обновление: {changes} изменений")
                except Exception as e:
                    print(f"Ошибка горячего обновления: {e}")
                time.sleep(self.check_interval)
        
        self.thread = threading.Thread(target=watch_loop, daemon=True)
        self.thread.start()
        print(f"Горячее обновление запущено (интервал: {self.check_interval}с)")
    
    def stop_watching(self):
        """Остановка мониторинга"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Горячее обновление остановлено")