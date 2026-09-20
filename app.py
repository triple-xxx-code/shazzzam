from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import uuid
import time
from search import AudioSearcher
from index import AudioIndexer
from hot_reload import HotReloader
import tensorflow as tf

# Глобальные объекты
searcher = None
indexer = None
reloader = None

def init_services():
    global searcher, indexer, reloader
    try:
        indexer = AudioIndexer(use_cnn=True)
        indexer.load_database()
        searcher = AudioSearcher(use_vector=True)
        
        reloader = HotReloader(
            music_folder="music_library",
            db_path="database",
            check_interval=30,
            vector_index=searcher.vector_index if hasattr(searcher, 'vector_index') else None
        )
        reloader.start_watching(indexer)
        print("✅ Сервисы инициализированы")
    except Exception as e:
        print(f"⚠️ Ошибка инициализации: {e}")
        print("Запустите index.py для создания базы")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Запуск при старте
    init_services()
    yield
    # Корректное завершение при остановке (Ctrl+C)
    print("\n🛑 Завершение работы...")
    if reloader:
        reloader.stop_watching()
    # Очистка сессии TensorFlow предотвращает Segmentation Fault при выходе
    tf.keras.backend.clear_session()
    print("✅ Сервисы корректно остановлены")

# Инициализация с lifespan вместо устаревших @app.on_event
app = FastAPI(title="Audio Search", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем запросы с любых источников (для локальной разработки)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    tracks_count = len(indexer.fingerprints) if indexer else 0
    db_size = 0.0
    
    if indexer and getattr(indexer, 'metadata', None):
        try:
            db_size = sum(
                Path(m['path']).stat().st_size 
                for m in indexer.metadata.values() 
                if Path(m['path']).exists()
            ) / (1024**2)
        except Exception:
            db_size = 0.0
            
    # СОВРЕМЕННЫЙ СИНТАКСИС FASTAPI: request вынесен как отдельный аргумент
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stats": {
                "tracks": tracks_count,
                "db_size": round(db_size, 1)
            }
        }
    )

@app.post("/upload-and-search")
async def upload_and_search(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    min_similarity: float = Form(0.3),  # Новый параметр: порог уверенности
    top_k: int = Form(10)               # Новый параметр: количество результатов
):
    if not searcher:
        return JSONResponse({"error": "База не инициализирована"}, status_code=500)
    
    file_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{file_id}{file_ext}"
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        start_time = time.time()
        # Передаем параметры из формы в метод поиска
        results = searcher.search(str(save_path), top_k=top_k, min_similarity=min_similarity)
        search_time = time.time() - start_time
        
        return JSONResponse({
            "file_id": file_id,
            "filename": file.filename,
            "results": results,
            "search_time": round(search_time, 3),
            "total_results": len(results)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/reindex")
async def reindex(background_tasks: BackgroundTasks):
    if not indexer:
        return JSONResponse({"error": "Indexer не инициализирован"}, status_code=500)
    
    def do_reindex():
        try:
            indexer.index_library()
            print("✅ Переиндексация завершена")
        except Exception as e:
            print(f"❌ Ошибка переиндексации: {e}")
    
    background_tasks.add_task(do_reindex)
    return JSONResponse({"status": "Переиндексация запущена в фоне"})

@app.get("/stats")
async def get_stats():
    if not indexer:
        return JSONResponse({"error": "База не инициализирована"}, status_code=500)
    
    return JSONResponse({
        "tracks_count": len(indexer.fingerprints),
        "method": "CNN + FAISS" if getattr(indexer, 'use_cnn', False) else "librosa",
        "vector_index_size": searcher.vector_index.index.ntotal if hasattr(searcher, 'vector_index') and searcher.vector_index.index else 0
    })

@app.get("/track/{track_id}")
async def get_track(track_id: str):
    if not indexer or track_id not in indexer.metadata:
        return JSONResponse({"error": "Трек не найден"}, status_code=404)
    
    meta = indexer.metadata[track_id]
    return JSONResponse({
        "track_id": track_id,
        "path": meta['path'],
        "size": meta['size'],
        "indexed_at": meta['indexed_at']
    })

@app.get("/hot-reload-status")
async def hot_reload_status():
    if not reloader:
        return JSONResponse({"error": "Hot reloader не запущен"}, status_code=500)
    
    current_files, changes = reloader.scan_folder()
    return JSONResponse({
        "running": reloader.running,
        "check_interval": reloader.check_interval,
        "tracked_files": len(reloader.file_state),
        "pending_changes": {
            "added": len(changes['added']),
            "modified": len(changes['modified']),
            "removed": len(changes['removed'])
        }
    })

@app.post("/apply-hot-reload")
async def apply_hot_reload():
    if not reloader or not indexer:
        return JSONResponse({"error": "Сервисы не инициализированы"}, status_code=500)
    
    changes = reloader.apply_changes(indexer)
    return JSONResponse({
        "status": "ok",
        "changes_applied": changes
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)