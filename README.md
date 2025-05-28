# fastapi

**Beschrijving**  
Een Python FastAPI-service die voorspellingen levert voor inkomende data. Wordt aangeroepen door de Core API.

---

## Vereisten
- Python 3.11

## Installatie
```bash
git clone https://github.com/TralaAI/fastapi.git
cd tralaAI-fastapi
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
## Runnen
```bash
uvicorn app.main:app --reload --host $HOST --port $PORT
```
