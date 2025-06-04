# Fastapi

**Beschrijving**  
Een Python FastAPI-service die voorspellingen levert voor inkomende data. Wordt aangeroepen door de .NET API.

---

## Vereisten
- Python 3.11

## Installatie
```bash
git clone https://github.com/TralaAI/Fastapi.git
cd fastapi
pip install -r requirements.txt
```
## Runnen
```bash
uvicorn app.main:app --reload --host $HOST --port $PORT
```
