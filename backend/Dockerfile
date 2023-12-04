FROM python:3.10.13

WORKDIR /app

COPY requirements.txt .

COPY req.txt .

#RUN pip install -r requirements.txt
RUN pip hash -a sha256 requirements.txt
RUN pip hash -a sha256 req.txt
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt
RUN pip install --no-cache-dir --timeout=1000 -r req.txt


COPY . .

EXPOSE 8080

CMD ["python", "main.py"]

# CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT 
