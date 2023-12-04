# frontend/Dockerfile

FROM python:3.10.13

WORKDIR /app

COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip hash -a sha256 requirements.txt
#RUN pip hash -a sha256 req.txt
RUN pip install --no-cache-dir --timeout=1500 -r requirements.txt
#RUN pip install --no-cache-dir --timeout=1000 -r req.txt

COPY . .

#EXPOSE 8501

#CMD ["streamlit", "run", "app.py"]

CMD python -m streamlit run --server.port $PORT app.py