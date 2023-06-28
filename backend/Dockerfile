FROM python:3.9-buster

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app .

CMD ["uvicorn", "server:api", "--host", "0.0.0.0", "--port", "5000"]
