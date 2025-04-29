FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_fastapi.py .

EXPOSE 3000

CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "3000"]
