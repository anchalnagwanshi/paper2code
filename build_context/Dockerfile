
FROM python:3.10-slim
WORKDIR /app

# Install torch
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other libraries
RUN pip install --no-cache-dir mlflow transformers datasets scikit-learn pandas

COPY train.py /app/train.py
COPY recipe.json /app/recipe.json

CMD ["python", "train.py"]
