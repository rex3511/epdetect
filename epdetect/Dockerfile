FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && wget -qO- https://mkvtoolnix.download/gpg.key | apt-key add - \
    && echo "deb https://mkvtoolnix.download/debian/ bullseye main" > /etc/apt/sources.list.d/mkvtoolnix.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends mkvtoolnix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
ENTRYPOINT []
