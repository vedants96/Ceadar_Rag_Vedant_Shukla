
FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ /app/

RUN mkdir -p /app/logs

ENV PORT=7860
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0


CMD ["streamlit", "run", "streamlit_app.py"]