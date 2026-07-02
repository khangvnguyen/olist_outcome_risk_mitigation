FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ and output/ are meant to be bind-mounted (see docker-compose.yml)
# so results and cached downloads persist on the host.
RUN mkdir -p data/raw output

CMD ["python", "-m", "src.pipeline"]
