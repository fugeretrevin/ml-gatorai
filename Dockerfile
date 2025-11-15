# lightweight python image
FROM python:3.10-slim

# working directory
WORKDIR /app

# copy files
COPY . .

# install dependencies from the txt
RUN pip install --no-cache-dir -r requirements.txt

# the port Google Cloud Run uses
ENV PORT=8080

# start web server on that port
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
