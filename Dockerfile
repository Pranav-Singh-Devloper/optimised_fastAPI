FROM python:3.11-slim
WORKDIR .
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["bash", "start.sh"]
