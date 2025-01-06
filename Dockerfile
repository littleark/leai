FROM python:3.10

COPY ./packages.txt .

RUN apt-get update
RUN xargs -a packages.txt apt-get install -y

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "run_servers.py"]
