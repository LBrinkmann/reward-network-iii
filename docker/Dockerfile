FROM python:3.9-slim as base
RUN pip --no-cache-dir install -U pip

WORKDIR /app
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt
COPY common /app/common
COPY setup.py .
RUN pip --no-cache-dir install --no-deps -e .

FROM base as task-explorer-dev
COPY requirements_viz.txt .
RUN pip --no-cache-dir install -r requirements_viz.txt

FROM base as task-explorer
ADD task_explorer/app /app/task_explorer/app
CMD ["streamlit", "run", "task_explorer/app/app.py", "--server.port=5000", "--server.address=0.0.0.0"]
