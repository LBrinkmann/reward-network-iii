FROM python:3.9-slim as base
RUN pip --no-cache-dir install -U pip

WORKDIR /app
COPY setup/requirements/requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt
COPY common /app/common
COPY setup.py .
RUN pip --no-cache-dir install --no-deps -e .

# Task Explorer

FROM base as task-explorer-dev
COPY setup/requirements/requirements_viz.txt .
RUN pip --no-cache-dir install -r requirements_viz.txt

FROM task-explorer-dev as task-explorer
ADD task_explorer/app /app/task_explorer/app
ADD config /app/config
CMD ["streamlit", "run", "task_explorer/app/app.py", "--server.port=5000", "--server.address=0.0.0.0"]

# Backend

FROM base as backend-dev
COPY setup/requirements/requirements_backend.txt .
RUN pip --no-cache-dir install -r requirements_backend.txt
COPY setup/requirements/requirements_dev.txt .
RUN pip --no-cache-dir install -r requirements_dev.txt

FROM base as backend
COPY setup/requirements/requirements_backend.txt .
RUN pip --no-cache-dir install -r requirements_backend.txt
COPY backend/app .
CMD ["uvicorn", "server:api", "--host", "0.0.0.0", "--port", "5000"]
