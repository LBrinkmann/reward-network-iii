FROM python:3.9-slim as streamlit-template-base
RUN pip --no-cache-dir install -U pip

WORKDIR /app
COPY requirements.txt .
RUN pip --no-cache-dir install -r requirements.txt

# SEE: https://github.com/rusty1s/pytorch_scatter#binaries
ENV CUDA="cpu"
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+${CUDA}.html

FROM streamlit-template-base as streamlit-template-devapp
COPY requirements_dev.txt .
RUN pip --no-cache-dir install -r requirements_dev.txt
COPY run_dev.sh ./
CMD ["./run_dev.sh"]

FROM streamlit-template-base
ADD app ./app
COPY run.sh .
CMD ["./run.sh"]
