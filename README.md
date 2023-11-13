# Reward Network III

## Setup

### Prerequisites

```
python3 -m venv .venv
# Mac/Linux
. .venv/bin/activate
# Windows
# source .venv/Scripts/Activate


pip install --upgrade pip
pip install wheel setuptools
pip install -e ".[dev,train,viz]"
```

## Quick start



### Generate Networks

```
docker compose run common python common/generate/generation.py -i data/23_11_13/networks.yml -o data/23_11_13/networks.json
```

### Generate Solutions

```
docker compose run common python common/solve/rule_based.py -c data/23_11_13/solution.yml -n data/23_11_13/networks.json -o data/23_11_13/solution
```


## Deployment

- Frontend
  URL: [https://rn-iii-frontend.eks-test-default.mpg-chm.com](https://rn-iii-frontend.eks-test-default.mpg-chm.com)
- Backend
  URL: [https://rn-iii-backend.eks-test-default.mpg-chm.com](https://rn-iii-backend.eks-test-default.mpg-chm.com)

## DB models

### Session

<p align="centre">
<img alt="Backend-Frontend interaction" height="auto" src="backend/app/models/session.png" width="100%"/>
</p>

### Subject

<p align="centre">
<img alt="Backend-Frontend interaction" height="auto" src="backend/app/models/subject.png" width="25%"/>
</p>

### Experiment Settings

<p align="centre">
<img alt="Backend-Frontend interaction" height="auto" src="backend/app/models/config.png" width="25%"/>
</p>

## Development environment


Set up the development environment:

```bash

docker compose up

```

NOTE: you can use `--build` flag to rebuild images and `-d` to run in detached
mode.

To run only the backend containers one can use the following command:

```bash

docker compose up -d backend

```

Stop everything and remove all volumes:

```bash

docker-compose down --volumes

```

### Links when the docker-compose-dev.yml is running

- React: http://localhost:9000/
- Storybook: http://localhost:6006/
- FastAPI: http://localhost:5000/
- Swagger UI FastAPI: http://localhost:5000/docs

### Run pytest

Run all test

```zsh

sudo docker compose run backend python -m pytest -vv -s

```

Or start the dev container in the interactive mode:

```zsh
sudo docker compose run fastapi /bin/bash
```

and run individual tests:

```zsh
pytest -k test_session
```

````

Or

See [pytest docs](https://docs.pytest.org/en/7.1.x/how-to/usage.html#invoke-python)
for more details.

### `apiTypes.ts`

`apiTypes.ts` file is generated in the `server.py` each time FastAPI server is
restarted and then copied in frontend `src` folder.

Useful commands to clean up the system:

```bash

docker system df # check disk usage
docker system prune --all --force # clean up unused images and volumes
docker system prune --volumes --force # clean up unused volumes

````

## Deployment notes

- Frontend and backend pipelines are triggered by the parent
  pipeline [`.gitlab-ci.yml`](.gitlab-ci.yml).
- In order to run two separate pipelines from the same repo, we need to copy the
  content of [`frontend`](frontend) and [`backend`](backend) folders to the root
  of the repo before running the corresponding child
  pipeline ([`.gitlab-ci-frontend.yml`](`.gitlab-ci-frontend.yml`)
  or [`.gitlab-ci-backend.yml`](.gitlab-ci-backend.yml)).
- [`frontend`](frontend) folder should be deleted before the `build` stage of
  the [`.gitlab-ci-frontend.yml`](.gitlab-ci-frontend.yml) pipeline. This is
  necessary to avoid typescript errors when building the frontend.

## Backend-Frontend interaction scheme

<p align="centre">
<img alt="Backend-Frontend interaction" height="auto" src="docs/backend-frontend.png" width="50%"/>
</p>
