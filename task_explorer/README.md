# URL
https://rn-iii-task-explorer.eks-test-default.mpg-chm.com/

# Steamlit Template

Demonstrates the deployment of a simple streamlit app on our cluster.

For more information on streamlit, visit the official documentation:
https://docs.streamlit.io/

## Adapt template

1. Change app name in `app.yml`
2. Replace `streamlit-template` in Dockerfile and docker-compose.yml with app
   name

## Run locally without docker

Requires python3.9

```
./run_local.sh
```

Streamlit app is available at: http://0.0.0.0:5040/

## Run locally with docker

Requires docker.

```
docker compose up devapp --build
```

Streamlit app is available at: http://0.0.0.0:5050/

The python code in this case is linked into the docker image as a volume.
This allows to run the streamlit app within docker, while still be able to edit
the code live.

## Run the production docker image locally

Requires docker.

```
docker compose up app --build
```

Streamlit app is available at: http://0.0.0.0:5055/

In this case the code of the streamlit app is part of the docker image and
cannot be modified after building. Therefore the image has to be rebuild once
for a change to be effective.

## Run the production docker image on the cluster

No requirement. The master branch is automatically deployed on the cluster.

The app is available at:
https://streamlit-template.eks-test-default.mpg-chm.com/

The first part of the URL depends on the app name chosen in the app.yml.

It generally takes a couple of minutes for a new deployment to become available.
The progress can be see on gitlab, i.e.
https://gitlab.gwdg.de/mpib/chm/hci/technical-prototypes/streamlit-template/-/pipelines
