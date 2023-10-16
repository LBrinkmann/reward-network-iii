# SEE: https://github.com/PacktPublishing/Building-Python-Web-APIs-with-FastAPI
import asyncio

import httpx
import pytest
from httpx import AsyncClient

from database.connection import DatabaseSettings
from models.config import ExperimentSettings
from models.session import Session
from models.subject import Subject
from server import api
from study_setup.generate_sessions import generate_sessions, reset_networks


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def default_client():
    test_settings = DatabaseSettings()
    # test_settings.MONGO_URL = "mongodb://localhost:27017"
    test_settings.DATABASE_NAME = "test-reward-network-iii"

    await test_settings.initialize_database()

    async with AsyncClient(app=api, base_url="http://testserver/") as client:
        yield client

        # Clean up resources
        await Session.find().delete()
        await Subject.find().delete()
        await ExperimentSettings.find().delete()


@pytest.fixture(scope="function")
async def e_config(default_client):
    # create a new config
    config = ExperimentSettings()
    config.active = True
    await config.save()
    return config


@pytest.fixture(scope="function")
async def create_empty_experiment(
    default_client: httpx.AsyncClient, e_config: ExperimentSettings
):
    reset_networks()
    for replication in range(e_config.n_session_tree_replications):
        await generate_sessions(
            experiment_num=replication,
            config=e_config,
        )
