import os

import pytest

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def alibaba_csv():
    return os.path.join(FIXTURES, "alibaba_sample.csv")


@pytest.fixture
def google_csv():
    return os.path.join(FIXTURES, "google_task_events_sample.csv")


@pytest.fixture
def normalized_df(alibaba_csv):
    from gputrace import loaders
    return loaders.get_loader("alibaba2020").load(alibaba_csv)
