# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PipelinesClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase


@pytest.fixture
def mock_kfp_client():
    """Create a mock kfp.Client with all methods we wrap."""
    client = MagicMock()
    client.get_user_namespace.return_value = ""
    client.get_pipeline_id.return_value = "pipeline-id-123"
    client.get_experiment.return_value = SimpleNamespace(experiment_id="exp-id-123")
    return client


@pytest.fixture
def client(mock_kfp_client, monkeypatch):
    """Create PipelinesClient with mock kfp.Client."""
    from kubeflow.pipelines.api.pipelines_client import PipelinesClient

    mock_kfp_class = MagicMock(return_value=mock_kfp_client)
    monkeypatch.setattr("kubeflow.pipelines.api.pipelines_client.PipelinesClient.__init__", None)

    pc = PipelinesClient.__new__(PipelinesClient)
    pc._client = mock_kfp_client
    return pc


# ------------------------------------------------------------------
# Constructor tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="raises helpful ImportError when kfp not installed",
            expected_status=FAILED,
            config={"base_url": "http://localhost"},
            expected_error=ImportError,
        ),
    ],
)
def test_init_import_error(test_case, monkeypatch):
    """Test that __init__ raises helpful ImportError when kfp is missing."""
    from kubeflow.pipelines.api.pipelines_client import PipelinesClient

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def mock_import(name, *args, **kwargs):
        if name == "kfp":
            raise ImportError("No module named 'kfp'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    try:
        PipelinesClient(**test_case.config)
        assert test_case.expected_status == SUCCESS
    except ImportError as e:
        assert test_case.expected_status == FAILED
        assert "pip install 'kubeflow[pipelines]'" in str(e)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="http URL infers port=8080 and verify_ssl=False",
            expected_status=SUCCESS,
            config={"base_url": "http://localhost"},
            expected_output={"host": "http://localhost:8080", "verify_ssl": False},
        ),
        TestCase(
            name="https URL infers port=443",
            expected_status=SUCCESS,
            config={"base_url": "https://ml-pipeline.example.com"},
            expected_output={"host": "https://ml-pipeline.example.com:443"},
        ),
        TestCase(
            name="explicit port overrides inference",
            expected_status=SUCCESS,
            config={"base_url": "http://localhost", "port": 9090},
            expected_output={"host": "http://localhost:9090", "verify_ssl": False},
        ),
        TestCase(
            name="user_token maps to existing_token",
            expected_status=SUCCESS,
            config={
                "base_url": "https://ml-pipeline.example.com",
                "user_token": "tok-123",
            },
            expected_output={
                "host": "https://ml-pipeline.example.com:443",
                "existing_token": "tok-123",
            },
        ),
        TestCase(
            name="custom_ca maps to ssl_ca_cert",
            expected_status=SUCCESS,
            config={
                "base_url": "https://ml-pipeline.example.com",
                "custom_ca": "/path/to/ca.pem",
            },
            expected_output={
                "host": "https://ml-pipeline.example.com:443",
                "ssl_ca_cert": "/path/to/ca.pem",
            },
        ),
        TestCase(
            name="namespace is forwarded",
            expected_status=SUCCESS,
            config={
                "base_url": "https://ml-pipeline.example.com",
                "namespace": "my-ns",
            },
            expected_output={
                "host": "https://ml-pipeline.example.com:443",
                "namespace": "my-ns",
            },
        ),
    ],
)
def test_init(test_case, monkeypatch):
    """Test PipelinesClient constructor maps params correctly to kfp.Client."""
    from kubeflow.pipelines.api.pipelines_client import PipelinesClient

    mock_kfp_class = MagicMock()
    monkeypatch.setattr("kfp.Client", mock_kfp_class)

    with patch(
        "kubeflow.pipelines.api.pipelines_client.PipelinesClient.__init__",
        wraps=PipelinesClient.__init__,
    ) as _:
        # Re-import to pick up the monkeypatched kfp.Client
        import importlib

        import kubeflow.pipelines.api.pipelines_client as mod

        importlib.reload(mod)

        pc = mod.PipelinesClient(**test_case.config)

        assert test_case.expected_status == SUCCESS
        mock_kfp_class.assert_called_once()
        call_kwargs = mock_kfp_class.call_args[1]
        for key, value in test_case.expected_output.items():
            assert call_kwargs[key] == value, f"Expected {key}={value}, got {call_kwargs.get(key)}"


# ------------------------------------------------------------------
# Health / context tests
# ------------------------------------------------------------------


def test_get_health(client, mock_kfp_client):
    """Test get_health delegates to get_kfp_healthz."""
    client.get_health()
    mock_kfp_client.get_kfp_healthz.assert_called_once()


def test_set_user_namespace(client, mock_kfp_client):
    """Test set_user_namespace delegates correctly."""
    client.set_user_namespace("new-ns")
    mock_kfp_client.set_user_namespace.assert_called_once_with("new-ns")


def test_get_user_namespace(client, mock_kfp_client):
    """Test get_user_namespace delegates correctly."""
    result = client.get_user_namespace()
    mock_kfp_client.get_user_namespace.assert_called_once()
    assert result == ""


# ------------------------------------------------------------------
# Experiment tests (name-first)
# ------------------------------------------------------------------


def test_create_experiment(client, mock_kfp_client):
    """Test create_experiment delegates correctly."""
    client.create_experiment("my-exp", description="desc", namespace="ns")
    mock_kfp_client.create_experiment.assert_called_once_with(
        name="my-exp", description="desc", namespace="ns"
    )


def test_list_experiments(client, mock_kfp_client):
    """Test list_experiments delegates correctly."""
    client.list_experiments(page_size=20, namespace="ns")
    mock_kfp_client.list_experiments.assert_called_once_with(
        page_token="",
        page_size=20,
        sort_by="",
        namespace="ns",
        filter=None,
    )


def test_get_experiment_by_name(client, mock_kfp_client):
    """Test get_experiment resolves by name."""
    client.get_experiment("my-exp")
    mock_kfp_client.get_experiment.assert_called_once_with(
        experiment_id=None, experiment_name="my-exp", namespace=None
    )


def test_get_experiment_by_id(client, mock_kfp_client):
    """Test get_experiment uses ID when provided."""
    client.get_experiment(experiment_id="exp-123")
    mock_kfp_client.get_experiment.assert_called_once_with(
        experiment_id="exp-123", experiment_name=None, namespace=None
    )


def test_archive_experiment_by_name(client, mock_kfp_client):
    """Test archive_experiment resolves name to ID."""
    client.archive_experiment("my-exp")
    mock_kfp_client.get_experiment.assert_called_once()
    mock_kfp_client.archive_experiment.assert_called_once_with(experiment_id="exp-id-123")


def test_delete_experiment_by_id(client, mock_kfp_client):
    """Test delete_experiment passes ID directly."""
    client.delete_experiment(experiment_id="exp-456")
    mock_kfp_client.get_experiment.assert_not_called()
    mock_kfp_client.delete_experiment.assert_called_once_with(experiment_id="exp-456")


# ------------------------------------------------------------------
# Pipeline tests (name-first)
# ------------------------------------------------------------------


def test_get_pipeline_by_name(client, mock_kfp_client):
    """Test get_pipeline resolves name to ID."""
    client.get_pipeline("my-pipeline")
    mock_kfp_client.get_pipeline_id.assert_called_once_with("my-pipeline")
    mock_kfp_client.get_pipeline.assert_called_once_with(pipeline_id="pipeline-id-123")


def test_get_pipeline_by_id(client, mock_kfp_client):
    """Test get_pipeline uses ID directly."""
    client.get_pipeline(pipeline_id="p-123")
    mock_kfp_client.get_pipeline_id.assert_not_called()
    mock_kfp_client.get_pipeline.assert_called_once_with(pipeline_id="p-123")


def test_get_pipeline_not_found(client, mock_kfp_client):
    """Test get_pipeline raises when name not found."""
    mock_kfp_client.get_pipeline_id.return_value = None
    with pytest.raises(ValueError, match="not found"):
        client.get_pipeline("nonexistent")


def test_delete_pipeline_by_name(client, mock_kfp_client):
    """Test delete_pipeline resolves name."""
    client.delete_pipeline("my-pipeline")
    mock_kfp_client.delete_pipeline.assert_called_once_with(pipeline_id="pipeline-id-123")


def test_list_pipeline_versions_by_name(client, mock_kfp_client):
    """Test list_pipeline_versions resolves name."""
    client.list_pipeline_versions("my-pipeline", page_size=5)
    mock_kfp_client.list_pipeline_versions.assert_called_once_with(
        pipeline_id="pipeline-id-123",
        page_token="",
        page_size=5,
        sort_by="",
        filter=None,
    )


def test_get_pipeline_requires_name_or_id(client):
    """Test that get_pipeline raises when neither name nor ID provided."""
    with pytest.raises(ValueError, match="Either name or pipeline_id"):
        client.get_pipeline()


# ------------------------------------------------------------------
# Upload tests
# ------------------------------------------------------------------


def test_upload_pipeline(client, mock_kfp_client):
    """Test upload_pipeline delegates correctly."""
    client.upload_pipeline("/path/to/pipeline.yaml", pipeline_name="my-pipe")
    mock_kfp_client.upload_pipeline.assert_called_once_with(
        pipeline_package_path="/path/to/pipeline.yaml",
        pipeline_name="my-pipe",
        description=None,
        namespace=None,
    )


def test_upload_pipeline_version(client, mock_kfp_client):
    """Test upload_pipeline_version delegates correctly."""
    client.upload_pipeline_version(
        "/path/to/pipeline.yaml", "v2", pipeline_name="my-pipe"
    )
    mock_kfp_client.upload_pipeline_version.assert_called_once_with(
        pipeline_package_path="/path/to/pipeline.yaml",
        pipeline_version_name="v2",
        pipeline_id=None,
        pipeline_name="my-pipe",
        description=None,
    )


# ------------------------------------------------------------------
# Run tests (ID-based)
# ------------------------------------------------------------------


def test_run_pipeline_with_experiment_name(client, mock_kfp_client):
    """Test run_pipeline resolves experiment name."""
    client.run_pipeline(
        "my-run",
        experiment_name="my-exp",
        pipeline_id="p-1",
        version_id="v-1",
    )
    mock_kfp_client.get_experiment.assert_called_once()
    mock_kfp_client.run_pipeline.assert_called_once()
    call_kwargs = mock_kfp_client.run_pipeline.call_args[1]
    assert call_kwargs["experiment_id"] == "exp-id-123"
    assert call_kwargs["job_name"] == "my-run"


def test_run_pipeline_with_experiment_id(client, mock_kfp_client):
    """Test run_pipeline passes experiment ID directly."""
    client.run_pipeline("my-run", experiment_id="e-1", pipeline_id="p-1", version_id="v-1")
    mock_kfp_client.run_pipeline.assert_called_once()
    call_kwargs = mock_kfp_client.run_pipeline.call_args[1]
    assert call_kwargs["experiment_id"] == "e-1"


def test_get_run(client, mock_kfp_client):
    """Test get_run delegates correctly."""
    client.get_run("run-123")
    mock_kfp_client.get_run.assert_called_once_with(run_id="run-123")


def test_wait_for_run_status_default(client, mock_kfp_client):
    """Test wait_for_run_status returns when run reaches terminal state."""
    mock_kfp_client.get_run.return_value = SimpleNamespace(state="Succeeded")
    result = client.wait_for_run_status("run-123", timeout=10)
    assert result.state == "Succeeded"


def test_wait_for_run_status_custom(client, mock_kfp_client):
    """Test wait_for_run_status with custom status set."""
    mock_kfp_client.get_run.return_value = SimpleNamespace(state="Running")
    result = client.wait_for_run_status("run-123", status={"running"}, timeout=10)
    assert result.state == "Running"


def test_wait_for_run_status_timeout(client, mock_kfp_client):
    """Test wait_for_run_status raises TimeoutError."""
    mock_kfp_client.get_run.return_value = SimpleNamespace(state="Pending")
    with pytest.raises(TimeoutError, match="did not reach"):
        client.wait_for_run_status("run-123", timeout=0, polling_interval=0)


def test_archive_run(client, mock_kfp_client):
    """Test archive_run delegates correctly."""
    client.archive_run("run-123")
    mock_kfp_client.archive_run.assert_called_once_with(run_id="run-123")


def test_terminate_run(client, mock_kfp_client):
    """Test terminate_run delegates correctly."""
    client.terminate_run("run-123")
    mock_kfp_client.terminate_run.assert_called_once_with(run_id="run-123")


# ------------------------------------------------------------------
# Recurring run tests (ID-based)
# ------------------------------------------------------------------


def test_create_recurring_run(client, mock_kfp_client):
    """Test create_recurring_run resolves experiment and delegates."""
    client.create_recurring_run(
        "my-schedule",
        experiment_name="my-exp",
        cron_expression="0 0 * * *",
        pipeline_id="p-1",
        version_id="v-1",
    )
    mock_kfp_client.create_recurring_run.assert_called_once()
    call_kwargs = mock_kfp_client.create_recurring_run.call_args[1]
    assert call_kwargs["experiment_id"] == "exp-id-123"
    assert call_kwargs["cron_expression"] == "0 0 * * *"


def test_get_recurring_run(client, mock_kfp_client):
    """Test get_recurring_run delegates correctly."""
    client.get_recurring_run("rr-123")
    mock_kfp_client.get_recurring_run.assert_called_once_with(recurring_run_id="rr-123")


def test_enable_recurring_run(client, mock_kfp_client):
    """Test enable_recurring_run delegates correctly."""
    client.enable_recurring_run("rr-123")
    mock_kfp_client.enable_recurring_run.assert_called_once_with(recurring_run_id="rr-123")


def test_disable_recurring_run(client, mock_kfp_client):
    """Test disable_recurring_run delegates correctly."""
    client.disable_recurring_run("rr-123")
    mock_kfp_client.disable_recurring_run.assert_called_once_with(recurring_run_id="rr-123")


def test_delete_recurring_run(client, mock_kfp_client):
    """Test delete_recurring_run delegates correctly."""
    client.delete_recurring_run("rr-123")
    mock_kfp_client.delete_recurring_run.assert_called_once_with(recurring_run_id="rr-123")
