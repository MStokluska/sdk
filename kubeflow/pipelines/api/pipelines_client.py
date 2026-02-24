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

from __future__ import annotations

import datetime
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kfp.components import base_component
    from kfp_server_api import (
        V2beta1Experiment,
        V2beta1GetHealthzResponse,
        V2beta1ListExperimentsResponse,
        V2beta1ListPipelineVersionsResponse,
        V2beta1ListPipelinesResponse,
        V2beta1ListRecurringRunsResponse,
        V2beta1ListRunsResponse,
        V2beta1Pipeline,
        V2beta1PipelineVersion,
        V2beta1RecurringRun,
        V2beta1Run,
    )

logger = logging.getLogger(__name__)

_DEFAULT_TERMINAL_STATES = frozenset({"succeeded", "failed", "skipped", "error"})


class PipelinesClient:
    """Client for Kubeflow Pipelines operations.

    Wraps ``kfp.Client`` with a name-first API for pipelines and experiments,
    a constructor aligned with ``ModelRegistryClient``, and DSL re-exports for
    the full author-to-monitor flow from a single SDK.

    Requires the kfp package to be installed. Install it with:

        pip install 'kubeflow[pipelines]'

    """

    def __init__(
        self,
        base_url: str,
        port: int | None = None,
        *,
        user_token: str | None = None,
        is_secure: bool | None = None,
        custom_ca: str | None = None,
        namespace: str | None = None,
    ):
        """Initialize the PipelinesClient.

        Args:
            base_url: Base URL of the KFP API server including scheme.
                     Examples: "https://ml-pipeline.example.com", "http://localhost"

        Keyword Args:
            port: Server port. If not provided, inferred from base_url scheme:
                 - https:// defaults to 443
                 - http:// defaults to 8080
                 - no scheme defaults to 443
            user_token: Bearer token for authentication.
            is_secure: Whether to use a secure connection. If not provided, inferred from base_url:
                      - https:// sets is_secure=True
                      - http:// sets is_secure=False
                      - no scheme defaults to True
            custom_ca: Path to the PEM-encoded root certificates as a string.
            namespace: Kubernetes namespace for multi-user deployments. Default ``None``
                      lets the server decide (single-user deployments should omit this).

        Raises:
            ImportError: If kfp is not installed.

        Examples:
            PipelinesClient("https://ml-pipeline.example.com")
            PipelinesClient("http://localhost", port=8888)
            PipelinesClient("https://ml-pipeline.example.com", user_token="eyJ...")
        """
        try:
            from kfp import Client as KfpClient
        except ImportError as e:
            raise ImportError(
                "kfp is not installed. Install it with:\n\n"
                "  pip install 'kubeflow[pipelines]'\n"
            ) from e

        is_http = base_url.startswith("http://")
        if is_secure is None:
            is_secure = not is_http
        if port is None:
            port = 8080 if is_http else 443

        host = base_url.rstrip("/")
        if f":{port}" not in host:
            host = f"{host}:{port}"

        kfp_kwargs: dict[str, Any] = {"host": host}
        if user_token is not None:
            kfp_kwargs["existing_token"] = user_token
        if custom_ca is not None:
            kfp_kwargs["ssl_ca_cert"] = custom_ca
        if not is_secure:
            kfp_kwargs["verify_ssl"] = False
        if namespace is not None:
            kfp_kwargs["namespace"] = namespace

        self._client = KfpClient(**kfp_kwargs)

    # ------------------------------------------------------------------
    # Name → ID resolution helpers
    # ------------------------------------------------------------------

    def _resolve_pipeline_id(
        self,
        name: str | None,
        pipeline_id: str | None,
    ) -> str:
        """Resolve a pipeline name to its ID, or validate that an ID was given."""
        if pipeline_id is not None:
            return pipeline_id
        if name is None:
            raise ValueError("Either name or pipeline_id is required.")
        resolved = self._client.get_pipeline_id(name)
        if resolved is None:
            raise ValueError(f"Pipeline {name!r} not found.")
        return resolved

    def _resolve_experiment_id(
        self,
        name: str | None,
        experiment_id: str | None,
        namespace: str | None = None,
    ) -> str:
        """Resolve an experiment name to its ID, or validate that an ID was given."""
        if experiment_id is not None:
            return experiment_id
        if name is None:
            raise ValueError("Either name or experiment_id is required.")
        exp = self._client.get_experiment(experiment_name=name, namespace=namespace)
        return exp.experiment_id

    # ------------------------------------------------------------------
    # Health / context
    # ------------------------------------------------------------------

    def get_health(self) -> V2beta1GetHealthzResponse:
        """Ping the KFP server and return deployment info.

        Returns:
            Health response including multi-user mode status.
        """
        return self._client.get_kfp_healthz()

    def set_user_namespace(self, namespace: str) -> None:
        """Set the namespace scope for subsequent API calls.

        Useful for multi-user deployments where you need to switch namespace
        mid-session without creating a new client.

        Args:
            namespace: Kubernetes namespace to use.
        """
        self._client.set_user_namespace(namespace)

    def get_user_namespace(self) -> str:
        """Return the currently configured namespace.

        Returns:
            The active namespace, or empty string if not set.
        """
        return self._client.get_user_namespace()

    # ------------------------------------------------------------------
    # Experiments (name-first, ID fallback)
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1Experiment:
        """Create an experiment (or return existing one with the same name).

        Args:
            name: Display name for the experiment.
            description: Optional description.
            namespace: Optional namespace override.

        Returns:
            The created or existing experiment.
        """
        return self._client.create_experiment(
            name=name, description=description, namespace=namespace
        )

    def list_experiments(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        namespace: str | None = None,
        filter: str | None = None,
    ) -> V2beta1ListExperimentsResponse:
        """List experiments.

        Args:
            page_token: Token for paginated responses.
            page_size: Number of results per page.
            sort_by: Sort string, e.g. ``'display_name desc'``.
            namespace: Optional namespace override.
            filter: JSON-serialized filter.

        Returns:
            Paginated list of experiments.
        """
        return self._client.list_experiments(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            namespace=namespace,
            filter=filter,
        )

    def get_experiment(
        self,
        name: str | None = None,
        *,
        experiment_id: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1Experiment:
        """Get an experiment by name or ID.

        Args:
            name: Display name of the experiment (preferred).
            experiment_id: ID of the experiment (fallback).
            namespace: Optional namespace override.

        Returns:
            The experiment.
        """
        return self._client.get_experiment(
            experiment_id=experiment_id, experiment_name=name, namespace=namespace
        )

    def archive_experiment(
        self,
        name: str | None = None,
        *,
        experiment_id: str | None = None,
    ) -> dict:
        """Archive an experiment by name or ID.

        Args:
            name: Display name of the experiment.
            experiment_id: ID of the experiment.

        Returns:
            Empty dictionary.
        """
        resolved_id = self._resolve_experiment_id(name, experiment_id)
        return self._client.archive_experiment(experiment_id=resolved_id)

    def unarchive_experiment(
        self,
        name: str | None = None,
        *,
        experiment_id: str | None = None,
    ) -> dict:
        """Unarchive an experiment by name or ID.

        Args:
            name: Display name of the experiment.
            experiment_id: ID of the experiment.

        Returns:
            Empty dictionary.
        """
        resolved_id = self._resolve_experiment_id(name, experiment_id)
        return self._client.unarchive_experiment(experiment_id=resolved_id)

    def delete_experiment(
        self,
        name: str | None = None,
        *,
        experiment_id: str | None = None,
    ) -> dict:
        """Delete an experiment by name or ID.

        Args:
            name: Display name of the experiment.
            experiment_id: ID of the experiment.

        Returns:
            Empty dictionary.
        """
        resolved_id = self._resolve_experiment_id(name, experiment_id)
        return self._client.delete_experiment(experiment_id=resolved_id)

    # ------------------------------------------------------------------
    # Pipelines (name-first, ID fallback)
    # ------------------------------------------------------------------

    def list_pipelines(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        filter: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1ListPipelinesResponse:
        """List pipelines.

        Args:
            page_token: Token for paginated responses.
            page_size: Number of results per page.
            sort_by: Sort string, e.g. ``'display_name desc'``.
            filter: JSON-serialized filter.
            namespace: Optional namespace override.

        Returns:
            Paginated list of pipelines.
        """
        return self._client.list_pipelines(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            filter=filter,
            namespace=namespace,
        )

    def get_pipeline(
        self,
        name: str | None = None,
        *,
        pipeline_id: str | None = None,
    ) -> V2beta1Pipeline:
        """Get a pipeline by name or ID.

        Args:
            name: Display name of the pipeline (preferred).
            pipeline_id: ID of the pipeline (fallback).

        Returns:
            The pipeline.
        """
        resolved_id = self._resolve_pipeline_id(name, pipeline_id)
        return self._client.get_pipeline(pipeline_id=resolved_id)

    def delete_pipeline(
        self,
        name: str | None = None,
        *,
        pipeline_id: str | None = None,
    ) -> dict:
        """Delete a pipeline by name or ID.

        Args:
            name: Display name of the pipeline.
            pipeline_id: ID of the pipeline.

        Returns:
            Empty dictionary.
        """
        resolved_id = self._resolve_pipeline_id(name, pipeline_id)
        return self._client.delete_pipeline(pipeline_id=resolved_id)

    def list_pipeline_versions(
        self,
        name: str | None = None,
        *,
        pipeline_id: str | None = None,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        filter: str | None = None,
    ) -> V2beta1ListPipelineVersionsResponse:
        """List versions of a pipeline by name or ID.

        Args:
            name: Display name of the pipeline.
            pipeline_id: ID of the pipeline.
            page_token: Token for paginated responses.
            page_size: Number of results per page.
            sort_by: Sort string.
            filter: JSON-serialized filter.

        Returns:
            Paginated list of pipeline versions.
        """
        resolved_id = self._resolve_pipeline_id(name, pipeline_id)
        return self._client.list_pipeline_versions(
            pipeline_id=resolved_id,
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            filter=filter,
        )

    def get_pipeline_version(
        self,
        pipeline_version_id: str,
        *,
        name: str | None = None,
        pipeline_id: str | None = None,
    ) -> V2beta1PipelineVersion:
        """Get a pipeline version.

        Args:
            pipeline_version_id: ID of the pipeline version.
            name: Display name of the parent pipeline.
            pipeline_id: ID of the parent pipeline.

        Returns:
            The pipeline version.
        """
        resolved_id = self._resolve_pipeline_id(name, pipeline_id)
        return self._client.get_pipeline_version(
            pipeline_id=resolved_id, pipeline_version_id=pipeline_version_id
        )

    def delete_pipeline_version(
        self,
        pipeline_version_id: str,
        *,
        name: str | None = None,
        pipeline_id: str | None = None,
    ) -> dict:
        """Delete a pipeline version.

        Args:
            pipeline_version_id: ID of the pipeline version.
            name: Display name of the parent pipeline.
            pipeline_id: ID of the parent pipeline.

        Returns:
            Empty dictionary.
        """
        resolved_id = self._resolve_pipeline_id(name, pipeline_id)
        return self._client.delete_pipeline_version(
            pipeline_id=resolved_id, pipeline_version_id=pipeline_version_id
        )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_pipeline(
        self,
        pipeline_package_path: str,
        pipeline_name: str | None = None,
        description: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1Pipeline:
        """Upload a pipeline from a local package file.

        Args:
            pipeline_package_path: Path to compiled pipeline (YAML/JSON/archive).
            pipeline_name: Display name. If omitted, extracted from the package.
            description: Optional description.
            namespace: Optional namespace override.

        Returns:
            The uploaded pipeline.
        """
        return self._client.upload_pipeline(
            pipeline_package_path=pipeline_package_path,
            pipeline_name=pipeline_name,
            description=description,
            namespace=namespace,
        )

    def upload_pipeline_version(
        self,
        pipeline_package_path: str,
        pipeline_version_name: str,
        *,
        pipeline_id: str | None = None,
        pipeline_name: str | None = None,
        description: str | None = None,
    ) -> V2beta1PipelineVersion:
        """Upload a new version of an existing pipeline.

        Specify the parent pipeline by name or ID.

        Args:
            pipeline_package_path: Path to compiled pipeline.
            pipeline_version_name: Display name for this version.
            pipeline_id: ID of the parent pipeline.
            pipeline_name: Name of the parent pipeline.
            description: Optional description.

        Returns:
            The uploaded pipeline version.
        """
        return self._client.upload_pipeline_version(
            pipeline_package_path=pipeline_package_path,
            pipeline_version_name=pipeline_version_name,
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            description=description,
        )

    def upload_pipeline_from_pipeline_func(
        self,
        pipeline_func: base_component.BaseComponent,
        pipeline_name: str | None = None,
        description: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1Pipeline:
        """Compile a pipeline function and upload it in one step.

        Args:
            pipeline_func: Function decorated with ``@dsl.pipeline``.
            pipeline_name: Display name. If omitted, uses the function name.
            description: Optional description.
            namespace: Optional namespace override.

        Returns:
            The uploaded pipeline.
        """
        return self._client.upload_pipeline_from_pipeline_func(
            pipeline_func=pipeline_func,
            pipeline_name=pipeline_name,
            description=description,
            namespace=namespace,
        )

    def upload_pipeline_version_from_pipeline_func(
        self,
        pipeline_func: base_component.BaseComponent,
        pipeline_version_name: str,
        *,
        pipeline_id: str | None = None,
        pipeline_name: str | None = None,
        description: str | None = None,
    ) -> V2beta1PipelineVersion:
        """Compile a pipeline function and upload as a new version.

        Args:
            pipeline_func: Function decorated with ``@dsl.pipeline``.
            pipeline_version_name: Display name for this version.
            pipeline_id: ID of the parent pipeline.
            pipeline_name: Name of the parent pipeline.
            description: Optional description.

        Returns:
            The uploaded pipeline version.
        """
        return self._client.upload_pipeline_version_from_pipeline_func(
            pipeline_func=pipeline_func,
            pipeline_version_name=pipeline_version_name,
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            description=description,
        )

    # ------------------------------------------------------------------
    # Runs (ID-based — run names aren't unique in KFP)
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        job_name: str,
        *,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        pipeline_package_path: str | None = None,
        params: dict[str, Any] | None = None,
        pipeline_id: str | None = None,
        version_id: str | None = None,
        pipeline_root: str | None = None,
        enable_caching: bool | None = None,
        service_account: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1Run:
        """Run a pipeline. Experiment can be specified by name or ID.

        Args:
            job_name: Display name for the run.
            experiment_name: Name of the experiment to run under.
            experiment_id: ID of the experiment (takes precedence over name).
            pipeline_package_path: Path to compiled pipeline.
            params: Pipeline parameters as a dict.
            pipeline_id: ID of the pipeline to run.
            version_id: ID of the pipeline version to run.
            pipeline_root: Root path for pipeline outputs.
            enable_caching: Whether to enable caching.
            service_account: Kubernetes service account for the run.
            namespace: Optional namespace for experiment resolution.

        Returns:
            The created run.
        """
        resolved_exp_id = self._resolve_experiment_id(
            experiment_name, experiment_id, namespace=namespace
        )
        return self._client.run_pipeline(
            experiment_id=resolved_exp_id,
            job_name=job_name,
            pipeline_package_path=pipeline_package_path,
            params=params,
            pipeline_id=pipeline_id,
            version_id=version_id,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account,
        )

    def run_pipeline_from_func(
        self,
        pipeline_func: base_component.BaseComponent,
        arguments: dict[str, Any] | None = None,
        run_name: str | None = None,
        *,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        namespace: str | None = None,
        pipeline_root: str | None = None,
        enable_caching: bool | None = None,
        service_account: str | None = None,
    ) -> Any:
        """Compile a pipeline function and run it in one step.

        Args:
            pipeline_func: Function decorated with ``@dsl.pipeline``.
            arguments: Pipeline arguments as a dict.
            run_name: Display name for the run.
            experiment_name: Name of the experiment.
            experiment_id: ID of the experiment.
            namespace: Optional namespace override.
            pipeline_root: Root path for pipeline outputs.
            enable_caching: Whether to enable caching.
            service_account: Kubernetes service account for the run.

        Returns:
            RunPipelineResult with run details.
        """
        return self._client.create_run_from_pipeline_func(
            pipeline_func=pipeline_func,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
            namespace=namespace,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account,
            experiment_id=experiment_id,
        )

    def run_pipeline_from_package(
        self,
        pipeline_file: str,
        arguments: dict[str, Any] | None = None,
        run_name: str | None = None,
        *,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        namespace: str | None = None,
        pipeline_root: str | None = None,
        enable_caching: bool | None = None,
        service_account: str | None = None,
    ) -> Any:
        """Run a pipeline from a compiled package file.

        Args:
            pipeline_file: Path to compiled pipeline package.
            arguments: Pipeline arguments as a dict.
            run_name: Display name for the run.
            experiment_name: Name of the experiment.
            experiment_id: ID of the experiment.
            namespace: Optional namespace override.
            pipeline_root: Root path for pipeline outputs.
            enable_caching: Whether to enable caching.
            service_account: Kubernetes service account for the run.

        Returns:
            RunPipelineResult with run details.
        """
        return self._client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=run_name,
            experiment_name=experiment_name,
            namespace=namespace,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account,
            experiment_id=experiment_id,
        )

    def list_runs(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        experiment_id: str | None = None,
        namespace: str | None = None,
        filter: str | None = None,
    ) -> V2beta1ListRunsResponse:
        """List runs.

        Args:
            page_token: Token for paginated responses.
            page_size: Number of results per page.
            sort_by: Sort string.
            experiment_id: Filter by experiment ID.
            namespace: Optional namespace override.
            filter: JSON-serialized filter.

        Returns:
            Paginated list of runs.
        """
        return self._client.list_runs(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
            namespace=namespace,
            filter=filter,
        )

    def get_run(self, run_id: str) -> V2beta1Run:
        """Get run details.

        Args:
            run_id: ID of the run.

        Returns:
            The run.
        """
        return self._client.get_run(run_id=run_id)

    def wait_for_run_status(
        self,
        run_id: str,
        timeout: int = 600,
        polling_interval: int = 5,
        status: set[str] | None = None,
    ) -> V2beta1Run:
        """Wait for a run to reach one of the target states.

        By default waits for any terminal state (succeeded, failed, skipped, error).
        Pass a custom ``status`` set to wait for specific states, e.g.
        ``{"running"}`` to wait until execution begins.

        Args:
            run_id: ID of the run.
            timeout: Maximum seconds to wait.
            polling_interval: Seconds between status checks.
            status: Set of target state names (lowercase). Defaults to all
                    terminal states.

        Returns:
            The run once it reaches a target state.

        Raises:
            TimeoutError: If the run does not reach a target state within the timeout.
        """
        target_states = {s.lower() for s in status} if status else _DEFAULT_TERMINAL_STATES
        start = datetime.datetime.now()
        while True:
            run = self._client.get_run(run_id=run_id)
            current_state = run.state
            if current_state is not None and current_state.lower() in target_states:
                return run
            elapsed = (datetime.datetime.now() - start).total_seconds()
            if elapsed > timeout:
                raise TimeoutError(
                    f"Run {run_id} did not reach {target_states} within {timeout}s "
                    f"(last state: {current_state})"
                )
            time.sleep(polling_interval)

    def archive_run(self, run_id: str) -> dict:
        """Archive a run.

        Args:
            run_id: ID of the run.

        Returns:
            Empty dictionary.
        """
        return self._client.archive_run(run_id=run_id)

    def unarchive_run(self, run_id: str) -> dict:
        """Unarchive a run.

        Args:
            run_id: ID of the run.

        Returns:
            Empty dictionary.
        """
        return self._client.unarchive_run(run_id=run_id)

    def delete_run(self, run_id: str) -> dict:
        """Delete a run.

        Args:
            run_id: ID of the run.

        Returns:
            Empty dictionary.
        """
        return self._client.delete_run(run_id=run_id)

    def terminate_run(self, run_id: str) -> dict:
        """Terminate a running pipeline execution.

        Args:
            run_id: ID of the run.

        Returns:
            Empty dictionary.
        """
        return self._client.terminate_run(run_id=run_id)

    # ------------------------------------------------------------------
    # Recurring runs (ID-based)
    # ------------------------------------------------------------------

    def create_recurring_run(
        self,
        job_name: str,
        *,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        description: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        interval_second: int | None = None,
        cron_expression: str | None = None,
        max_concurrency: int | None = 1,
        no_catchup: bool | None = None,
        params: dict[str, Any] | None = None,
        pipeline_package_path: str | None = None,
        pipeline_id: str | None = None,
        version_id: str | None = None,
        enabled: bool = True,
        pipeline_root: str | None = None,
        enable_caching: bool | None = None,
        service_account: str | None = None,
        namespace: str | None = None,
    ) -> V2beta1RecurringRun:
        """Create a recurring run (schedule).

        Exactly one of ``cron_expression`` or ``interval_second`` must be provided.
        Experiment can be specified by name or ID.

        Args:
            job_name: Display name for the recurring run.
            experiment_name: Name of the experiment.
            experiment_id: ID of the experiment.
            description: Optional description.
            start_time: RFC3339 start time.
            end_time: RFC3339 end time.
            interval_second: Seconds between runs (periodic schedule).
            cron_expression: Cron expression for the schedule.
            max_concurrency: Maximum parallel runs.
            no_catchup: Whether to skip missed intervals.
            params: Pipeline parameters as a dict.
            pipeline_package_path: Path to compiled pipeline.
            pipeline_id: ID of the pipeline.
            version_id: ID of the pipeline version.
            enabled: Whether the schedule starts active.
            pipeline_root: Root path for pipeline outputs.
            enable_caching: Whether to enable caching.
            service_account: Kubernetes service account.
            namespace: Optional namespace for experiment resolution.

        Returns:
            The created recurring run.
        """
        resolved_exp_id = self._resolve_experiment_id(
            experiment_name, experiment_id, namespace=namespace
        )
        return self._client.create_recurring_run(
            experiment_id=resolved_exp_id,
            job_name=job_name,
            description=description,
            start_time=start_time,
            end_time=end_time,
            interval_second=interval_second,
            cron_expression=cron_expression,
            max_concurrency=max_concurrency,
            no_catchup=no_catchup,
            params=params,
            pipeline_package_path=pipeline_package_path,
            pipeline_id=pipeline_id,
            version_id=version_id,
            enabled=enabled,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            service_account=service_account,
        )

    def list_recurring_runs(
        self,
        page_token: str = "",
        page_size: int = 10,
        sort_by: str = "",
        experiment_id: str | None = None,
        namespace: str | None = None,
        filter: str | None = None,
    ) -> V2beta1ListRecurringRunsResponse:
        """List recurring runs.

        Args:
            page_token: Token for paginated responses.
            page_size: Number of results per page.
            sort_by: Sort string.
            experiment_id: Filter by experiment ID.
            namespace: Optional namespace override.
            filter: JSON-serialized filter.

        Returns:
            Paginated list of recurring runs.
        """
        return self._client.list_recurring_runs(
            page_token=page_token,
            page_size=page_size,
            sort_by=sort_by,
            experiment_id=experiment_id,
            namespace=namespace,
            filter=filter,
        )

    def get_recurring_run(self, recurring_run_id: str) -> V2beta1RecurringRun:
        """Get recurring run details.

        Args:
            recurring_run_id: ID of the recurring run.

        Returns:
            The recurring run.
        """
        return self._client.get_recurring_run(recurring_run_id=recurring_run_id)

    def delete_recurring_run(self, recurring_run_id: str) -> dict:
        """Delete a recurring run.

        Args:
            recurring_run_id: ID of the recurring run.

        Returns:
            Empty dictionary.
        """
        return self._client.delete_recurring_run(recurring_run_id=recurring_run_id)

    def enable_recurring_run(self, recurring_run_id: str) -> dict:
        """Enable a recurring run.

        Args:
            recurring_run_id: ID of the recurring run.

        Returns:
            Empty dictionary.
        """
        return self._client.enable_recurring_run(recurring_run_id=recurring_run_id)

    def disable_recurring_run(self, recurring_run_id: str) -> dict:
        """Disable a recurring run.

        Args:
            recurring_run_id: ID of the recurring run.

        Returns:
            Empty dictionary.
        """
        return self._client.disable_recurring_run(recurring_run_id=recurring_run_id)
