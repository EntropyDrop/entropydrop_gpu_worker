import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

import s3_utils


def make_no_such_key_error():
    return ClientError(
        {
            "Error": {
                "Code": "NoSuchKey",
                "Message": "The specified key does not exist.",
            },
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "GetObject",
    )


def test_direct_no_such_key_is_terminal(monkeypatch):
    calls = []

    monkeypatch.setattr(s3_utils, "load_proxies", lambda: [])
    monkeypatch.setattr(
        s3_utils,
        "get_s3_client",
        lambda force_new=False: object(),
    )
    monkeypatch.setattr(
        s3_utils.time,
        "sleep",
        lambda seconds: pytest.fail("NoSuchKey must not sleep or retry"),
    )

    def action(client):
        calls.append(client)
        raise make_no_such_key_error()

    with pytest.raises(s3_utils.S3ObjectNotFoundError):
        s3_utils.execute_s3_with_failover("download_from_s3", action)

    assert len(calls) == 1


def test_proxy_no_such_key_is_verified_once_directly(monkeypatch):
    proxy_client = object()
    direct_client = object()
    calls = []

    monkeypatch.setattr(
        s3_utils,
        "load_proxies",
        lambda: ["http://127.0.0.1:9100"],
    )
    monkeypatch.setattr(
        s3_utils,
        "get_s3_client",
        lambda force_new=False: proxy_client,
    )
    monkeypatch.setattr(
        s3_utils.boto3,
        "client",
        lambda **kwargs: direct_client,
    )
    monkeypatch.setattr(
        s3_utils.time,
        "sleep",
        lambda seconds: pytest.fail("Confirmed NoSuchKey must not retry"),
    )

    def action(client):
        calls.append(client)
        raise make_no_such_key_error()

    with pytest.raises(s3_utils.S3ObjectNotFoundError):
        s3_utils.execute_s3_with_failover("download_from_s3", action)

    assert calls == [proxy_client, direct_client]


def test_network_failure_keeps_retrying_until_success(monkeypatch):
    attempts = 0
    sleeps = []

    monkeypatch.setattr(s3_utils, "load_proxies", lambda: [])
    monkeypatch.setattr(
        s3_utils,
        "get_s3_client",
        lambda force_new=False: object(),
    )
    monkeypatch.setattr(s3_utils.time, "sleep", sleeps.append)

    def action(client):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise EndpointConnectionError(endpoint_url="https://s3.test")
        return b"ok"

    result = s3_utils.execute_s3_with_failover(
        "download_from_s3",
        action,
    )

    assert result == b"ok"
    assert attempts == 3
    assert sleeps == [1, 2]


def test_cancellation_aborts_before_s3_request(monkeypatch):
    monkeypatch.setattr(
        s3_utils,
        "get_s3_client",
        lambda force_new=False: pytest.fail("S3 client must not be created"),
    )

    with pytest.raises(s3_utils.S3OperationAbortedError):
        s3_utils.execute_s3_with_failover(
            "download_from_s3",
            lambda client: pytest.fail("S3 action must not run"),
            abort_if=lambda: True,
        )
