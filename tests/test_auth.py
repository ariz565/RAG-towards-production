"""Auth — token round-trip, tamper rejection, password verification.

(Requires runtime deps; runs in CI / a real env.)
"""

from app.services.auth import UserStore, create_token, verify_token


def test_token_roundtrip(tmp_path):
    store = UserStore(str(tmp_path / "users.json"))
    user = store.create("me@acme.com", "supersecret")
    payload = verify_token(create_token(user))
    assert payload is not None
    assert payload["tenant"] == user.tenant_id
    assert payload["typ"] == "access" and payload["iss"] == "vision"


def test_tampered_token_rejected():
    assert verify_token("not-a-real.token") is None
    assert verify_token("") is None


def test_password_verification(tmp_path):
    store = UserStore(str(tmp_path / "users.json"))
    store.create("a@b.com", "supersecret")
    assert store.verify("a@b.com", "wrong-password") is None
    assert store.verify("a@b.com", "supersecret") is not None


def test_duplicate_email_rejected(tmp_path):
    store = UserStore(str(tmp_path / "users.json"))
    store.create("dup@b.com", "supersecret")
    try:
        store.create("dup@b.com", "supersecret")
        assert False, "expected duplicate-email error"
    except ValueError:
        pass
