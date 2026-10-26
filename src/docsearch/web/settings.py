"""Django settings for the local search page: no database, no sessions, no admin."""

import secrets

SECRET_KEY = secrets.token_hex(32)  # nothing is signed: the page has no login and no sessions
DEBUG = False
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", "testserver"]
ROOT_URLCONF = "docsearch.web.urls"
INSTALLED_APPS: list[str] = []
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "django.middleware.common.CommonMiddleware"]
DATABASES: dict = {}
USE_TZ = True
