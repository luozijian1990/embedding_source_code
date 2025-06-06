import os

from babel.support import Translations

from fastapi_admin.constants import BASE_DIR
from fastapi_admin.template import templates

TRANSLATIONS = {
    "zh_CN": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["zh_CN"]),
    "en_US": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["en_US"]),
    "es_PY": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["es_PY"]),
    "fr_FR": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["fr_FR"]),
    "fa_IR": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["fa_IR"]),
    "tr_TR": Translations.load(os.path.join(BASE_DIR, "locales"), locales=["tr_TR"]),
}

translations = TRANSLATIONS.get("en_US")


def set_locale(locale: str):
    global translations
    translations = TRANSLATIONS.get(locale) or TRANSLATIONS.get("en_US")
    templates.env.install_gettext_translations(translations)
    translations.install(locale)


def _(msg: str):
    return translations.ugettext(msg)
