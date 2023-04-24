import json

import argostranslate.package
import argostranslate.translate


def translate(text: str, from_code: str, to_code: str):
    try:
        from_code = from_code
        to_code = to_code

        # Download and install Argos Translate package
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())
        # Translate
        translatedText = argostranslate.translate.translate(text, from_code, to_code)
    except Exception:
        return {"message": "There is exception translating the text"}

    return json.dump({'translation': translatedText})