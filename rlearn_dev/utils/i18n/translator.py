from typing import Dict
from .dictionary import dictionary as default_dictionary

class Translator:
    def __init__(self, to_lang='en',
                 dictionary: Dict[str, Dict[str, str]] | None = None):
        self.to_lang = to_lang
        self.dictionary = dictionary or default_dictionary
    
    def __call__(self, key, lang=None):
        return self.translate(key, lang)
    
    def translate(self, key, lang=None):
        return self.dictionary.get(key, {}).get(lang or self.to_lang, key)
    
    def add_translation(self, key, lang, translation):
        if key not in self.dictionary:
            self.dictionary[key] = {}
        self.dictionary[key][lang] = translation
    
    def remove_translation(self, key, lang):
        if key in self.dictionary and lang in self.dictionary[key]:
            del self.dictionary[key][lang]
            if not self.dictionary[key]:
                del self.dictionary[key]
    
    def get_dictionary(self):
        return self.dictionary
    
    def set_dictionary(self, dictionary: Dict[str, Dict[str, str]]):
        self.dictionary = dictionary

def translate(key, lang='en', dictionary: Dict[str, Dict[str, str]] | Translator | None = None):
    if isinstance(dictionary, Translator):
        return dictionary.translate(key, lang)
    elif isinstance(dictionary, dict):
        return dictionary.get(key, {}).get(lang, key)
    else:
        raise ValueError("Invalid dictionary type")