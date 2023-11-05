from __future__ import unicode_literals, print_function, division

import translation_model


test_sentence = "Now you are getting very nasty"
model = translation_model.TranslationModel()
sent = (model.translate(test_sentence))

print(sent)
