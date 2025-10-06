import os


TEMPLATE = """
testname: codec mushra test
testId: codecmushra
bufferSize: 2048
stopOnErrors: true
showButtonPreviousPage: true
remoteService: service/write.php

pages:

    - type: generic
      id: START
      name: 
      content: <ul> <li>This listening test is estimated to take ~20 mins.</li> <li>Please read and follow the directions.</li></ul>

    - type: volume
      id: vol
      name: Volume settings
      content: Please adjust the volume!
      stimulus: configs/resources/language_for_regularization/acoustic/wavs_16khz/TEST_FILE
      defaultVolume: 0.5

    - type: generic
      id: MUSHRA_DIRECTIONS
      name: MUSHRA Test Directions
      content: <ul> <li>In this listening test, you will mark <strong>10 sets of utterances</strong> </li> <li>Your task is to rate how similar the utterance is to a reference utterance on a scale from 0 to 100, where 0 means "the utterance is not similar at all" and 100 means "the utterance sounds the same". </li> <li>Use the full range of the scale and try to be consistent in your ratings. </li> <li>There are no right or wrong answers, we are interested in your honest opinion.</li></ul>

    -
        - random
ACOUSTIC_TEMPLATE

    - type: finish
      name: Thank you
      content: If you were told to input a code, please input it below. Otherwise, you can just input anything and finish the test.
      showResults: true
      writeResults: true
      questionnaire:
          - type: text
            label: code
            name: code
"""

ACOUSTIC_QUESTION = """        - type: mushra
          id: acousticNUMBER
          name: mushra - Random NUMBER
          content: Please rate how similar the utterance is to the reference. 0 is "not similar", 100 is "same".
          stimuli:
              C1: configs/resources/language_for_regularization/acoustic/2_5_asr_bs1/FILEBASENAME
              C2: configs/resources/language_for_regularization/acoustic/2_5_ttr_bs1/FILEBASENAME
              C3: configs/resources/language_for_regularization/acoustic/2_5_distill_bs1/FILEBASENAME
              C4: configs/resources/language_for_regularization/acoustic/2_5_bs1/FILEBASENAME
          enableLooping: true
          createAnchor35: true
          createAnchor70: false
          reference: configs/resources/language_for_regularization/acoustic/wavs_16khz/FILEBASENAME
"""

def fill_template(acoustic_fns):

    acoustic_parts = []
    for i, fn in enumerate(acoustic_fns):
        part = ACOUSTIC_QUESTION.replace("NUMBER", str(i+1)).replace("FILEBASENAME", fn)
        acoustic_parts.append(part)
    acoustic_str = "".join(acoustic_parts)

    final_str = TEMPLATE.replace("ACOUSTIC_TEMPLATE", acoustic_str).replace("TEST_FILE", fn)
    return final_str

def make_config(root):
    acoustic_root = os.path.join(root, "acoustic")

    semantic_texts = []

    acoustic_filelist = os.listdir(os.path.join(acoustic_root, "2_5_bs1"))
    acoustic_filebasenames = [os.path.basename(f) for f in acoustic_filelist if f.endswith(".wav")]

    filled = fill_template(acoustic_filebasenames)
    with open(os.path.join(root, "language_for_regularization_acoustic_config.yaml"), "w") as f:
        f.write(filled)
    
if __name__ == "__main__":
    make_config("z_final_listening_test")