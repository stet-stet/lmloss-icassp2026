import os


TEMPLATE = """
testname: acoustic and semantic test
testId: acoustic_semantic
bufferSize: 2048
stopOnErrors: true
showButtonPreviousPage: true
remoteService: service/write.php

pages:

    - type: generic
      id: START
      name: 
      content: <ul> <li>A <strong>semantic</strong> listening test will be administered. This is estimated to take ~10 mins for those with a good grasp of English.</li> <li>Please read and follow the directions.</li></ul>

    - type: volume
      id: vol
      name: Volume settings
      content: Please adjust the volume!
      stimulus: configs/resources/language_for_regularization/acoustic/wavs_16khz/TEST_FILE
      defaultVolume: 0.5

    - type: generic
      id: SEMANTIC_DIRECTIONS
      name: Semantic Test Directions
      content: <ul> <li>In this listening test, you will mark <strong>15 sets of utterances</strong> </li> <li>Your task is to rate how well the utterance matches the transcription, where 1 means "the utterance does not match the text at all" and 7 means "the utterance perfectly matches the text". </li> <li>Please consider only how well the utterance <strong>matches the text</strong>, and ignore any other aspects of the utterance such as audio quality or speaker characteristics. </li> <li>Use the full range of the scale and try to be consistent in your ratings. </li> <li>There are no right or wrong answers, we are interested in your honest opinion.</li></ul>

    -
        - random
SEMANTIC_TEMPLATE

    - type: finish
      name: Thank you
      content: If you were told to input a code, please input it below. Otherwise, you can just finish the test.
      showResults: true
      writeResults: true
      questionnaire:
          - type: text
            label: code
            name: code
"""

SEMANTIC_QUESTION = """        - type: likert_multi_stimulus
          id: semanticNUMBER
          name: Semantic - Random NUMBER
          content: Please rate how well each utterance matches this text: <br><br> "<strong>THETRANSCRIPT</strong>"
          stimuli:
              ref: configs/resources/language_for_regularization/semantic/wavs_16khz/FILEBASENAME
              C1: configs/resources/language_for_regularization/semantic/2_5_asr_bs1/FILEBASENAME
              C2: configs/resources/language_for_regularization/semantic/2_5_ttr_bs1/FILEBASENAME
              C3: configs/resources/language_for_regularization/semantic/2_5_distill_bs1/FILEBASENAME
              C4: configs/resources/language_for_regularization/semantic/2_5_bs1/FILEBASENAME
          mustRate: true
          response:
              - value: 1
                label: no matching words
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 2
                label: many unmatching words
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 3
                label: ...
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 4
                label: some phonemes gone/flipped
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 5
                label: ...
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 6
                label: matches, but strange pronunciations here and there
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
              - value: 7
                label: Perfectly matches the text
                img: configs/resources/images/star_off.png
                imgSelected: configs/resources/images/star_on.png
                imgHigherResponseSelected: configs/resources/images/star_on.png
"""

def fill_template(semantic_fns, semantic_texts):
    semantic_parts = []
    for i, (fn, text) in enumerate(zip(semantic_fns, semantic_texts)):
        part = SEMANTIC_QUESTION.replace("NUMBER", str(i+1)).replace("FILEBASENAME", fn).replace("THETRANSCRIPT", text)
        semantic_parts.append(part)
    semantic_str = "".join(semantic_parts)

    final_str = TEMPLATE.replace("SEMANTIC_TEMPLATE", semantic_str).replace("TEST_FILE", fn)
    return final_str

def make_config(root):
    semantic_root = os.path.join(root, "semantic")

    semantic_texts = []

    semantic_filelist = os.listdir(os.path.join(semantic_root, "2_5_bs1"))
    semantic_filebasenames = [os.path.basename(f) for f in semantic_filelist if f.endswith(".wav")]
    for sample in semantic_filebasenames:

        with open(os.path.join(semantic_root, "text", sample.replace(".wav", ".txt"))) as f:
            text = f.read().strip()
            semantic_texts.append(text)

    filled = fill_template(semantic_filebasenames, semantic_texts)
    with open(os.path.join(root, "language_for_regularization_semantic_config.yaml"), "w") as f:
        f.write(filled)
    
if __name__ == "__main__":
    make_config("z_final_listening_test")