mkdir -p inferred/gt/gt
for book in "LJ021" "LJ022" "LJ023" "LJ024" 
do 
    cp local_preprocessed_data/LJSpeech-1.1/wavs_16khz/${book}* inferred/gt/gt/
done