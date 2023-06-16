# CS-678 Advanced NLP Project: DialQA

Folder Structure.

1. bert: https://github.com/google-research/bert

2. tydiqa: https://github.com/google-research-datasets/tydiqa

3. data: Make a data folder at root level and put data from https://github.com/ffaisal93/DialQA/tree/main/data inside the data folder.

4. multi_cased_L-12_H-768_A-12: This is the pretrained model we have used. Download it from https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip

Once you have done the above setup, your folder structure should look like this:

### Folder Structure
```
bert/
tydiqa/
    baseline/
    gold_passage_baseline/
        run_gold_passage_baseline.sh
        eval_gold_passage_baseline.sh
data/
	dialqa-train.json
	dialqa-dev-og.json
	dialqa-dev-aug.json
	audio/
		dev/
			{lang}/
				{dialect-region}/
					{lang}-{id}-{dialect-region}.wav
multi_cased_L-12_H-768_A-12/
README.md
```

Once your folder structure starts looking like this you can run and evaluate baseline using the below command:

1. Your PWD should be: tydiqa/gold_passage_baseline

2. Run: ./run_gold_passage_baseline.sh

3. Eval: ./eval_gold_passage_baseline.sh