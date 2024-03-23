from transformers import AutoTokenizer, AutoModelForQuestionAnswering,default_data_collator
from torch.utils.data import DataLoader
from utils import *
import utils
from datasets import load_dataset

import torch

if torch.cuda.is_available():
    print(f"Total GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs found.")

squad_train = "data/Spoken-SQuAD/spoken_train-v1.1.json"
squad_test = "data/Spoken-SQuAD/spoken_test-v1.1.json"
squad_test_WER44 = "data/Spoken-SQuAD/spoken_test-v1.1_WER44.json"
squad_test_WER54 = "data/Spoken-SQuAD/spoken_test-v1.1_WER54.json"

# parse the original data files, save them for future loading
squad_train_parsed = parse_json(squad_train)
squad_test_parsed = parse_json(squad_test)
squad_test_WER44_parsed = parse_json(squad_test_WER44)
squad_test_WER54_parsed = parse_json(squad_test_WER54)

# create the dataset
dataset = load_dataset('json',
                                    data_files= { 'train': squad_train_parsed,
                                                  'test': squad_test_parsed,
                                                  'test_WER44': squad_test_WER44_parsed,
                                                  'test_WER54': squad_test_WER54_parsed },
                                    field = 'data')

# create the model from bert-base-uncased
# Automatically add one linear layer (,2) to map the results to start and end position
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
print(model)

print("Preprocessing training data...")
train_dataset = dataset['train'].map(
    preprocess_training_data,
    batched = True,
    remove_columns=dataset['train'].column_names
)

print("Preprocessing test data (NO NOISE: 22.73% WER)...")
test_dataset = dataset['test'].map(
    process_test_data,
    batched = True,
    remove_columns=dataset['test'].column_names
)

print("Preprocessing V1 noise test data (44.22% WER)...")
test_WER44_dataset = dataset['test_WER44'].map(
    process_test_data,
    batched = True,
    remove_columns=dataset['test_WER44'].column_names
)

print("Preprocessing V2 noise test data (54.82% WER)...")
test_WER54_dataset = dataset['test_WER54'].map(
    process_test_data,
    batched = True,
    remove_columns=dataset['test_WER54'].column_names
)

train_dataset.set_format("torch")
# remove the example_id, offset_mappling when creating the dataloader
test_set = test_dataset.remove_columns(["example_id", "offset_mapping"])
test_set.set_format("torch")
test_WER44_set = test_WER44_dataset.remove_columns(["example_id", "offset_mapping"])
test_WER44_set.set_format("torch")
test_WER54_set = test_WER54_dataset.remove_columns(["example_id", "offset_mapping"])
test_WER54_set.set_format("torch")





print("Creating train dataloader...")
train_dataloader = DataLoader(
    train_dataset,
    shuffle = True,
    collate_fn=default_data_collator,
    batch_size=64
)

print("Creating test dataloader...")
test_dataloader = DataLoader(
    test_set, collate_fn=default_data_collator, batch_size=8
)
print("Creating test V1 dataloader...")
test_WER44_dataloader = DataLoader(
    test_WER44_set, collate_fn=default_data_collator, batch_size=8
)
print("Creating test V2 dataloader...")
test_WER54_dataloader = DataLoader(
    test_WER54_set, collate_fn=default_data_collator, batch_size=8
)

output_dir = "bert-base-uncased-finetuned"

train_model(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            test_dataset=test_dataset,
            dataset=dataset,
            epochs=2,
            output_dir=output_dir,)

print("Evaluating model on Test Set...")
test_metrics = evaluate_model(model, test_dataloader, test_dataset, dataset['test'])
print("Evaluating model on Test V1 Set...")
test_v1_metrics = evaluate_model(model, test_WER44_dataloader, test_WER44_dataset, dataset['test_WER44'])
print("Evaluating model on Test V2 Set...")
test_v2_metrics = evaluate_model(model, test_WER54_dataloader, test_WER54_dataset, dataset['test_WER54'])

print("============= RESULTS =============")
print("Test Set    (NO NOISE - 22.73% WER) - exact match: " + str(test_metrics['exact_match']) + ", F1 score: " + str(test_metrics['f1']))
print("Test V1 Set (V1 NOISE - 44.22% WER) - exact match: " + str(test_v1_metrics['exact_match']) + ", F1 score: " + str(test_v1_metrics['f1']))
print("Test V2 Set (V2 NOISE - 54.82% WER) - exact match: " + str(test_v2_metrics['exact_match']) + ", F1 score: " + str(test_v2_metrics['f1']))
print("===================================")


