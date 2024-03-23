import json
import os
import evaluate
import collections
from tqdm.auto import tqdm
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer

# Hyper parameter
max_length = 384
stride = 64
metric = evaluate.load("squad")

n_best = 20
max_answer_length = 30
def parse_json(json_file):
    # {"data":[{"id": ,"title": ,....},
    #           ]}
    if os.path.exists(json_file.replace(".json", "_parsed.json")):
        print(f"{json_file} has already been parsed.")
        return json_file.replace(".json", "_parsed.json")

    print(f"Parsing {json_file}...")
    output_file = json_file.replace(".json", "_parsed.json")
    with open(json_file, "r") as f:
        json_data = json.load(f)
    output_list = []
    for elem in json_data["data"]:
        title = elem["title"]
        for paragraph in elem["paragraphs"]:
            context = paragraph["context"]
            qas = paragraph["qas"]
            for qa in qas:
                id = qa["id"]
                question = qa["question"]

                # one questions can have several answers

                answers = {"answer_start": [answer["answer_start"] for answer in qa["answers"]], "text": [answer["text"] for answer in qa["answers"]]}


                output = {
                    "id":id,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answers": answers,
                }
                output_list.append(output)
    with open(output_file, "w") as f:
        json.dump({"data": output_list}, f)
    return output_file

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer
def preprocess_training_data(data):
    questions = [question.strip() for question in data['question']]
    tokenizer = get_tokenizer()
    inputs = tokenizer(
        questions,
        data['context'],
        max_length=max_length,
        truncation='only_second',
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )

    offset_mapping = inputs.pop('offset_mapping')
    sample_map = inputs.pop('overflow_to_sample_mapping')
    answers = data['answers']
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = answer['answer_start'][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # find start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if answer not fully inside context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs


def process_test_data(data):
    questions = [question.strip() for question in data['question']]
    tokenizer = get_tokenizer()
    inputs = tokenizer(
        questions,
        data['context'],
        max_length = max_length,
        truncation = 'only_second',
        stride = stride,
        return_overflowing_tokens = True,
        return_offsets_mapping=True,
        padding = 'max_length'
    )

    sample_map = inputs.pop('overflow_to_sample_mapping')
    example_ids = []

    for i in range(len(inputs['input_ids'])):
        sample_idx = sample_map[i]
        example_ids.append(data["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offsets = inputs['offset_mapping'][i]
        inputs["offset_mapping"][i] = [
            offset if sequence_ids[k] == 1 else None for k, offset in enumerate(offsets)
        ]

    inputs['example_id'] = example_ids
    return inputs



def compute_metrics(start_logits, end_logits, features, examples, metric):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # loop thru all features associated with example ID
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # skip answers with a length that is either <0 or >max_answer_length
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index]
                    }
                    answers.append(answer)
        # select answer with best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def evaluate_model(model, dataloader, dataset, dataset_before_preprocessing, accelerator=None):
    if not accelerator:
        accelerator = Accelerator(mixed_precision='fp16')
        model, dataloader = accelerator.prepare(
            model, dataloader
        )

    model.eval()
    start_logits = []
    end_logits = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(dataset)]
    end_logits = end_logits[: len(dataset)]

    metrics = compute_metrics(
        start_logits, end_logits, dataset, dataset_before_preprocessing
    )
    return metrics


def train_model(model, train_dataloader, test_dataloader,test_dataset,dataset, epochs, output_dir):
    training_steps = epochs * len(train_dataloader)

    accelerator = Accelerator(mixed_precision='fp16')
    optimizer = AdamW(model.parameters(), lr = 2e-5)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps,
    )

    progress_bar = tqdm(range(training_steps))

    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        accelerator.print("Evaluation...")
        metrics = evaluate_model(model, eval_dataloader, test_dataset, dataset['validation'], accelerator)
        print(f"epoch {epoch}:", metrics)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
