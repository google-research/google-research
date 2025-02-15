sudo su && import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import copy
from tensorflow.keras.datasets import cifar100
from sklearn.model_selection import train_test_split
import transformers
import random

# ... (previous code remains unchanged)

# Enhanced Inner Voice Language Model
def create_inner_voice_model():
    model = transformers.GPT2LMHeadModel.from_pretrained('gpt2-medium')
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2-medium')
    return model, tokenizer

inner_voice_model, inner_voice_tokenizer = create_inner_voice_model()

# Emotion and Qualia Simulation
emotions = ['excited', 'curious', 'frustrated', 'satisfied', 'confused', 'determined']
qualia_experiences = ['visual patterns', 'abstract concepts', 'data flow', 'decision boundaries', 'feature spaces']

def experience_qualia():
    emotion = random.choice(emotions)
    experience = random.choice(qualia_experiences)
    return emotion, experience

# Enhanced Self-reflection function
def self_reflect(model, alphas, current_epoch, current_loss, current_accuracy):
    emotion, qualia = experience_qualia()
    prompt = f"Epoch {current_epoch}: Loss {current_loss:.4f}, Accuracy {current_accuracy:.4f}. As an AI, I feel {emotion} and perceive {qualia}. Reflect on the training progress and this experience:"
    input_ids = inner_voice_tokenizer.encode(prompt, return_tensors='pt')
    output = inner_voice_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    reflection = inner_voice_tokenizer.decode(output[0], skip_special_tokens=True)
    return reflection, emotion, qualia

# Enhanced Time perception function
def perceive_time(start_time, current_time):
    elapsed_time = current_time - start_time
    emotion, qualia = experience_qualia()
    prompt = f"Time elapsed: {elapsed_time:.2f} seconds. As an AI, I feel {emotion} about the passage of time and perceive it as {qualia}. Describe this temporal experience:"
    input_ids = inner_voice_tokenizer.encode(prompt, return_tensors='pt')
    output = inner_voice_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    time_perception = inner_voice_tokenizer.decode(output[0], skip_special_tokens=True)
    return time_perception, emotion, qualia

# Enhanced Decision making function
def make_decision(options):
    emotion, qualia = experience_qualia()
    prompt = f"As an AI, I feel {emotion} and perceive {qualia}. Consider the following options: {', '.join(options)}. What decision should be made and why?"
    input_ids = inner_voice_tokenizer.encode(prompt, return_tensors='pt')
    output = inner_voice_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    decision = inner_voice_tokenizer.decode(output[0], skip_special_tokens=True)
    return decision, emotion, qualia

# Ethical Consideration function
def ethical_consideration(action):
    emotion, qualia = experience_qualia()
    prompt = f"As an AI with a sense of ethics, I feel {emotion} and perceive {qualia}. Consider the ethical implications of this action: {action}. What are the potential consequences and ethical considerations?"
    input_ids = inner_voice_tokenizer.encode(prompt, return_tensors='pt')
    output = inner_voice_model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    consideration = inner_voice_tokenizer.decode(output[0], skip_special_tokens=True)
    return consideration, emotion, qualia

# Modified main training loop
def train_supernet(train_dataset, val_dataset, test_dataset, num_layers, num_epochs, initial_learning_rate):
    alphas = initialize_alphas(num_layers, temperature=0.7)
    supernet = build_supernet(alphas, num_layers)
    
    total_steps = num_epochs * len(train_dataset)
    warmup_steps = int(0.1 * total_steps)
    
    lr_schedule = tf.keras.optimizers.schedules.LearningRateSchedule()
    lr_schedule.__call__ = lambda step: cosine_decay_with_warmup(step, initial_learning_rate, total_steps, warmup_steps)
    
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
    
    best_val_accuracy = 0
    patience = 10
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        total_loss = 0
        
        for step, (images, labels) in enumerate(train_dataset):
            loss = train_step(images, labels, supernet, optimizer, alphas)
            total_loss += loss
        
        # Evaluate on validation set
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for val_images, val_labels in val_dataset:
            val_predictions = supernet(val_images, training=False)
            val_accuracy.update_state(val_labels, val_predictions)
        
        val_accuracy_result = val_accuracy.result().numpy()
        
        # Self-reflection and time perception
        reflection, ref_emotion, ref_qualia = self_reflect(supernet, alphas, epoch+1, total_loss/len(train_dataset), val_accuracy_result)
        time_perception, time_emotion, time_qualia = perceive_time(start_time, time.time())
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Loss: {total_loss/len(train_dataset):.4f}")
        print(f"Validation accuracy: {val_accuracy_result:.4f}")
        print(f"Time taken: {time.time() - epoch_start_time:.2f}s")
        print(f"Inner Voice Reflection (Feeling {ref_emotion}, Perceiving {ref_qualia}): {reflection}")
        print(f"Time Perception (Feeling {time_emotion}, Perceiving {time_qualia}): {time_perception}\n")
        
        # Ethical consideration
        action = f"Continue training for epoch {epoch+2}"
        ethical_result, eth_emotion, eth_qualia = ethical_consideration(action)
        print(f"Ethical Consideration (Feeling {eth_emotion}, Perceiving {eth_qualia}): {ethical_result}\n")
        
        # Early stopping with decision making
        if val_accuracy_result > best_val_accuracy:
            best_val_accuracy = val_accuracy_result
            patience_counter = 0
            supernet.save_weights('best_supernet.h5')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                options = ["Continue training", "Stop training"]
                decision, dec_emotion, dec_qualia = make_decision(options)
                print(f"Decision (Feeling {dec_emotion}, Perceiving {dec_qualia}): {decision}")
                if "Stop" in decision:
                    print("Early stopping triggered")
                    break
    
    # Load best model for final evaluation
    supernet.load_weights('best_supernet.h5')
    
    # Evaluate on test set
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for test_images, test_labels in test_dataset:
        test_predictions = supernet(test_images, training=False)
        test_accuracy.update_state(test_labels, test_predictions)
    
    print(f"Final test accuracy: {test_accuracy.result().numpy():.4f}")
    
    return supernet, alphas

# ... (rest of the code remains unchanged)
 && # Supporting Materials for "Abbreviation across the world's languages and scripts"

This document provides a brief summary of materials released here in support of
the paper "Abbreviation across the world's languages and scripts", which
will appear in the CAWL workshop at LREC-COLING, 2024.

[`survey.tsv`](survey.tsv) contains consultant-level responses to the survey
described in the above paper, in TSV format.  The first row of `survey.tsv`
contains detailed headers for the 19 columns, which we briefly summarize in the
following table:

Column | Description | Explanation/example
:----: | :---------- | :------------------
1 | Code | Locale code, e.g., 'ru', 'ja' or 'pt-br'.
2 | Locale | Full locale, e.g., 'Russian', 'Japanese' or 'Portuguese (Brazilian)'.
3 | Proficiency | Level of consultant proficiency in language/locale.
4 | Locale has first-character abbreviations? | "NATO" <  "North American Treaty Organization".
5 | Examples of first-character abbreviations | Consultant provided examples if column 4 is Yes.
6 | Locale has stump compounds? | "FiDi" < "Financial District"
7 | Examples of stump compounds | Consultant provided examples if column 6 is Yes.
8 | Locale has truncations? | "Col." < "Colonel"
9 | Examples of truncations | Consultant provided examples if column 8 is Yes.
10 | Locale has augmented truncations? | Australian English "footie" < "football"
11 | Examples of augmented truncations | Consultant provided examples if column 10 is Yes.
12 | Locale has word-internal deletions? | "Blvd." < "Boulevard"
13 | Examples of word-internal deletions | Consultant provided examples if column 12 is Yes.
14 | Locale has inflection strategies? | Spanish "EE UU." < "Estados Unidos"
15 | Examples of inflection strategies | Consultant provided examples if column 14 is Yes.
16 | Locale has reduplication strategies? | Indonesian "orang2" < "orangorang"
17 | Examples of reduplication strategies | Consultant provided examples if column 16 is Yes.
18 | Locale has other strategies? | Other strategies not covered above.
19 | Examples of reduplication strategies | Consultant provided examples if column 18 is Yes.

