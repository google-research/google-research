# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import json
import torch
import shutil
import socket
import argparse
import random
import logging
import datetime
import wandb
from overall_metrics import overall_metrics
from merge import lora_merge
from evaluate import evaluate, evaluate_test, update_only_one_or_two, lora_weight_visualize
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

def log_with_flush(message, level=logging.INFO):
    """
    log with flush
    """
    logging.log(level, message)
    logging.getLogger().handlers[0].flush()

def curret_time_string():
    """Return the current time string."""
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return current_time

def assign_gpu(num_gpus, process_idx, total_processes):
    """Assign GPU to process."""
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

def initialize_search_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge, starting_velocity_mode):
    """
    Initialize a directory in search/ for the Model Swarms search.
    search_pass_name: the name of the search pass
    particle_paths: the paths of the particles, a list
    eval_type: the evaluation type
    dataset: the dataset
    gpus: the gpus available, a list
    base_model: the base model of the lora experts
    fast_merge: whether to use fast merge by only loading the safetensor file
    starting_velocity_mode: the starting velocity mode: zero, random, best
    """
    for i in range(len(particle_paths)):
        os.mkdir(os.path.join("search", search_pass_name, "particle_"+str(i)))
        for checkpoint_type in ["personal_best", "now", "velocity"]:
            os.mkdir(os.path.join("search", search_pass_name, "particle_"+str(i), checkpoint_type))
    os.mkdir(os.path.join("search", search_pass_name, "global_best")) # weights directly in this folder
    os.mkdir(os.path.join("search", search_pass_name, "global_worst")) # weights directly in this folder
    utility_scratchpad = {"g": None, "g_worst": None, "g_history": []}
    for i in range(len(particle_paths)):
        utility_scratchpad[f"particle_{i}_now"] = None
        utility_scratchpad[f"particle_{i}_best"] = None
        utility_scratchpad[f"particle_{i}_history"] = []
    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)

    # initialize particle now weights and personal_best
    for i in range(len(particle_paths)):
        shutil.copytree(particle_paths[i], os.path.join("search", search_pass_name, "particle_"+str(i), "now"), dirs_exist_ok=True)
        shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), dirs_exist_ok=True)

    # initialize particle now velocity
    if starting_velocity_mode == "zero":
        merge_args = []
        for i in range(len(particle_paths)):
            merge_args.append(([0], [os.path.join("search", search_pass_name, "particle_"+str(i), "now")], os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge))

        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
    # the default setting
    elif starting_velocity_mode == "random":
        merge_args = []
        for i in range(len(particle_paths)):
            secret_lover_id = random.randint(0, len(particle_paths)-1)
            while secret_lover_id == i:
                secret_lover_id = random.randint(0, len(particle_paths)-1)
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(secret_lover_id), "now")], os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge))

        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
    elif starting_velocity_mode == "best":
        # wait for starting validation utility eval
        pass

    # evaluate the utility of starting particles
    eval_args = []
    for i in range(len(particle_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, True))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
        utility_scratchpad = json.load(f)
    utility_scratchpad["g"] = max(results)
    utility_scratchpad["g_worst"] = min(results)
    utility_scratchpad["g_history"].append(utility_scratchpad["g"])

    for i in range(len(particle_paths)):
        utility_scratchpad[f"particle_{i}_now"] = results[i]
        utility_scratchpad[f"particle_{i}_best"] = results[i]
        utility_scratchpad[f"particle_{i}_history"].append(results[i])

    # logging at iteration=0
    wandb_log = {
        "g": utility_scratchpad["g"],
        "g_worst": utility_scratchpad["g_worst"],
    }
    for i in range(len(particle_paths)):
        wandb_log["particle_" + str(i) + "_now"] = utility_scratchpad["particle_" + str(i) + "_now"]

    wandb.log(wandb_log)

    with open(os.path.join("search", search_pass_name, "utility_scratchpad.json"), "w") as f:
        json.dump(utility_scratchpad, f, indent=4)

    # initialize global best checkpoint
    best_idx = results.index(max(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(best_idx), "now"), os.path.join("search", search_pass_name, "global_best"), dirs_exist_ok=True)

    # initialize global worst checkpoint
    worst_idx = results.index(min(results))
    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(worst_idx), "now"), os.path.join("search", search_pass_name, "global_worst"), dirs_exist_ok=True)

    if starting_velocity_mode == "best":
        global_best_path = os.path.join("search", search_pass_name, "global_best")
        merge_args = []
        for i in range(len(particle_paths)):
            merge_args.append(([-1, 1], [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), global_best_path], os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(particle_paths))], fast_merge))

        pool = Pool(processes=1)
        pool.starmap(lora_merge, merge_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

def particle_update(i, gpu_id, search_pass_name, weight_randomess, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag):
    """
    the main juice of the Model Swarms search: update velocity then update position of particles
    i: the index of the particle, bounded by max particle count
    gpu_id: the gpu id to use
    search_pass_name: the name of the search pass
    weight_randomess: whether to use weight randomess
    inertia: inertia of particle weight update
    cognitive_coeff: cognitive coefficient of particle weight update
    social_coeff: social coefficient of particle weight update
    repel_coeff: repel coefficient of particle weight update
    fast_merge: whether to use fast merge by only loading the safetensor file
    step_length: step length of the search in the direction of velocity
    """

    # log_with_flush("particle "+str(i)+" update starting!")

    particle_path = os.path.join("search", search_pass_name, "particle_"+str(i))
    now_path = os.path.join(particle_path, "now")
    best_path = os.path.join(particle_path, "personal_best")
    velocity_path = os.path.join(particle_path, "velocity")

    if restart_flag:
        shutil.copytree(best_path, now_path, dirs_exist_ok=True)
        lora_merge([0], [now_path], velocity_path, gpu_id, fast_merge)

    # weight randomness
    if weight_randomess == 1:
        r_w = random.uniform(0, 1)
        r_p = random.uniform(0, 1)
        r_s = random.uniform(0, 1)
        r_b = random.uniform(0, 1) # b for bad, repel term weight
    else:
        r_w = 1
        r_p = 1
        r_s = 1
        r_b = 1

    # weight normalize
    self_weight = r_w * inertia
    cognitive_weight = r_p * cognitive_coeff
    social_weight = r_s * social_coeff
    repel_weight = r_b * repel_coeff if repel_term else 0
    weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

    # normalize weights
    self_weight = self_weight / weight_sum
    cognitive_weight = cognitive_weight / weight_sum
    social_weight = social_weight / weight_sum
    repel_weight = repel_weight / weight_sum

    # p_i-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "p_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # g-x_i task vector
    lora_merge(
        weights = [1, -1],
        lora_name_list = [os.path.join("search", search_pass_name, "global_best"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "g_x"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # x_i - w task vector
    lora_merge(
        weights = [-1, 1],
        lora_name_list = [os.path.join("search", search_pass_name, "global_worst"), os.path.join("search", search_pass_name, "particle_"+str(i), "now")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "x_w"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update velocity
    lora_merge(
        weights = [self_weight, cognitive_weight, social_weight, repel_weight],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "p_x"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "g_x"),
                            os.path.join("search", search_pass_name, "particle_"+str(i), "x_w")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "velocity"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # update current position
    lora_merge(
        weights = [1, step_length],
        lora_name_list = [os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "velocity")],
        output_name = os.path.join("search", search_pass_name, "particle_"+str(i), "now"),
        gpu_id = gpu_id,
        directly_load_safetensors = fast_merge
    )

    # log_with_flush("particle_"+str(i)+" updated")

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-n", "--name", help="name of this model swarms search, also directory name in search/")
    argParser.add_argument("-e", "--eval_type", help="evaluation types") # multiple_choice, exact_match, multitask, rm_default, rm_verbose, rm_concise, human
    argParser.add_argument("-d", "--dataset", help="dataset as the search objective/evaluation") # file names in data/eval, be mindful of using the right --eval_type
    argParser.add_argument("-g", "--gpus", help="available gpu ids in a string") # such as 0,1,2,3,4
    argParser.add_argument("--num_cpu_when_merging", default=1, help="number of cpu cores when merging") # you don't need to change this honestly
    argParser.add_argument("--inertia", default = 0.4, help="inertia of particle weight update")
    argParser.add_argument("--cognitive_coeff", default = 0.3, help="cognitive coefficient of particle weight update")
    argParser.add_argument("--social_coeff", default = 0.3, help="social coefficient of particle weight update")
    argParser.add_argument("--repel_coeff", default = 0.3, help="repel coefficient of particle weight update")
    argParser.add_argument("--step_length", default = 1, help="step length of the search in the direction of velocity")
    argParser.add_argument("-p", "--patience", default = 10, help="patience of the search")
    argParser.add_argument("-m", "--max_iteration", default = 200, help="max iteration of the search")
    argParser.add_argument("--weight_randomess", default = 1, help="whether to use weight randomess") # 0, 1
    argParser.add_argument("-i", "--initial_expert_directory", default="./initial_experts", help="initial expert directory") # make it a directory of initial expert checkpoints, see initial_experts/ for example
    argParser.add_argument("-b", "--base_model", default="google/gemma-7b-it", help="base model of the lora experts")
    argParser.add_argument("--starting_test_set_eval", default=0, help="starting test set evaluation") # 0, 1
    argParser.add_argument("--fast_merge", default=1, help="whether to use fast merge by only loading the safetensor file") # just keep it 1 unless you absolutely know what you're doing
    argParser.add_argument("--project_name_wb", default="swarm", help="wandb project name") # as you wish
    argParser.add_argument("--populate_initial_experts", default=0, help="whether to populate initial experts") # 0, 1
    argParser.add_argument("--initial_experts_num", default=None, help="number of initial experts to populate, when populate flag is 1")
    argParser.add_argument("--starting_velocity_mode", default="random", help="starting velocity mode: zero, random, best") # zero, random, best
    argParser.add_argument("--repel_term", default=1, help="whether to incorporate a repel term with global_worst") # 0, 1
    argParser.add_argument("--step_length_factor", default=0.95, help="step length *= step_length_factor every iteration") # 1 for no scheduling, 0.95 maybe?
    argParser.add_argument("--minimum_step_length", default=0.1, help="minimum step length")
    argParser.add_argument("--restart_stray_particles", default=1, help="whether to restart stray particles") # 0, 1
    argParser.add_argument("--restart_patience", default=0.5, help="restart patience * patience = when to restart particles")
    argParser.add_argument("--clean_up_on_end", default=1, help="whether to clean up on end") # 0, 1
    argParser.add_argument("--only_one_or_two", default=None, help="whether to only optimize with dataset 1 or 2 in multitask") # safely ignore this
    argParser.add_argument("--to_visualize", default=False, help="whether to visualize the search process") # 0, 1, for Fig 8
    argParser.add_argument("--correctness_emergence", default=False, help="whether to track correctness changes wrt iteration") # 0, 1, for Fig 2
    argParser.add_argument("--dropK", default=0, help="dropout-K, 0-1") # for fig 9
    argParser.add_argument("--dropN", default=0, help="dropout-N, 0-1") # for fig 9

    args = argParser.parse_args()
    search_pass_name = args.name
    eval_type = args.eval_type
    dataset = args.dataset
    gpus = args.gpus
    num_cpu_when_merging = int(args.num_cpu_when_merging)
    inertia = float(args.inertia)
    cognitive_coeff = float(args.cognitive_coeff)
    social_coeff = float(args.social_coeff)
    repel_coeff = float(args.repel_coeff)
    patience = int(args.patience)
    step_length = float(args.step_length)
    max_iteration = int(args.max_iteration)
    weight_randomess = int(args.weight_randomess)
    initial_expert_directory = args.initial_expert_directory
    base_model = args.base_model
    starting_test_set_eval = int(args.starting_test_set_eval)
    fast_merge = int(args.fast_merge)
    project_name_wb = args.project_name_wb
    populate_initial_experts = int(args.populate_initial_experts)
    try:
        initial_experts_num = int(args.initial_experts_num)
    except:
        initial_experts_num = None
    starting_velocity_mode = args.starting_velocity_mode
    repel_term = int(args.repel_term)
    step_length_factor = float(args.step_length_factor)
    minimum_step_length = float(args.minimum_step_length)
    restart_stray_particles = int(args.restart_stray_particles)
    restart_patience = float(args.restart_patience)
    clean_up_on_end = int(args.clean_up_on_end)
    only_one_or_two = args.only_one_or_two
    update_only_one_or_two(only_one_or_two)
    to_visualize_flag = args.to_visualize
    correctness_emergence = args.correctness_emergence
    dropK = float(args.dropK)
    dropN = float(args.dropN)

    search_pass_name += ("_" + socket.gethostname())
    args.name = search_pass_name

    perplexity_extrinsic_test_dict = {
        "legal": ["hearsay", "citation_prediction_classification"],
        "medical": ["medqa", "medmcqa"],
        "science": ["scifact", "stem"],
        "culture": ["normad_country", "normad_value"]
    }

    # create search directory

    if os.path.exists(os.path.join("search", search_pass_name)):
        search_pass_name += curret_time_string().replace(" ", "_")
        # exit("search directory already exists!")
    os.mkdir(os.path.join("search", search_pass_name))

    # write args to file
    with open(os.path.join("search", args.name, "args.txt"), "w") as f:
        f.write(str(args))

    run = wandb.init(name=search_pass_name, project=project_name_wb)
    run.config.update(args)
    torch.multiprocessing.set_start_method('spawn')
    random.seed(42)
    # Configure logging to write to a file
    logging.basicConfig(filename=os.path.join("search", search_pass_name, "log.txt"), level=logging.DEBUG)

    gpus = [int(gpu) for gpu in gpus.split(",")]
    particle_paths = []
    for particle_path in os.listdir(initial_expert_directory):
        if os.path.isdir(os.path.join(initial_expert_directory, particle_path)):
            particle_paths.append(os.path.join(initial_expert_directory, particle_path))
    particle_paths = sorted(particle_paths)

    # populate initial experts
    if populate_initial_experts and initial_experts_num and len(particle_paths) < initial_experts_num:
        log_with_flush("populating initial experts...")
        log_with_flush("previously " + str(len(particle_paths)) + " experts")
        log_with_flush("now " + str(initial_experts_num))
        log_with_flush("adding " + str(initial_experts_num - len(particle_paths)) + " experts")

        os.mkdir(os.path.join("search", search_pass_name, "tmp"))
        particles_now = len(particle_paths)
        for i in range(initial_experts_num - particles_now):
            parent_1 = random.choice(particle_paths)
            parent_2 = random.choice(particle_paths)
            while parent_1 == parent_2:
                parent_2 = random.choice(particle_paths)
            child_path = os.path.join("search", search_pass_name, "tmp", "child_"+str(i))
            w_1 = random.random() * 2 # half interpolation, half extrapolation
            w_2 = 1 - w_1
            shutil.copytree(parent_1, child_path)
            lora_merge([w_1, w_2], [parent_1, parent_2], child_path, gpus[0], fast_merge)
            particle_paths.append(child_path)

    correctness_emergence_dict = {}
    if correctness_emergence:
        correctness_emergence_dict = {}
        for i in range(len(particle_paths)):
            correctness_emergence_dict[i] = []

    particle_trajectory = {}
    if to_visualize_flag:
        particle_trajectory = {}
        for i in range(len(particle_paths)):
            particle_trajectory[i] = []

    log_with_flush("initializing search... "+curret_time_string())
    initialize_search_records(search_pass_name, particle_paths, eval_type, dataset, gpus, base_model, fast_merge, starting_velocity_mode)
    log_with_flush("search initialized")
    for i in range(len(particle_paths)):
        log_with_flush("expert " + str(i) + ": " + particle_paths[i])

    if os.path.exists(os.path.join("search", search_pass_name, "tmp")):
        shutil.rmtree(os.path.join("search", search_pass_name, "tmp"))

    # test set evaluation
    if starting_test_set_eval:
        eval_test_args = []
        for i in range(len(particle_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        log_with_flush("Test set results:")
        for i in range(len(particle_paths)):
            log_with_flush("particle_"+str(i)+": "+str(results[i]))

    log_with_flush("starting search... "+curret_time_string())

    # main search iteration
    iter_count = 0
    while iter_count < max_iteration:
        iter_count += 1
        log_with_flush("--------------------------")
        log_with_flush("iteration "+str(iter_count)+"! "+curret_time_string())
        log_with_flush("updating particles...")

        # patience and ending condition
        with open(os.path.join("search", search_pass_name, "utility_scratchpad.json")) as f:
            utility_scratchpad = json.load(f)
        g_best = utility_scratchpad["g"]
        g_history = utility_scratchpad["g_history"]
        if len(g_history) > patience:
            g_history = g_history[-patience:]
            # if g_history hasn't changed
            if max(g_history) == min(g_history):
                log_with_flush("patience reached!")
                break

        if to_visualize_flag:
            for i in range(len(particle_paths)):
                lora_weight_path = os.path.join("search", search_pass_name, "particle_"+str(i), "now", "adapter_model.safetensors")
                coords = lora_weight_visualize(lora_weight_path)
                particle_trajectory[i].append(coords)
            with open(os.path.join("search", search_pass_name, "particle_trajectory.json"), "w") as f:
                json.dump(particle_trajectory, f, indent=4)

            with open(os.path.join("search", search_pass_name, "particle_trajectory.json"), "w") as f:
                json.dump(particle_trajectory, f, indent=4)

        if correctness_emergence:
            for i in range(len(particle_paths)):
                model_path = os.path.join("search", search_pass_name, "particle_"+str(i), "now")
                golds = json.load(open(os.path.join(model_path, "golds_dev.json"), "r"))
                preds = json.load(open(os.path.join(model_path, "preds_dev.json"), "r"))
                correctness = []
                assert len(golds) == len(preds)
                for j in range(len(golds)):
                    if golds[j] == preds[j]:
                        correctness.append(1)
                    else:
                        correctness.append(0)
                correctness_emergence_dict[i].append(correctness)

            with open(os.path.join("search", search_pass_name, "correctness_emergence.json"), "w") as f:
                json.dump(correctness_emergence_dict, f, indent=4)

        # update each particle
        update_args = []
        for i in range(len(particle_paths)):
            if restart_stray_particles:
                particle_history = utility_scratchpad["particle_"+str(i)+"_history"]
                particle_best_so_far = utility_scratchpad["particle_"+str(i)+"_best"]
                first_time_best_idx = particle_history.index(particle_best_so_far)
                if len(particle_history) - first_time_best_idx >= restart_patience * patience:
                    restart_flag = True
                    log_with_flush("particle_"+str(i)+" restarted!")
                else:
                    restart_flag = False
            else:
                restart_flag = False

            update_args.append((i, gpus[assign_gpu(len(gpus), i, len(particle_paths))], search_pass_name, weight_randomess, inertia, cognitive_coeff, social_coeff, repel_coeff, fast_merge, step_length, repel_term, restart_flag))

        pool = Pool(processes=num_cpu_when_merging)
        results = pool.starmap(particle_update, update_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()
        log_with_flush("all particles updated! "+curret_time_string())

        # evaluate each particle and update utility_scratchpad and weights
        log_with_flush("evaluating particles...")

        if random.random() < dropK: # iteration drop
            log_with_flush("dropped iteration!")
            global_skip_flag = True
        else:
            global_skip_flag = False

        eval_args = []
        for i in range(len(particle_paths)):

            if random.random() < dropN: # particle drop
                local_skip_flag = True
            else:
                local_skip_flag = False

            if not correctness_emergence:
                eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, False, None, global_skip_flag or local_skip_flag))
            else:
                eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "now"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, True, None, global_skip_flag or local_skip_flag))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
            utility_scratchpad = json.load(f)

        # if skipped, pull performance from last step
        for i in range(len(particle_paths)):
            if results[i] is None:
                results[i] = utility_scratchpad["particle_"+str(i)+"_now"]
                assert results[i] == utility_scratchpad["particle_"+str(i)+"_history"][-1]

        # personal bests update
        for i in range(len(particle_paths)):
            utility_scratchpad["particle_" + str(i) + "_now"] = results[i]
            utility_scratchpad["particle_" + str(i) + "_history"].append(results[i])
            if results[i] > utility_scratchpad["particle_" + str(i) + "_best"]:
                utility_scratchpad["particle_" + str(i) + "_best"] = results[i]
                shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), dirs_exist_ok=True)
                log_with_flush("new personal best for particle_"+str(i)+": "+str(results[i]))

        # global best update
        if max(results) > utility_scratchpad["g"]:
            utility_scratchpad["g"] = max(results)
            utility_scratchpad["g_history"].append(max(results))
            log_with_flush("new global best: "+str(utility_scratchpad["g"]))
            for i in range(len(particle_paths)):
                if results[i] == utility_scratchpad["g"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "global_best"), dirs_exist_ok=True)
                    break
        else:
            utility_scratchpad["g_history"].append(utility_scratchpad["g"])

        # global worst update
        if min(results) < utility_scratchpad["g_worst"]:
            utility_scratchpad["g_worst"] = min(results)
            for i in range(len(particle_paths)):
                if results[i] == utility_scratchpad["g_worst"]:
                    shutil.copytree(os.path.join("search", search_pass_name, "particle_"+str(i), "now"), os.path.join("search", search_pass_name, "global_worst"), dirs_exist_ok=True)

        wandb_log = {
            "g": utility_scratchpad["g"],
            "g_worst": utility_scratchpad["g_worst"],
        }
        for i in range(len(particle_paths)):
            wandb_log["particle_" + str(i) + "_now"] = utility_scratchpad["particle_" + str(i) + "_now"]

        wandb.log(wandb_log)

        with open("search/"+search_pass_name+"/utility_scratchpad.json", "w") as f:
            json.dump(utility_scratchpad, f, indent=4)

        log_with_flush("all particles evaluated! "+curret_time_string())
        log_with_flush("--------------------------")

        # step length update
        step_length = max(step_length * step_length_factor, minimum_step_length)

    log_with_flush("ending search and starting test set evaluation... "+curret_time_string())

    # which particle is global best?
    with open("search/"+search_pass_name+"/utility_scratchpad.json", "r") as f:
        utility_scratchpad = json.load(f)
    g_best = utility_scratchpad["g"]
    global_best_particle = -1
    for i in range(len(particle_paths)):
        if utility_scratchpad["particle_" + str(i) + "_best"] == g_best:
            global_best_particle = i
    log_with_flush("global best particle: "+str(global_best_particle))

    # dev set evaluation for personal bests
    eval_args = []
    for i in range(len(particle_paths)):
        eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model, True))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    # test set evaluation
    eval_test_args = []
    for i in range(len(particle_paths)):
        eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), eval_type, dataset, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

    pool = Pool(processes=len(gpus))
    results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
    pool.close()
    pool.join()

    log_with_flush("Test set results:")
    for i in range(len(particle_paths)):
        log_with_flush("particle_"+str(i)+": "+str(results[i]))

    final_metrics = overall_metrics(search_pass_name, eval_type)

    if eval_type == "AbstainQA":
        best_particle_idx = final_metrics["ending_best_particle_on_validation"]
        final_metrics["ending_best_single_test_accuracy"] = results[best_particle_idx]

    if eval_type == "perplexity" or eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_test_args = []
        for i in range(len(particle_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_1_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_1_name] = results[final_metrics["ending_best_particle_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_test_args = []
        for i in range(len(particle_paths)):
            eval_test_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_2_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate_test, eval_test_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_test_" + dataset_2_name] = results[final_metrics["ending_best_particle_on_validation"]]

    if eval_type == "multitask":
        dataset_1_name = perplexity_extrinsic_test_dict[dataset][0]
        eval_args = []
        for i in range(len(particle_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_1_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_1_name] = results[final_metrics["ending_best_particle_on_validation"]]

        dataset_2_name = perplexity_extrinsic_test_dict[dataset][1]
        eval_args = []
        for i in range(len(particle_paths)):
            eval_args.append((os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best"), "multiple_choice", dataset_2_name, gpus[assign_gpu(len(gpus), i, len(particle_paths))], base_model))

        pool = Pool(processes=len(gpus))
        results = pool.starmap(evaluate, eval_args, chunksize=math.ceil(len(particle_paths)/len(gpus)))
        pool.close()
        pool.join()

        final_metrics["ending_best_single_dev_" + dataset_2_name] = results[final_metrics["ending_best_particle_on_validation"]]

    wandb.log(final_metrics)
    log_with_flush("final metrics for test: "+str(final_metrics))

    # ensemble for dev set
    try:
        for i in range(len(particle_paths)):
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds.json"))

            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "golds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "preds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "golds.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "preds.json"))
    except:
        for i in range(len(particle_paths)):
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores.json"))
            os.remove(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores.json"))

            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "personal_best", "scores.json"))
            os.rename(os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores_dev.json"), os.path.join("search", search_pass_name, "particle_"+str(i), "now", "scores.json"))

    final_metrics = overall_metrics(search_pass_name, eval_type)
    dev_final_metrics = {
        "starting_top-k_ensemble_dev_accuracy": final_metrics["starting_top-k_ensemble_test_accuracy"],
        "ending_top-k_ensemble_dev_accuracy": final_metrics["ending_top-k_ensemble_test_accuracy"]
    }
    wandb.log(dev_final_metrics)
    log_with_flush("final ensemble metrics for dev: "+str(dev_final_metrics))

    if clean_up_on_end:
        shutil.rmtree(os.path.join("search", search_pass_name, "global_worst"))
        for i in range(len(particle_paths)):
            for aux in ["g_x", "p_x", "velocity", "x_w"]:
                shutil.rmtree(os.path.join("search", search_pass_name, "particle_"+str(i), aux))

    log_with_flush("the end of search... "+curret_time_string())