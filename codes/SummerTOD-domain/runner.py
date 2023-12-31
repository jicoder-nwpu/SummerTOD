"""
   MTTOD: runner.py

   implements train and predict function for MTTOD model.

   Copyright 2021 ETRI LIRS, Yohan Lee

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import os
import re
import copy
import math
import time
import glob
import shutil
from abc import *
import rouge
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_outputs import BaseModelOutput
from tensorboardX import SummaryWriter

from model import T5ForSummaryGeneration
from reader import MultiWOZIterator, MultiWOZReader
from evaluator import MultiWozEvaluator

from utils import definitions
from utils.io_utils import get_or_create_logger, load_json, save_json


evaluator = rouge.Rouge()

def evaluate_rouge(examples):
    true_sum_arr = []
    pred_sum_arr = []
    for dia in examples:
        for turn in examples[dia]:
            true_sum_arr.append(turn['sum'])
            pred_sum_arr.append(turn['sum_gen'])
    assert len(true_sum_arr) == len(pred_sum_arr)
    scores = evaluator.get_scores(pred_sum_arr, true_sum_arr, avg=True)
    return scores


class Reporter(object):
    def __init__(self, log_frequency, model_dir, logger):
        self.log_frequency = log_frequency
        self.logger = logger
        self.summary_writer = SummaryWriter(os.path.join(model_dir, "tensorboard"))

        self.global_step = 0
        self.lr = 0
        self.init_stats()

    def init_stats(self):
        self.step_time = 0.0

        self.summary_loss = 0.0
        self.resp_loss = 0.0

        self.summary_correct = 0.0
        self.resp_correct = 0.0

        self.summary_count = 0.0
        self.resp_count = 0.0

    def step(self, start_time, lr, step_outputs, force_info=False, is_train=True):
        self.global_step += 1
        self.step_time += (time.time() - start_time)

        self.summary_loss += step_outputs["summary"]["loss"]
        self.summary_correct += step_outputs["summary"]["correct"]
        self.summary_count += step_outputs["summary"]["count"]

        if "resp" in step_outputs:
            self.resp_loss += step_outputs["resp"]["loss"]
            self.resp_correct += step_outputs["resp"]["correct"]
            self.resp_count += step_outputs["resp"]["count"]

            do_resp_stats = True
        else:
            do_resp_stats = False

        if is_train:
            self.lr = lr
            self.summary_writer.add_scalar("lr", lr, global_step=self.global_step)

            if self.global_step % self.log_frequency == 0:
                self.info_stats("train", self.global_step, do_resp_stats)

    def info_stats(self, data_type, global_step, do_resp_stats=False):
        avg_step_time = self.step_time / self.log_frequency

        summary_ppl = math.exp(self.summary_loss / self.summary_count)
        summary_acc = (self.summary_correct / self.summary_count) * 100

        self.summary_writer.add_scalar(
            "{}/summary_loss".format(data_type), self.summary_loss, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/summary_ppl".format(data_type), summary_ppl, global_step=global_step)

        self.summary_writer.add_scalar(
            "{}/summary_acc".format(data_type), summary_acc, global_step=global_step)

        if data_type == "train":
            common_info = "step {0:d}; step-time {1:.2f}s; lr {2:.2e};".format(
                global_step, avg_step_time, self.lr)
        else:
            common_info = "[Validation]"

        summary_info = "[summary] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
            self.summary_loss, summary_ppl, summary_acc)

        if do_resp_stats:
            resp_ppl = math.exp(self.resp_loss / self.resp_count)
            resp_acc = (self.resp_correct / self.resp_count) * 100

            self.summary_writer.add_scalar(
                "{}/resp_loss".format(data_type), self.resp_loss, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_ppl".format(data_type), resp_ppl, global_step=global_step)

            self.summary_writer.add_scalar(
                "{}/resp_acc".format(data_type), resp_acc, global_step=global_step)

            resp_info = "[resp] loss {0:.2f}; ppl {1:.2f}; acc {2:.2f}".format(
                self.resp_loss, resp_ppl, resp_acc)
        else:
            resp_info = ""

        self.logger.info(
            " ".join([common_info, summary_info, resp_info]))

        self.init_stats()


class BaseRunner(metaclass=ABCMeta):
    def __init__(self, cfg, reader):
        self.cfg = cfg
        self.reader = reader

        self.model = self.load_model()

    def load_model(self):
        if self.cfg.ckpt is not None:
            model_path = self.cfg.ckpt
            initialize_additional = False
        elif self.cfg.train_from is not None:
            model_path = self.cfg.train_from
            initialize_additional = False
        else:
            model_path = self.cfg.backbone
            initialize_additional = True

        self.logger.info("Load model from {}".format(model_path))

        model_wrapper = T5ForSummaryGeneration

        model = model_wrapper.from_pretrained(model_path)

        model.resize_token_embeddings(self.reader.vocab_size)

        if initialize_additional:
            model.initialize_additional()
        '''
        if self.cfg.num_gpus > 1:
            model = torch.nn.DataParallel(model)
        '''
        model.to(self.cfg.device)

        return model

    def save_model(self, epoch):
        latest_ckpt = "ckpt-epoch{}".format(epoch)
        save_path = os.path.join(self.cfg.model_dir, latest_ckpt)
        '''
        if self.cfg.num_gpus > 1:
            model = self.model.module
        else:
            model = self.model
        '''
        model = self.model

        model.save_pretrained(save_path)

        # keep chekpoint up to maximum
        checkpoints = sorted(
            glob.glob(os.path.join(self.cfg.model_dir, "ckpt-*")),
            key=os.path.getmtime,
            reverse=True)

        checkpoints_to_be_deleted = checkpoints[self.cfg.max_to_keep_ckpt:]

        #force remove checkpoint-dirs
        for ckpt in checkpoints_to_be_deleted:
            shutil.rmtree(ckpt)

        return latest_ckpt

    def get_optimizer_and_scheduler(self, num_traininig_steps_per_epoch, train_batch_size):
        '''
        num_train_steps = (num_train_examples *
            self.cfg.epochs) // (train_batch_size * self.cfg.grad_accum_steps)
        '''
        num_train_steps = (num_traininig_steps_per_epoch *
            self.cfg.epochs) // self.cfg.grad_accum_steps

        if self.cfg.warmup_steps >= 0:
            num_warmup_steps = self.cfg.warmup_steps
        else:
            #num_warmup_steps = int(num_train_steps * 0.2)
            num_warmup_steps = int(num_train_steps * self.cfg.warmup_ratio)

        self.logger.info("Total training steps = {}, warmup steps = {}".format(
            num_train_steps, num_warmup_steps))

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        if self.cfg.no_learning_rate_decay:
            scheduler = get_constant_schedule(optimizer)
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps)

        return optimizer, scheduler

    #statistic pred tokens of labels
    def count_tokens(self, pred, label, pad_id):
        pred = pred.view(-1)
        label = label.view(-1)

        num_count = label.ne(pad_id).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    def count_spans(self, pred, label):
        pred = pred.view(-1, 2)

        num_count = label.ne(-1).long().sum()
        num_correct = torch.eq(pred, label).long().sum()

        return num_correct, num_count

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class MultiWOZRunner(BaseRunner):
    def __init__(self, cfg):
        self.logger = get_or_create_logger(__name__, cfg.model_dir)
        reader = MultiWOZReader(cfg.backbone, cfg.version, self.logger, cfg.summary_context_size, cfg.woz_type)

        self.iterator = MultiWOZIterator(reader, self.logger)

        super(MultiWOZRunner, self).__init__(cfg, reader)

    def step_fn(self, summary_encoder_input_ids, response_encoder_input_ids, updated_history_input_ids, last_summary_ids, summary_labels, resp_labels, summary_train):
        summary_encoder_input_ids = summary_encoder_input_ids.to(self.cfg.device)
        summary_encoder_attention_mask = torch.where(summary_encoder_input_ids == self.reader.pad_token_id, 0, 1)
        
        response_encoder_input_ids = response_encoder_input_ids.to(self.cfg.device)
        response_encoder_attention_mask = torch.where(response_encoder_input_ids == self.reader.pad_token_id, 0, 1)
        
        updated_history_input_ids = updated_history_input_ids.to(self.cfg.device)
        updated_history_attention_mask = torch.where(updated_history_input_ids == self.reader.pad_token_id, 0, 1)
        
        last_summary_ids = last_summary_ids.to(self.cfg.device)
        last_summary_attention_mask = torch.where(last_summary_ids == self.reader.pad_token_id, 0, 1)
        
        summary_history_encoder_outputs = self.model(input_ids=summary_encoder_input_ids,
                                            attention_mask=summary_encoder_attention_mask,
                                            encoder_type='history',
                                            encoder_only=True)

        response_history_encoder_outputs = self.model(input_ids=response_encoder_input_ids,
                                              attention_mask=response_encoder_attention_mask,
                                              encoder_type='history',
                                              encoder_only=True)
        
        updated_history_encoder_outputs = self.model(input_ids=updated_history_input_ids,
                                            attention_mask=updated_history_attention_mask,
                                            encoder_type='history',
                                            encoder_only=True)
        
        last_summary_encoder_outputs = self.model(input_ids=last_summary_ids,
                                            attention_mask=last_summary_attention_mask,
                                            encoder_outputs=updated_history_encoder_outputs,
                                            summary_attention_mask=updated_history_attention_mask,
                                            encoder_type='summary',
                                            encoder_only=True)
        
        # context_encoder_output = self.model(input_ids=context_inputs,
        #                                     attention_mask=context_attention_mask,
        #                                     encoder_type='history',
        #                                     encoder_only=True)
    
        # summary_encoder_output = self.model(input_ids=last_summary_ids,
        #                                     attention_mask=last_summary_attention_mask,
        #                                     encoder_type='summary',
        #                                     encoder_only=True)
        
        # updated_encoder_output = self.model(input_ids=updated_history_input_ids,
        #                                     attention_mask=updated_history_attention_mask,
        #                                     summary_encoder_outputs=summary_encoder_output,
        #                                     summary_attention_mask=last_summary_attention_mask,
        #                                     encoder_type='history',
        #                                     encoder_only=True)

        summary_labels = summary_labels.to(self.cfg.device)
        resp_labels = resp_labels.to(self.cfg.device)

        summary_outputs = self.model(encoder_outputs=summary_history_encoder_outputs,
                                     attention_mask=summary_encoder_attention_mask,
                                     summary_encoder_outputs=last_summary_encoder_outputs,
                                     summary_attention_mask=last_summary_attention_mask,
                                     lm_labels=summary_labels,
                                     return_dict=False,
                                     decoder_type='summary')
        
        # summary_outputs = self.model(encoder_outputs=context_encoder_output,
        #                              attention_mask=context_attention_mask,
        #                              summary_encoder_outputs=updated_encoder_output,
        #                              summary_attention_mask=updated_history_attention_mask,
        #                              lm_labels=summary_labels,
        #                              return_dict=False)

        summary_loss = summary_outputs[0]
        summary_pred = summary_outputs[1]

        summary_sequence_output = torch.argmax(summary_outputs[3], dim=-1)
        
        eos_id = 1
        
        batch_size, seq_len = summary_sequence_output.size()
        for e in range(batch_size):
            flag = False
            for s in range(seq_len):
                if flag:
                    summary_sequence_output[e][s] = self.reader.pad_token_id
                elif summary_sequence_output[e][s] == self.reader.eos_token_id:
                    flag = True
        
        cur_summary_input_ids = summary_labels.to(self.cfg.device)
        cur_summary_attention_mask = torch.where(cur_summary_input_ids == self.reader.pad_token_id, 0, 1)
        
        cur_summary_encoder_outputs = self.model(input_ids=cur_summary_input_ids,
                                      attention_mask=cur_summary_attention_mask,
                                      encoder_type='summary',
                                      encoder_only=True)
        
        response_outputs = self.model(encoder_outputs=response_history_encoder_outputs,
                                     attention_mask=response_encoder_attention_mask,
                                     summary_encoder_outputs=cur_summary_encoder_outputs,
                                     summary_attention_mask=cur_summary_attention_mask,
                                     lm_labels=resp_labels,
                                     return_dict=False,
                                     decoder_type='resp')
        
        resp_loss = response_outputs[0]
        resp_pred = response_outputs[1]
    
        num_resp_correct, num_resp_count = self.count_tokens(
            resp_pred, resp_labels, pad_id=self.reader.pad_token_id)

        num_summary_correct, num_summary_count = self.count_tokens(
            summary_pred, summary_labels, pad_id=0)
        
        # loss = 0
        # if summary_train:
        loss = summary_loss

        loss += (self.cfg.resp_loss_coeff * resp_loss)
        
        step_outputs = {}

        step_outputs["summary"] = {"loss": summary_loss.item(),
                                "correct": num_summary_correct.item(),
                                "count": num_summary_count.item()}

        step_outputs["resp"] = {"loss": resp_loss.item(),
                                "correct": num_resp_correct.item(),
                                "count": num_resp_count.item()}

        return loss, step_outputs, summary_sequence_output

    def train_epoch(self, train_iterator, optimizer, scheduler, reporter=None):
        self.model.train()
        self.model.zero_grad()

        last_summary_ids = None
        first_summary_ids = self.reader.encode_text("there is no summary .",
                                                bos_token=definitions.BOS_SUM_TOKEN,
                                                eos_token=definitions.EOS_SUM_TOKEN)
        for step, batch in enumerate(train_iterator):
            start_time = time.time()

            # context_inputs, updated_history_input_ids, last_summary_ids, labels = batch
            summary_encoder_input_ids, response_encoder_input_ids, updated_history_input_ids, last_summary_flag, labels = batch
            
            summary_train = False
            if step <= 705:
                summary_train = True
            
            flag = True
            for idx in range(len(first_summary_ids)):
                if last_summary_flag[0][idx] != first_summary_ids[idx]:
                    flag = False
                    break
            if flag:
                last_summary_ids = last_summary_flag

            loss, step_outputs, last_summary_ids = self.step_fn(summary_encoder_input_ids, response_encoder_input_ids, updated_history_input_ids, last_summary_flag, *labels, summary_train)

            if self.cfg.grad_accum_steps > 1:
                loss = loss / self.cfg.grad_accum_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.max_grad_norm)

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                lr = scheduler.get_last_lr()[0]

                if reporter is not None:
                    reporter.step(start_time, lr, step_outputs)

    def train(self):
        train_batches, num_training_steps_per_epoch, _, _ = self.iterator.get_batches(
            "train", self.cfg.batch_size, self.cfg.num_gpus, shuffle=True,
            num_dialogs=self.cfg.num_train_dialogs, excluded_domains=self.cfg.excluded_domains)

        optimizer, scheduler = self.get_optimizer_and_scheduler(
            num_training_steps_per_epoch, self.cfg.batch_size)

        reporter = Reporter(self.cfg.log_frequency, self.cfg.model_dir, self.logger)

        for epoch in range(1, self.cfg.epochs + 1):
            train_iterator = self.iterator.get_data_iterator(
                train_batches, self.cfg.task, self.cfg.ururu, self.cfg.context_size, self.cfg.summary_context_size)

            self.train_epoch(train_iterator, optimizer, scheduler, reporter)

            self.logger.info("done {}/{} epoch".format(epoch, self.cfg.epochs))

            self.save_model(epoch)

            if not self.cfg.no_validation:
                self.validation(reporter.global_step)

    def validation(self, global_step):
        self.model.eval()

        dev_batches, num_steps, _, _ = self.iterator.get_batches(
            "dev", self.cfg.batch_size, self.cfg.num_gpus)

        dev_iterator = self.iterator.get_data_iterator(
            dev_batches, self.cfg.task, self.cfg.ururu, self.cfg.context_size, self.cfg.summary_context_size)

        reporter = Reporter(1000000, self.cfg.model_dir, self.logger)

        torch.set_grad_enabled(False)
        last_summary_ids = None
        first_summary_ids = self.reader.encode_text("there is no summary .",
                                                bos_token=definitions.BOS_SUM_TOKEN,
                                                eos_token=definitions.EOS_SUM_TOKEN)
        for batch in tqdm(dev_iterator, total=num_steps, desc="Validaction"):
            start_time = time.time()

            # context_inputs, updated_history_input_ids, last_summary_ids, labels = batch
            summary_encoder_input_ids, response_encoder_input_ids, updated_history_input_ids, last_summary_flag, labels = batch
            
            flag = True
            for idx in range(len(first_summary_ids)):
                if last_summary_flag[0][idx] != first_summary_ids[idx]:
                    flag = False
                    break
            if flag:
                last_summary_ids = last_summary_flag

            _, step_outputs, last_summary_ids = self.step_fn(summary_encoder_input_ids, response_encoder_input_ids, updated_history_input_ids, last_summary_flag, *labels, summary_train=False)

            reporter.step(start_time, lr=None, step_outputs=step_outputs, is_train=False)

        do_resp_stats = True if "resp" in step_outputs else False

        reporter.info_stats("dev", global_step, do_resp_stats)

        torch.set_grad_enabled(True)

    def finalize_bspn(self, belief_outputs, domain_history, constraint_history, span_outputs=None, input_ids=None):
        eos_token_id = self.reader.get_token_id(definitions.EOS_BELIEF_TOKEN)

        batch_decoded = []
        for i, belief_output in enumerate(belief_outputs):
            if belief_output[0] == self.reader.pad_token_id:
                belief_output = belief_output[1:]

            if eos_token_id not in belief_output:
                eos_idx = len(belief_output) - 1
            else:
                eos_idx = belief_output.index(eos_token_id)

            bspn = belief_output[:eos_idx + 1]

            decoded = {}

            decoded["bspn_gen"] = bspn

            # update bspn using span output
            if span_outputs is not None and input_ids is not None:
                span_output = span_outputs[i]
                input_id = input_ids[i]

                #print(self.reader.tokenizer.decode(input_id))
                #print(self.reader.tokenizer.decode(bspn))

                eos_idx = input_id.index(self.reader.eos_token_id)
                input_id = input_id[:eos_idx]

                span_result = {}

                bos_user_id = self.reader.get_token_id(definitions.BOS_USER_TOKEN)

                span_output = span_output[:eos_idx]

                b_slot = None
                for t, span_token_idx in enumerate(span_output):
                    turn_id = max(input_id[:t].count(bos_user_id) - 1, 0)
                    turn_domain = domain_history[i][turn_id]

                    if turn_domain not in definitions.INFORMABLE_SLOTS:
                        continue

                    span_token = self.reader.span_tokens[span_token_idx]

                    if span_token not in definitions.INFORMABLE_SLOTS[turn_domain]:
                        b_slot = span_token
                        continue

                    if turn_domain not in span_result:
                        span_result[turn_domain] = defaultdict(list)

                    if b_slot != span_token:
                        span_result[turn_domain][span_token] = [input_id[t]]
                    else:
                        span_result[turn_domain][span_token].append(input_id[t])

                    b_slot = span_token

                for domain, sv_dict in span_result.items():
                    for s, v_list in sv_dict.items():
                        value = v_list[-1]
                        span_result[domain][s] = self.reader.tokenizer.decode(
                            value, clean_up_tokenization_spaces=False)

                span_dict = copy.deepcopy(span_result)

                ontology = self.reader.db.extractive_ontology

                flatten_span = []
                for domain, sv_dict in span_result.items():
                    flatten_span.append("[" + domain + "]")

                    for s, v in sv_dict.items():
                        if domain in ontology and s in ontology[domain]:
                            if v not in ontology[domain][s]:
                                del span_dict[domain][s]
                                continue

                        if s == "destination" or s == "departure":
                            _s = "destination" if s == "departure" else "departure"

                            if _s in sv_dict and v == sv_dict[_s]:
                                if s in span_dict[domain]:
                                    del span_dict[domain][s]
                                if _s in span_dict[domain]:
                                    del span_dict[domain][_s]
                                continue

                        if s in ["time", "leave", "arrive"]:
                            v = v.replace(".", ":")
                            if re.match("[0-9]+:[0-9]+", v) is None:
                                del span_dict[domain][s]
                                continue
                            else:
                                span_dict[domain][s] = v

                        flatten_span.append("[value_" + s + "]")
                        flatten_span.append(v)

                    if len(span_dict[domain]) == 0:
                        del span_dict[domain]
                        flatten_span.pop()

                #print(flatten_span)

                #input()

                decoded["span"] = flatten_span

                constraint_dict = self.reader.bspn_to_constraint_dict(
                    self.reader.tokenizer.decode(bspn, clean_up_tokenization_spaces=False))

                if self.cfg.overwrite_with_span:
                    _constraint_dict = OrderedDict()

                    for domain, slots in definitions.INFORMABLE_SLOTS.items():
                        if domain in constraint_dict or domain in span_dict:
                            _constraint_dict[domain] = OrderedDict()

                        for slot in slots:
                            if domain in constraint_dict:
                                cons_value = constraint_dict[domain].get(slot, None)
                            else:
                                cons_value = None

                            if domain in span_dict:
                                span_value = span_dict[domain].get(slot, None)
                            else:
                                span_value = None

                            if cons_value is None and span_value is None:
                                continue

                            # priority: span_value > cons_value
                            slot_value = span_value or cons_value

                            _constraint_dict[domain][slot] = slot_value
                else:
                    _constraint_dict = copy.deepcopy(constraint_dict)

                bspn_gen_with_span = self.reader.constraint_dict_to_bspn(
                    _constraint_dict)

                bspn_gen_with_span = self.reader.encode_text(
                    bspn_gen_with_span,
                    bos_token=definitions.BOS_BELIEF_TOKEN,
                    eos_token=definitions.EOS_BELIEF_TOKEN)

                decoded["bspn_gen_with_span"] = bspn_gen_with_span

            batch_decoded.append(decoded)

        return batch_decoded

    def finalize_resp(self, resp_outputs):
        bos_action_token_id = self.reader.get_token_id(definitions.BOS_ACTION_TOKEN)
        eos_action_token_id = self.reader.get_token_id(definitions.EOS_ACTION_TOKEN)

        bos_resp_token_id = self.reader.get_token_id(definitions.BOS_RESP_TOKEN)
        eos_resp_token_id = self.reader.get_token_id(definitions.EOS_RESP_TOKEN)

        batch_decoded = []
        for resp_output in resp_outputs:
            resp_output = resp_output[1:]
            if self.reader.eos_token_id in resp_output:
                eos_idx = resp_output.index(self.reader.eos_token_id)
                resp_output = resp_output[:eos_idx]

            try:
                bos_action_idx = resp_output.index(bos_action_token_id)
                eos_action_idx = resp_output.index(eos_action_token_id)
            except ValueError:
                self.logger.warn("bos/eos action token not in : {}".format(
                    self.reader.tokenizer.decode(resp_output)))
                aspn = [bos_action_token_id, eos_action_token_id]
            else:
                aspn = resp_output[bos_action_idx:eos_action_idx + 1]

            try:
                bos_resp_idx = resp_output.index(bos_resp_token_id)
                eos_resp_idx = resp_output.index(eos_resp_token_id)
            except ValueError:
                self.logger.warn("bos/eos resp token not in : {}".format(
                    self.reader.tokenizer.decode(resp_output)))
                resp = [bos_resp_token_id, eos_resp_token_id]
            else:
                resp = resp_output[bos_resp_idx:eos_resp_idx + 1]

            decoded = {"aspn_gen": aspn, "resp_gen": resp}

            batch_decoded.append(decoded)

        return batch_decoded

    def predict(self):
        print(self.cfg)
        self.model.eval()

        pred_batches, _, _, _ = self.iterator.get_batches(
            self.cfg.pred_data_type, self.cfg.batch_size,
            self.cfg.num_gpus, excluded_domains=self.cfg.excluded_domains)

        early_stopping = True if self.cfg.beam_size > 1 else False

        eval_dial_list = None
        if self.cfg.excluded_domains is not None:
            eval_dial_list = []

            for domains, dial_ids in self.iterator.dial_by_domain.items():
                domain_list = domains.split("-")

                if len(set(domain_list) & set(self.cfg.excluded_domains)) == 0:
                    eval_dial_list.extend(dial_ids)

        results = {}
        for dial_batch in tqdm(pred_batches, total=len(pred_batches), desc="Prediction"):
            batch_size = len(dial_batch)

            dial_history = [[] for _ in range(batch_size)]
            domain_history = [[] for _ in range(batch_size)]
            constraint_dicts = [OrderedDict() for _ in range(batch_size)]
            batch_last_summarys = []
            for turn_batch in self.iterator.transpose_batch(dial_batch):
                batch_summary_history_encoder_input_ids = []
                batch_response_history_encoder_input_ids = []
                batch_updated_history_ids = []
                batch_last_summary_ids = []
                for t, turn in enumerate(turn_batch):
                    summary_context, last_response = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn["user"]), self.cfg.summary_context_size)
                    
                    summary_history_encoder_input_ids = summary_context + last_response + turn['user'] + [self.reader.eos_token_id]
                    
                    reponse_context, last_response = self.iterator.flatten_dial_history(
                        dial_history[t], len(turn["user"]), self.cfg.context_size)

                    response_history_encoder_input_ids = reponse_context + last_response + turn['user'] + [self.reader.eos_token_id]
                    
                    if len(batch_last_summarys) == 0:
                        last_summary_ids = self.reader.encode_text("there is no summary .",
                                                                   bos_token=definitions.BOS_SUM_TOKEN,
                                                                   eos_token=definitions.EOS_SUM_TOKEN)
                    else:
                        last_summary_ids = batch_last_summarys[t]
                    
                    updated_history_ids = last_response + turn['user'] + [self.reader.eos_token_id]

                    batch_summary_history_encoder_input_ids.append(self.iterator.tensorize(summary_history_encoder_input_ids))
                    batch_response_history_encoder_input_ids.append(self.iterator.tensorize(response_history_encoder_input_ids))
                    batch_updated_history_ids.append(self.iterator.tensorize(updated_history_ids))
                    batch_last_summary_ids.append(self.iterator.tensorize(last_summary_ids))

                    turn_domain = turn["turn_domain"][-1]

                    if "[" in turn_domain:
                        turn_domain = turn_domain[1:-1]

                    domain_history[t].append(turn_domain)
                
                batch_last_summarys = []

                batch_summary_history_encoder_input_ids = pad_sequence(batch_summary_history_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                
                batch_response_history_encoder_input_ids = pad_sequence(batch_response_history_encoder_input_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                
                batch_updated_history_ids = pad_sequence(batch_updated_history_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                
                batch_last_summary_ids = pad_sequence(batch_last_summary_ids,
                                                       batch_first=True,
                                                       padding_value=self.reader.pad_token_id)
                
                batch_summary_history_encoder_input_ids = batch_summary_history_encoder_input_ids.to(self.cfg.device)

                batch_response_history_encoder_input_ids = batch_response_history_encoder_input_ids.to(self.cfg.device)
                
                batch_updated_history_ids = batch_updated_history_ids.to(self.cfg.device)
                
                batch_last_summary_ids = batch_last_summary_ids.to(self.cfg.device)

                summary_history_attention_mask = torch.where(
                    batch_summary_history_encoder_input_ids == self.reader.pad_token_id, 0, 1)
                
                response_history_attention_mask = torch.where(
                    batch_response_history_encoder_input_ids == self.reader.pad_token_id, 0, 1)
                
                updated_history_attention_mask = torch.where(
                    batch_updated_history_ids == self.reader.pad_token_id, 0, 1)
                
                batch_last_summary_attention_mask = torch.where(
                    batch_last_summary_ids == self.reader.pad_token_id, 0, 1)
                
                # belief tracking
                with torch.no_grad():
                    # encoder_outputs = self.model(input_ids=batch_encoder_input_ids,
                    #                              attention_mask=attention_mask,
                    #                              return_dict=False,
                    #                              encoder_only=True)

                    # encoder_hidden_states = encoder_outputs

                    # if isinstance(encoder_hidden_states, tuple):
                    #     last_hidden_state = encoder_hidden_states[0]
                    # else:
                    #     last_hidden_state = encoder_hidden_states

                    # # wrap up encoder outputs
                    # encoder_outputs = BaseModelOutput(
                    #     last_hidden_state=last_hidden_state)
                    
                    summary_history_encoder_outputs = self.model(input_ids=batch_summary_history_encoder_input_ids,
                                                        attention_mask=summary_history_attention_mask,
                                                        encoder_type='history',
                                                        encoder_only=True)
                    
                    response_history_encoder_outputs = self.model(input_ids=batch_response_history_encoder_input_ids,
                                                        attention_mask=response_history_attention_mask,
                                                        encoder_type='history',
                                                        encoder_only=True)
                    
                    updated_history_encoder_output = self.model(input_ids=batch_updated_history_ids,
                                                        attention_mask=updated_history_attention_mask,
                                                        encoder_type='history',
                                                        encoder_only=True)
                    
                    summary_encoder_outputs = self.model(input_ids=batch_last_summary_ids,
                                                        attention_mask=batch_last_summary_attention_mask,
                                                        encoder_outputs=updated_history_encoder_output,
                                                        summary_attention_mask=updated_history_attention_mask,
                                                        encoder_type='summary',
                                                        encoder_only=True)
                    
                    summary_prefixs = []

                    for turn in turn_batch:
                        summary_prefixs.append([self.reader.pad_token_id] + [self.reader.tokenizer.encode("<bos_sum>")[0]])
                        
                    summary_decoder_input_ids = self.iterator.tensorize(summary_prefixs)

                    summary_decoder_input_ids = summary_decoder_input_ids.to(self.cfg.device)

                    
                    summary_outputs = self.model.generate(encoder_outputs=summary_history_encoder_outputs,
                                                          attention_mask=summary_history_attention_mask,
                                                          summary_encoder_outputs=summary_encoder_outputs,
                                                          summary_attention_mask=batch_last_summary_attention_mask,
                                                          eos_token_id=self.reader.eos_token_id,
                                                          decoder_input_ids=summary_decoder_input_ids,
                                                          max_length=200,
                                                          do_sample=self.cfg.do_sample,
                                                          num_beams=self.cfg.beam_size,
                                                          early_stopping=early_stopping,
                                                          temperature=self.cfg.temperature,
                                                          top_k=self.cfg.top_k,
                                                          top_p=self.cfg.top_p,
                                                          output_hidden_states = True,
                                                          return_dict_in_generate=True,
                                                          decoder_type='summary')
                
                    summarys = summary_outputs.sequences.cpu().numpy().tolist()
                    
                    batch_decoded = []
                    for output in summarys:
                        output = output[1:]
                        batch_last_summarys.append(output)
                        if self.reader.eos_token_id in output:
                            eos_idx = output.index(self.reader.eos_token_id)
                            output = output[:eos_idx]
                        decoded = {'sum_gen': output}
                        batch_decoded.append(decoded)
                        
                    for t, turn in enumerate(turn_batch):
                        turn.update(**batch_decoded[t])

                    decoder_hidden_states = summary_outputs.decoder_hidden_states
                    seqs = summary_outputs.sequences.cpu().numpy().tolist()
                    seq_len = len(decoder_hidden_states)
                    dtype = decoder_hidden_states[0].dtype

                    summary_attention_mask = []
                    for b_i in range(batch_size):
                        summary_attention_mask.append([])
                        for s_i in range(seq_len):
                            if seqs[b_i][s_i + 1] == self.model.config.pad_token_id:
                                summary_attention_mask[-1].append(0)
                            else:
                                summary_attention_mask[-1].append(1)
                    summary_attention_mask = torch.tensor(summary_attention_mask, dtype=dtype)
                    summary_attention_mask = summary_attention_mask.to(self.cfg.device)

                    '''
                    summary embedding encoder
                    '''
                    summary_ids = []
                    for b_i in range(batch_size):
                        summary_ids.append([])
                        for s_i in range(seq_len):
                            summary_ids[-1].append(seqs[b_i][s_i + 1])
                    summary_ids = torch.tensor(summary_ids, dtype=torch.long)
                    summary_ids = summary_ids.to(self.cfg.device)

                    cur_summary_encoder_outputs = self.model(input_ids=summary_ids,
                                                            attention_mask=summary_attention_mask,
                                                            encoder_type='summary',
                                                            encoder_only=True)

                dbpn = []

                for turn in turn_batch:
                    dbpn.append(turn["dbpn"])

                for t, db in enumerate(dbpn):
                    if self.cfg.use_true_curr_aspn:
                        db += turn_batch[t]["aspn"]

                    # T5 use pad_token as start_decoder_token_id
                    # dbpn[t] = [self.reader.pad_token_id] + self.reader.tokenizer.encode(" this is a response </s> ") + db
                    dbpn[t] = [self.reader.pad_token_id] + db

                resp_decoder_input_ids = self.iterator.tensorize(dbpn)

                resp_decoder_input_ids = resp_decoder_input_ids.to(self.cfg.device)
                
                # response generation
                with torch.no_grad():
                    resp_outputs = self.model.generate(
                        encoder_outputs=response_history_encoder_outputs,
                        attention_mask=response_history_attention_mask,
                        summary_encoder_outputs=cur_summary_encoder_outputs,
                        summary_attention_mask=summary_attention_mask,
                        decoder_input_ids=resp_decoder_input_ids,
                        eos_token_id=self.reader.eos_token_id,
                        max_length=300,
                        do_sample=self.cfg.do_sample,
                        num_beams=self.cfg.beam_size,
                        early_stopping=early_stopping,
                        temperature=self.cfg.temperature,
                        # output_attentions=True,
                        return_dict_in_generate=True,
                        top_k=self.cfg.top_k,
                        top_p=self.cfg.top_p,
                        decoder_type='resp')
                    
                # cross_attentions = resp_outputs[2]

                resp_outputs = resp_outputs[0].cpu().numpy().tolist()

                decoded_resp_outputs = self.finalize_resp(resp_outputs)

                for t, turn in enumerate(turn_batch):
                    turn.update(**decoded_resp_outputs[t])

                
                #attention 可视化
                # cross_attens = torch.zeros([len(cross_attentions), summary_hidden_states.shape[1]], dtype=float)
                # iii = 0
                # for word in cross_attentions:
                #     ttt = word[-1].mean(-2).mean(-2).mean(-2)
                #     cross_attens[iii] = ttt
                #     iii += 1
                
                # cross_attens = torch.nn.functional.softmax(cross_attens, -1)

                # update dial_history
                for t, turn in enumerate(turn_batch):
                    pv_text = copy.copy(turn["user"])
                    pv_bspn = turn["bspn"]

                    pv_dbpn = turn["dbpn"]

                    if self.cfg.use_true_prev_aspn:
                        pv_aspn = turn["aspn"]
                    else:
                        pv_aspn = turn["aspn_gen"]

                    if self.cfg.use_true_prev_resp:
                        pv_resp = turn["redx"]
                        # pv_resp = turn["resp"]
                    else:
                        pv_resp = turn["resp_gen"]

                    if self.cfg.ururu:
                        pv_text += pv_resp
                    else:
                        pv_text += (pv_bspn + pv_dbpn + pv_aspn + pv_resp)

                    dial_history[t].append(pv_text)

            result = self.iterator.get_readable_batch(dial_batch)
            results.update(**result)

        if self.cfg.output:
            save_json(results, os.path.join(self.cfg.ckpt, self.cfg.output))

        evaluator = MultiWozEvaluator(self.reader, self.logger, self.cfg.pred_data_type)

        rouge_scores = evaluate_rouge(results)
        self.logger.info(rouge_scores)

        if self.cfg.task == "e2e":
            bleu, success, match = evaluator.e2e_eval(
                results, eval_dial_list=eval_dial_list)

            score = 0.5 * (success + match) + bleu

            self.logger.info('match: %2.2f; success: %2.2f; bleu: %2.2f; score: %.2f' % (
                match, success, bleu, score))
        else:
            joint_goal, f1, accuracy, count_dict, correct_dict = evaluator.dialog_state_tracking_eval(
                results)

            self.logger.info('joint acc: %2.2f; acc: %2.2f; f1: %2.2f;' % (
                joint_goal, accuracy, f1))

            for domain_slot, count in count_dict.items():
                correct = correct_dict.get(domain_slot, 0)

                acc = (correct / count) * 100

                self.logger.info('{0} acc: {1:.2f}'.format(domain_slot, acc))
