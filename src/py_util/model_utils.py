import torch, time, math, numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import json

class TorchModelHandler:
    '''
    Class that holds a model and provides the functionality to train it,
    save it, load it, and evaluate it. The model used here is assumed to be
    written in pytorch.
    '''
    def __init__(self, num_ckps=1, checkpoint_path="./data/checkpoints/", use_cuda=False, **params):

        self.model = params["model"]
        self.text_input_model = params["text_input_model"]
        self.topic_input_model = params.get("topic_input_model")
        self.is_joint_text_topic = bool(int(params.get("is_joint_text_topic", "0")))

        self.dataloader = params["dataloader"]
        self.num_labels = self.model.num_labels
        # self.output_dim = 1 if self.num_labels < 1 else self.num_labels
        self.output_dim = self.model.output_dim
        self.is_ensemble = self.model.is_ensemble if hasattr(self.model, "is_ensemble") else False

        self.name = params["name"]
        
        self.loss_function = params["loss_function"]
        self.optimizer = params["optimizer"]
        self.has_scheduler = "scheduler" in params
        if self.has_scheduler:
            self.scheduler = params["scheduler"]

        self.checkpoint_path = checkpoint_path
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_num = 0
        self.num_ckps = num_ckps
        # self.score_dict = dict()
        self.loss = 0
        self.epoch = 0

        self.use_cuda = use_cuda

    def save_best(self):
        '''
        Evaluates the model on data and then updates the best scores and saves the best model.
        '''
        self.save(num='BEST')

    def save(self, num=None):
        '''
        Saves the pytorch model in a checkpoint file.
        :param num: The number to associate with the checkpoint. By default uses
                    the internally tracked checkpoint number but this can be changed.
        '''
        if num is None:
            check_num = self.checkpoint_num
        else: check_num = num

        torch.save(
            obj = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss
            },
            f = f'{self.checkpoint_path}ckp-{self.name}-{check_num}.tar'
        )

        if num is None:
            self.checkpoint_num = (self.checkpoint_num + 1) % self.num_ckps

    def load(self, filename='data/checkpoints/ckp-[NAME]-FINAL.tar', use_cpu=False):
        '''
        Loads a saved pytorch model from a checkpoint file.
        :param filename: the name of the file to load from. By default uses
                        the final checkpoint for the model of this' name.
        '''
        filename = filename.replace('[NAME]', self.name)
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def train_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        self.model.train()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()
        
        self.loss = 0
        # partial_loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            #zero gradients before every optimizer step
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze(dim=-1)
            
            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")

            self.model.zero_grad()
            
            # get the embeddings for text and topic and creates a dict to pass as params to the model
            text_embeddings = self.text_input_model(**batch_data["text"])
            if self.is_joint_text_topic:
                model_inputs = {
                    "input":text_embeddings,
                    "input_length": batch_data["text"]["input_length"],
                }
            else:
                model_inputs = {
                    "text_embeddings": text_embeddings,
                    "text_length": batch_data["text"]["input_length"],
                }
                if self.topic_input_model:
                    topic_embeddings = self.topic_input_model(**batch_data["topic"])
                    model_inputs.update({
                        "topic_embeddings": topic_embeddings,
                        "topic_length": batch_data["topic"]["input_length"],
                    })
            
            if self.is_ensemble:
                clf_pred_label_1_tensor = torch.stack(batch_data["pred_label_1"]).T.type(torch.FloatTensor)
                if len(clf_pred_label_1_tensor.shape) > 1 and clf_pred_label_1_tensor.shape[-1] != 1:
                    clf_pred_label_1_tensor = clf_pred_label_1_tensor.argmax(dim=1).reshape(-1,1)
                clf_pred_label_1_tensor = clf_pred_label_1_tensor.squeeze(dim=-1)

                clf_pred_label_2_tensor = torch.stack(batch_data["pred_label_2"]).T.type(torch.FloatTensor)
                if len(clf_pred_label_2_tensor.shape) > 1 and clf_pred_label_2_tensor.shape[-1] != 1:
                    clf_pred_label_2_tensor = clf_pred_label_2_tensor.argmax(dim=1).reshape(-1,1)
                clf_pred_label_2_tensor = clf_pred_label_2_tensor.squeeze(dim=-1)

                model_inputs.update({
                    "clf1_pred": clf_pred_label_1_tensor,
                    "clf2_pred": clf_pred_label_2_tensor,
                })
            
            #apply the text and topic embeddings (and clf predictions if applicable) to the model
            y_pred = self.model(**model_inputs)
            if y_pred.squeeze(dim=-1).shape != torch.Size([]):
                y_pred = y_pred.squeeze(dim=-1)

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.loss_function(y_pred, label_tensor)
            if "sample_weight" in batch_data:
                weight_lst = batch_data["sample_weight"]
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            self.optimizer.step()

            #sum the loss
            # partial_loss += graph_loss.item()
            self.loss += graph_loss.item()

            #show the loss per batch once for each 1000 batches
            # if batch_n % 1000 == 999:
            #     last_loss = partial_loss / 1000 #loss per batch
            #     print(f"    batch {batch_n+1} loss: {last_loss}")
            #     partial_loss = 0

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def compute_scores(self, score_fn, true_labels, pred_labels, name, score_dict):
        '''
        Computes scores using the given scoring function of the given name. The scores
        are stored in the internal score dictionary.
        :param score_fn: the scoring function to use.
        :param true_labels: the true labels.
        :param pred_labels: the predicted labels.
        :param name: the name of this score function, to be used in storing the scores.
        :param score_dict: the dictionary used to store the scores.
        '''
        if hasattr(self, "is_llm") and self.is_llm:
            vals = score_fn(
                true_labels,
                np.floor(pred_labels*self.num_labels),
                average=None,
                labels=range(self.num_labels)
            )
        elif self.output_dim == 1:
            vals = score_fn(true_labels, (pred_labels>0.5)*1, average=None, labels=range(self.num_labels))
        else:
            if len(true_labels.shape) > 1 and true_labels.shape[-1] != 1:
                true_labels_ = np.argmax(true_labels, axis=1)
            else:
                true_labels_ = true_labels.squeeze()
            
            if len(pred_labels.shape) > 1 and pred_labels.shape[-1] != 1:
                pred_labels_ = np.argmax(pred_labels, axis=1)
            else:
                pred_labels_ = pred_labels.squeeze()
            
            vals = score_fn(
                true_labels_,
                pred_labels_,
                average=None,
                labels=range(self.num_labels)
            )
        if name not in score_dict:
            score_dict[name] = {}
        
        score_dict[name]['macro'] = sum(vals) / self.num_labels

        for i in range(self.num_labels):
            score_dict[name][i] = vals[i]
        
        return score_dict

    def eval_model(self, data=None, y_pred=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        '''

        if y_pred is not None:
            y_pred = np.reshape(a=y_pred, newshape=(-1,1))
            labels = self.get_labels(data=data)
            loss = 0
        else:
            y_pred, labels, loss = self.predict(data)

        score_dict = self.score(y_pred, labels)

        return score_dict, loss, y_pred

    def get_labels(self, data=None):
        if data is None:
            data = self.dataloader
        
        all_labels = torch.tensor([], device="cpu")
        for batch_n, batch_data in tqdm(enumerate(data)):
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            
            all_labels = torch.cat((all_labels, label_tensor))
        return all_labels.numpy()

    def predict(self, data=None):
        self.model.eval()
        self.text_input_model.eval()
        if not self.is_joint_text_topic and self.topic_input_model:
            self.topic_input_model.eval()
        
        partial_loss = 0
        all_y_pred = torch.tensor([], device="cpu")
        all_labels = torch.tensor([], device="cpu")
        
        if data is None:
            data = self.dataloader
        
        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze(dim=-1)

                all_labels = torch.cat((all_labels, label_tensor))
                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")

                # get the embeddings for text and topic and creates a dict to pass as params to the model
                text_embeddings = self.text_input_model(**batch_data["text"])
                if self.is_joint_text_topic:
                    model_inputs = {
                        "input":text_embeddings,
                        "input_length": batch_data["text"]["input_length"],
                    }
                else:
                    model_inputs = {
                        "text_embeddings": text_embeddings,
                        "text_length": batch_data["text"]["input_length"],
                    }
                    if self.topic_input_model:
                        topic_embeddings = self.topic_input_model(**batch_data["topic"])
                        model_inputs.update({
                            "topic_embeddings": topic_embeddings,
                            "topic_length": batch_data["topic"]["input_length"]
                        })
                
                if self.is_ensemble:
                    clf_pred_label_1_tensor = torch.stack(batch_data["pred_label_1"]).T.type(torch.FloatTensor)
                    if len(clf_pred_label_1_tensor.shape) > 1 and clf_pred_label_1_tensor.shape[-1] != 1:
                        clf_pred_label_1_tensor = clf_pred_label_1_tensor.argmax(dim=1).reshape(-1,1)
                    clf_pred_label_1_tensor = clf_pred_label_1_tensor.squeeze(dim=-1)

                    clf_pred_label_2_tensor = torch.stack(batch_data["pred_label_2"]).T.type(torch.FloatTensor)
                    if len(clf_pred_label_2_tensor.shape) > 1 and clf_pred_label_2_tensor.shape[-1] != 1:
                        clf_pred_label_2_tensor = clf_pred_label_2_tensor.argmax(dim=1).reshape(-1,1)
                    clf_pred_label_2_tensor = clf_pred_label_2_tensor.squeeze(dim=-1)

                    model_inputs.update({
                        "clf1_pred": clf_pred_label_1_tensor,
                        "clf2_pred": clf_pred_label_2_tensor,
                    })
                
                y_pred = self.model(**model_inputs)
                if y_pred.squeeze(dim=-1).shape != torch.Size([]):
                    y_pred = y_pred.squeeze(dim=-1)

                # if self.use_cuda:
                all_y_pred = torch.cat((all_y_pred, y_pred.cpu()))
                
                graph_loss = self.loss_function(y_pred, label_tensor)
                if "sample_weight" in batch_data:
                    weight_lst = batch_data["sample_weight"]
                    if self.use_cuda:
                        weight_lst = weight_lst.to('cuda')
                    
                    graph_loss = torch.mean(graph_loss * weight_lst)
            
                partial_loss += graph_loss.item()

        avg_loss = partial_loss / batch_n #loss per batch
        return all_y_pred.numpy(), all_labels.numpy(), avg_loss#partial_loss
    
    def eval_and_print(self, data=None, data_name=None, y_pred=None):
        '''
        Evaluates this model on the given data. Stores computed
        scores in the field "score_dict". Currently computes macro-averaged.
        Prints the results to the console.
        F1 scores, precision and recall. Can also compute scores on a class-wise basis.
        :param data: the data to use for evaluation. By default uses the internally stored data
                    (should be a DataSampler if passed as a parameter).
        :param data_name: the name of the data evaluating.
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        :return: a map from score names to values
        '''
        # Passing data_name to eval_model as evaluation of adv model on train and dev are different
        scores, loss, y_pred = self.eval_model(data=data, y_pred=y_pred)
        print("Evaluating on \"{}\" data".format(data_name))
        for metric_name, metric_dict in scores.items():
            for class_name, metric_val in metric_dict.items():
                print(f"{metric_name}_{class_name}: {metric_val:.4f}", end="\t")
            print()

        return scores, loss, y_pred

    def score(self, pred_labels, true_labels):
        '''
        Helper Function to compute scores. Stores updated scores in
        the field "score_dict".
        :param pred_labels: the predicted labels
        :param true_labels: the correct labels
        :param class_wise: flag to determine whether to compute class-wise scores in
                            addition to macro-averaged scores.
        '''
        score_dict = dict()
        # calculate class-wise and macro-averaged F1
        score_dict = self.compute_scores(f1_score, true_labels, pred_labels, 'f', score_dict)
        # calculate class-wise and macro-average precision
        score_dict = self.compute_scores(precision_score, true_labels, pred_labels, 'p', score_dict)
        # calculate class-wise and macro-average recall
        score_dict = self.compute_scores(recall_score, true_labels, pred_labels, 'r', score_dict)

        return score_dict

class AADTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )

        self.tgt_dataloader = params["tgt_dataloader"]
        self.src_encoder = self.text_input_model
        self.tgt_encoder = params["tgt_text_input_model"]

        self.discriminator_loss_fn = params["discriminator_loss_fn"]
        self.discriminator_clip_value = params["discriminator_clip_value"]
        self.discriminator_optimizer = params["discriminator_optimizer"]
        
        self.tgt_encoder_optimizer = params["tgt_encoder_optimizer"]
        self.tgt_encoder_temperature = params["tgt_encoder_temperature"]
        self.tgt_encoder_loss_fn = params["tgt_encoder_loss_fn"]
        self.tgt_loss_alpha = params["tgt_loss_alpha"]
        self.tgt_loss_beta = params["tgt_loss_beta"]
        self.max_grad_norm = params["max_grad_norm"]

        self.KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean')

    def pretrain_step(self):
        '''
        Runs one epoch of training on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        self.model.train()
        self.src_encoder.train()
       
        self.loss = 0
        # partial_loss = 0
        start_time = time.time()

        for batch_n, batch_data in tqdm(enumerate(self.dataloader)):
            #zero gradients before every optimizer step
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze()
            
            if self.use_cuda:
                label_tensor = label_tensor.to("cuda")

            self.model.zero_grad()
            
            # get the embeddings for text and topic and creates a dict to pass as params to the model
            text_embeddings = self.src_encoder(**batch_data["text"])[:,-1,:]

            #apply the text and topic embeddings to the model
            y_pred = self.model(text_embeddings=text_embeddings)

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.loss_function(y_pred.squeeze(), label_tensor)
            if "sample_weight" in batch_data:
                weight_lst = batch_data["sample_weight"]
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            self.optimizer.step()

            #sum the loss
            # partial_loss += graph_loss.item()
            self.loss += graph_loss.item()

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def adapt_step(self):
        '''
        Runs one epoch of adapt process on this model.
        '''
        print(f"[{self.name}] epoch {self.epoch}")
        # set train state to the right models
        self.src_encoder.eval()
        self.model.classifier.eval()
        self.tgt_encoder.train()
        self.model.discriminator.train()
       
        self.discriminator_loss = 0
        # partial_loss = 0
        start_time = time.time()

        data_iter = tqdm(enumerate(zip(self.dataloader, self.tgt_dataloader)))
        for batch_n, (src_batch_data, tgt_batch_data) in data_iter:
            # make sure only equal size batches are used from both source and target domain
            if len(src_batch_data["label"][0]) != len(tgt_batch_data["label"][0]):
                continue

            # def get_label_tensor(x, use_cuda):
            #     x_tensor = torch.stack(x).T
            #     if len(x_tensor.shape) > 1 and x_tensor.shape[-1] != 1:
            #         x_tensor = x_tensor.argmax(dim=1).reshape(-1,1)
                
            #     if use_cuda:
            #         x_tensor = x_tensor.to("cuda")
            #     return x_tensor

            # stance_lbl_src = get_label_tensor(src_batch_data["label"], self.use_cuda)
            # stance_lbl_tgt = get_label_tensor(tgt_batch_data["label"], self.use_cuda)

            #zero gradients before every optimizer step
            self.model.zero_grad()
            
            # get the embeddings for src and tgt text using the tgt_encoder and concat them to pass to the discriminator
            with torch.no_grad():
                src_text_embeddings = self.src_encoder(**src_batch_data["text"])[:,-1,:]
            src_tgt_text_embeddings = self.tgt_encoder(**src_batch_data["text"])[:,-1,:]
            tgt_text_embeddings = self.tgt_encoder(**tgt_batch_data["text"])[:,-1,:]
            embeddings_concat = torch.cat((src_tgt_text_embeddings, tgt_text_embeddings), 0)

            # prepare real and fake label to calculate the discriminator loss
            domain_label_src = torch.ones(src_tgt_text_embeddings.size(0)).unsqueeze(1)
            domain_label_tgt = torch.zeros(tgt_text_embeddings.size(0)).unsqueeze(1)

            if self.use_cuda:
                domain_label_src = domain_label_src.to("cuda")
                domain_label_tgt = domain_label_tgt.to("cuda")
            
            domain_label_concat = torch.cat((domain_label_src, domain_label_tgt), 0)

            # predict on discriminator
            domain_pred_concat = self.model.discriminator(embeddings_concat.detach())

            # calculate the loss, and backprogate it to update weights
            graph_loss = self.discriminator_loss_fn(domain_pred_concat, domain_label_concat)
            if "sample_weight" in src_batch_data or "sample_weight" in tgt_batch_data:
                if "sample_weight" in src_batch_data:
                    weight_lst = src_batch_data["sample_weight"]
                else:
                    weight_lst = [1] * len(src_batch_data["label"])
                
                if "sample_weight" in tgt_batch_data:
                    weight_lst = tgt_batch_data["sample_weight"]
                else:
                    weight_lst = [1] * len(tgt_batch_data["label"])
            
                if self.use_cuda:
                    weight_lst = weight_lst.to('cuda')
                graph_loss = torch.mean(graph_loss * weight_lst)
            
            graph_loss.backward()
            
            #clip the values if necessary
            if self.discriminator_clip_value:
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(-self.discriminator_clip_value, self.discriminator_clip_value)

            self.discriminator_optimizer.step()
            #sum the loss
            self.discriminator_loss += graph_loss.item()

            # zero gradients for optimizer
            self.tgt_encoder_optimizer.zero_grad()
            T = self.tgt_encoder_temperature

            # predict on discriminator
            pred_tgt = self.model.discriminator(tgt_text_embeddings)

            # logits for KL-divergence
            with torch.no_grad():
                src_prob = torch.nn.functional.softmax(self.model.classifier(src_text_embeddings) / T, dim=-1)
            tgt_prob = torch.nn.functional.log_softmax(self.model.classifier(src_tgt_text_embeddings) / T, dim=-1)

            kd_loss = self.KLDivLoss(tgt_prob, src_prob.detach()) * T * T
            tgt_encoder_loss = self.tgt_encoder_loss_fn(pred_tgt, domain_label_src)
            loss_tgt = self.tgt_loss_alpha * tgt_encoder_loss + self.tgt_loss_beta * kd_loss

            # multiply by the weights, if any, and calculate the mean value
            if "sample_weight" in tgt_batch_data:
                tgt_weight_lst = tgt_batch_data["sample_weight"]
                if self.use_cuda:
                    tgt_weight_lst = tgt_weight_lst.to('cuda')
                loss_tgt = torch.mean(loss_tgt * tgt_weight_lst)

            # compute loss for target encoder
            loss_tgt.backward()
            torch.nn.utils.clip_grad_norm_(self.tgt_encoder.parameters(), self.max_grad_norm)
            # optimize target encoder
            self.tgt_encoder_optimizer.step()

        total_time = (time.time() - start_time)/60
        print(f"    took: {total_time:.2f} min")

        self.epoch += 1

    def predict(self, data=None, encoder=None):
        self.model.eval()
        self.src_encoder.eval()
        self.tgt_encoder.eval()
        
        partial_main_loss = 0

        all_stance_pred = None
        all_stance_labels = None

        all_topic_pred = None
        all_topic_labels = None

        if data is None:
            data = self.dataloader
        
        if encoder is None:
            encoder = self.tgt_encoder

        for batch_n, batch_data in tqdm(enumerate(data)):
            with torch.no_grad():
                #zero gradients before every optimizer step
                label_tensor = torch.stack(batch_data["label"]).T.type(torch.FloatTensor)
                if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                    label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
                label_tensor = label_tensor.squeeze()

                if batch_n:
                    all_stance_labels = torch.cat((all_stance_labels, label_tensor))
                else:
                    all_stance_labels = label_tensor

                if self.use_cuda:
                    label_tensor = label_tensor.to("cuda")
            
                # get the embeddings for text and topic and creates a dict to pass as params to the model
                text_embeddings = encoder(**batch_data["text"])[:,-1,:]

                #apply the text and topic embeddings to the model
                y_pred = self.model(text_embeddings=text_embeddings)

                if batch_n:
                    all_stance_pred = torch.cat((all_stance_pred, y_pred.cpu()))
                else:
                    all_stance_pred = y_pred.cpu()

                # calculate the loss, and backprogate it to update weights
                graph_loss = self.loss_function(y_pred.squeeze(), label_tensor)
                if "sample_weight" in batch_data:
                    weight_lst = batch_data["sample_weight"]
                    if self.use_cuda:
                        weight_lst = weight_lst.to('cuda')
                    graph_loss = torch.mean(graph_loss * weight_lst)
                
                partial_main_loss += graph_loss.item()

            #sum the loss
            # partial_loss += graph_loss.item()
        avg_loss = partial_main_loss / batch_n #loss per batch

        return all_stance_pred, all_stance_labels, avg_loss

import re
class LLMTorchModelHandler(TorchModelHandler):
    def __init__(self, num_ckps=1, checkpoint_path='./data/checkpoints/', use_cuda=False, **params):

        TorchModelHandler.__init__(
            self,
            num_ckps=num_ckps,
            checkpoint_path=checkpoint_path,
            use_cuda=use_cuda,
            **params
        )
        
        self.is_llm = True
        self.model = params["model"]
        self.model_params = params.get("model_params", {})
        self.dataset = params["dataset"]
        self.tokenizer = self.dataset.tokenizer
        self.tokenized_input = self.tokenizer is None
        self.output_format = params["output_format"]
        self.output_max_score = int(params.get("output_max_score", "10"))
        self.output_parser = params.get("output_parser", self._output_parser_fn)
        self.save_every_n_batches = int(params.get("save_every_n_batches", "0"))
        self.output_err_default = params.get("output_err_default", "0.0")
        
        if self.output_format == "score":
            self.output_err_default = float(self.output_err_default)
        
        self.output_class_order = params.get("output_class_order")
        self.output_class_map = None
        if self.output_class_order is not None:
            self.output_class_order = self.output_class_order.split(",")
            self.output_class_map = [self.dataset.tgt2vec[k] for k in self.output_class_order]

    def _output_parser_fn(self, output):
        output = output.strip().lower()

        try:
            if self.output_format == "set":
                possible_values = '|'.join(self.dataset.tgt2vec.keys())
                set_regex = re.search(f"({possible_values})", output)

                prediction = output[set_regex.start():set_regex.end()].lower()
                prediction = self.dataset.convert_lbl_to_vec(prediction, self.dataset.tgt2vec)
                prediction = (np.argmax(prediction)+0.5)/self.num_labels
            
            elif self.output_format == "score":
                score_regex = re.search("([0-9]+)(/[0-9]+)*", output)
                
                prediction = output[score_regex.start():score_regex.end()]
                if "/" in prediction:
                    prediction = prediction.split("/")
                    prediction = float(prediction[0])/float(prediction[1])
                else:
                    prediction = float(prediction)/self.output_max_score
                
                if self.output_class_map is not None:
                    # convert float back to integer (up to num_labels)
                    prediction = math.floor(prediction*self.num_labels)
                    # convert int to vec (consistent with dataset)
                    prediction = self.output_class_map[prediction]
                    # convert vec to new int (order in dataset)
                    prediction = (np.argmax(prediction)+0.5)/self.num_labels

        except Exception as ex:
            prediction = None

        return prediction
    
    def predict(self, data=None):
        all_y_pred = []
        all_labels = []
        all_indices = []
        errors_ = []
        if data is None:
            data = self.dataloader
        
        call_count = 0
        for batch_n, batch_data in tqdm(enumerate(data)):
            label_tensor = torch.stack(batch_data["label"]).T.type(torch.LongTensor)
            if len(label_tensor.shape) > 1 and label_tensor.shape[-1] != 1:
                label_tensor = label_tensor.argmax(dim=1).reshape(-1,1)
            label_tensor = label_tensor.squeeze(-1).numpy()
                
            all_labels += label_tensor.tolist()
            all_indices += batch_data["index"]

            
            if self.model.model_type in ["llama_cpp", "hf_api"]:
                prompt_list = batch_data["prompt"]

                str_output_tensor = []
                for prompt_ in prompt_list:
                    str_output_tensor += [
                        self.model(
                            prompt = prompt_,
                            params = self.model_params,
                        )
                    ]
                                
            elif self.model.model_type == "hugging_face":

                str_output_tensor = []
                for prompt_ids in prompt_list:
                    output_tensor = self.model(
                        prompt_ids = prompt_ids,
                        params = self.model_params,
                    )

                    str_output_tensor += [self.dataset.decode_tokens(output_tensor)]

            for idx, str_output in zip(batch_data["index"], str_output_tensor):
                parsed_out = self.output_parser(str_output)
                if parsed_out is None:
                    parsed_out = self.output_err_default
                    errors_ += [(float(idx), prompt_, str_output)]

                all_y_pred += [parsed_out]

            if self.save_every_n_batches and batch_n % self.save_every_n_batches == 0:
                out_dict = {
                    "pred": np.array(all_y_pred).tolist(),
                    "index": np.array(all_indices).tolist(),
                }
                os.makedirs(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/", exist_ok=True)
                with open(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/{self.name}.ckp", mode="w", encoding="utf-8") as f_:
                    json.dump(out_dict, f_)

        # save the last batch that is not full size
        out_dict = {
            "pred": np.array(all_y_pred).tolist(),
            "index": np.array(all_indices).tolist(),
        }
        os.makedirs(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/", exist_ok=True)
        with open(f"{self.checkpoint_path}/llama_cpp_pred_checkpoints/{self.name}.ckp", mode="w", encoding="utf-8") as f_:
            json.dump(out_dict, f_)

        if len(errors_) > 0:
            os.makedirs(f"{self.checkpoint_path}/llama_cpp_errors/", exist_ok=True)
            with open(f"{self.checkpoint_path}/llama_cpp_errors/{self.name}.err", mode="w", encoding="utf-8") as f_:
                json.dump(errors_, f_)

        return np.array(all_y_pred), np.array(all_labels), None
    