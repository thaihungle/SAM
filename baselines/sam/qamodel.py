from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.sam.utils import MLP, LayerNorm, OptionalLayer
from baselines.sam.stm_basic import STM

AVAILABLE_ELEMENTS = ('e1', 'e2', 'r1', 'r2', 'r3')




class QAmodel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(QAmodel, self).__init__()
        self.mlp_size = config["hidden_size"]
        self.input_module = InputModule(config)
        self.update_module = STM(config["symbol_size"], output_size=config["symbol_size"],
                                        init_alphas=[1, None, 0],
                                        learn_init_mem=True, mlp_hid=config['hidden_size'],
                                         num_slot=config["role_size"],
                                         slot_size=config["entity_size"],
                                        rel_size=96)

        self.infer_module = InferenceModule(config=config)

        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        story_embed, query_embed = self.input_module(story, query)
        out, (_,_,R) = self.update_module(story_embed.permute(1,0,2))
        R = R.permute(0,2,1,3)
        logits = self.infer_module(query_embed, R)
        return logits



class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_sum = torch.einsum('bswe,we->bse', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_sum = torch.einsum('bwe,we->be', query_embed, self.pos_embed[:query_embed.shape[1]])
        return sentence_sum, query_sum



class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Parameter(torch.zeros(config["entity_size"], config["vocab_size"]))
        nn.init.xavier_uniform_(self.Z.data)

        # TODO: remove unused entity head?
        self.e = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.ent_size) for _ in range(2)])
        self.r = nn.ModuleList([MLP(equation='be,er->br', in_features=self.symbol_size,
                                    hidden_size=self.hidden_size, out_size=self.role_size) for _ in range(3)])
        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, query_embed: torch.Tensor, TPR: torch.Tensor):
        e1, e2 = [module(query_embed) for module in self.e]
        r1, r2, r3 = [module(query_embed) for module in self.r]

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = torch.einsum('bf,fl->bl', step_sum, self.Z.data)
        return logits

