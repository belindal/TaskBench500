import torch
import numpy as np
import torch.nn.functional as F


class PromptEmbedding(torch.nn.Module):

    def __init__(self, embed, n_prefix):
        super().__init__()
        self.embed = embed
        self.new_embed = torch.nn.Embedding(n_prefix, embed.embedding_dim)

        # randomly init to tokens in actual vocab
        indices = np.random.permutation(range(5000))[:n_prefix]
        init_weight = self.embed.state_dict()["weight"][indices]
        self.new_embed._load_from_state_dict({"weight": init_weight},
                                             "", None, True, [], [], "")

    def forward(self, input):
        return F.embedding(
            input,
            torch.cat([self.embed.weight, self.new_embed.weight], 0),
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse)


def set_extra_embeddings(arch, model, n_prefix):
    if 't5' in arch:
        model.set_input_embeddings(
            PromptEmbedding(model.shared, n_prefix)
        )
