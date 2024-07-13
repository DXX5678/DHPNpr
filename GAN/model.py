import torch
import torch.nn as nn


class ArgmaxLayer(nn.Module):
    def __init__(self):
        super(ArgmaxLayer, self).__init__()

    def forward(self, x):
        argmax_indices = torch.argmax(x, dim=-1, keepdim=False)
        return argmax_indices.long()


class Gan(nn.Module):
    def __init__(self, generator, discriminator):
        super(Gan, self).__init__()
        self.generator = generator
        self.lsm = nn.LogSoftmax(dim=-1)
        self.argmax_layer = ArgmaxLayer()
        self.discriminator = discriminator

    def forward(self, buggy_method_ids, buggy_method_mask, source_ids, source_mask, target_ids, target_mask, args,
                no_share):
        if args.model_type == "codet5":
            out = self.generator(source_ids, source_mask, target_ids, target_mask)
            patch_out = self.lsm(out.logits)
        elif args.model_type == "unixcoder":
            _, _, _, patch_out = self.generator(source_ids, target_ids)
            patch_out = self.lsm(patch_out)
        else:
            _, _, _, patch_out = self.generator(source_ids, source_mask, target_ids, target_mask)
            patch_out = self.lsm(patch_out)
        if no_share == 1:
            patch_ids = self.argmax_layer(patch_out).view(target_ids.shape[0], -1)
            output = self.discriminator(buggy_method_ids, patch_ids)
        elif no_share == 2:
            patch_ids = self.argmax_layer(patch_out).view(target_ids.shape[0], -1)
            output = self.discriminator(buggy_method_ids, patch_ids, buggy_method_mask, target_mask)
        else:
            patch_ids = self.argmax_layer(patch_out).view(target_ids.shape[0], -1)
            encoder_output1 = self.generator.encoder(buggy_method_ids, attention_mask=buggy_method_mask)
            encoder_output2 = self.generator.encoder(patch_ids, attention_mask=target_mask)
            s_embedding = encoder_output1[0]
            p_embedding = encoder_output2[0]
            output = self.discriminator(s_embedding, p_embedding, buggy_method_mask, target_mask)
        return output
