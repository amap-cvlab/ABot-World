"""Independent model text conditioning, without action catalogs or service state."""
import threading
import torch
from abot_world.iris.utils.text_encoder_staging import initial_encoder_residency
from abot_world.iris.utils.grouped_text_encoding import encode_initial_locals
from abot_world.iris.utils.runtime_memory import prepare_cpu_encoder, cpu_encoder_batch

class InferenceText:
    def __init__(self, encoder, device, cache=None):
        self.encoder, self.device = encoder, device
        self.lock = threading.RLock()
        self.global_prompt = None
        self.global_embedding = None
        self.precomputed_globals = cache["global"] if cache else {}
        self.locals = dict(cache["local"]) if cache else {}

    def prepare(self, global_prompt, local_prompts=("",)):
        """Encode a global row once and requested independent local rows on GPU."""
        with self.lock:
            if self.global_prompt != global_prompt:
                self.global_embedding = self.precomputed_globals.get(global_prompt)
                if self.global_embedding is None:
                    with initial_encoder_residency(self.encoder, self.device):
                        self.global_embedding = self.encoder(text_prompts=[global_prompt], device=self.device,
                                                             max_length=384)['prompt_embeds'].detach().cpu()
                prepare_cpu_encoder(self.encoder)
                self.global_prompt = global_prompt
            missing = list(dict.fromkeys(p for p in local_prompts if p not in self.locals))
            if str(self.device).split(':')[0] == 'cuda' and missing:
                rows = encode_initial_locals(self.encoder, missing, self.device, max_length=128)
            else:
                with cpu_encoder_batch(self.encoder):
                    rows = [self.encoder(text_prompts=[p], device='cpu', max_length=128)['prompt_embeds'].detach().cpu()
                            for p in missing]
            self.locals.update(zip(missing, rows))
            return {p: self.combine(self.locals[p]) for p in local_prompts}

    def encode_local(self, text):
        """Encode a later local update on the shared CPU encoder."""
        with self.lock:
            if text not in self.locals:
                with cpu_encoder_batch(self.encoder):
                    self.locals[text] = self.encoder(text_prompts=[text], device='cpu', max_length=128)['prompt_embeds'].detach().cpu()
            return self.combine(self.locals[text])

    def combine(self, local):
        if self.global_embedding is None:
            raise RuntimeError('Prepare global text first')
        local = local.detach().to(device='cpu', dtype=self.global_embedding.dtype)
        if self.global_embedding.shape[:2] != (1,384) or local.shape != (1,128,self.global_embedding.shape[2]):
            raise ValueError('Expected [1,384,D] global and [1,128,D] local rows')
        return torch.cat([self.global_embedding, local], dim=1)
