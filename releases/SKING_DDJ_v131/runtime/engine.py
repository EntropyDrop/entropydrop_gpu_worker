"""Provider-independent fixed-camera edited image to a 64x64 classic skin."""
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from base_engine import UVEngine as BaseEngine, sha
from head_alpha import assemble
from torso_models import (
    IDS, OUTER, FB_IDS, FB_MASK, FACE_IDS, Normalizer,
    TorsoAdapter, FlatJudge, quantize, fuse, crop_image,
)
from torso_geometry import FastRaster, analyze, remove_components
from renderer import DifferentiableRenderer


class UVEngine(BaseEngine):
    def __init__(self, root=None):
        super().__init__(root)
        state = torch.load(self.root / 'parser.pt', map_location='cpu', weights_only=True)
        self.torso_adapter = TorsoAdapter().cuda().eval().requires_grad_(False)
        self.torso_adapter.load_state_dict(state['torso_adapter'], strict=True)
        self.face_judge = FlatJudge().cuda().eval().requires_grad_(False)
        self.face_judge.load_state_dict(state['face_judge'], strict=True)
        self.normalizer = Normalizer(self.root / 'background_guard.npz')
        self.torso_query = (self.head.region % 6 == 1).nonzero().flatten()
        assert np.array_equal(self.head.ids[self.torso_query].cpu().numpy(), IDS)
        self.torso_features = None
        self.torso_aux = None
        self.head.norm.register_forward_hook(self._capture_torso)
        self.head.register_forward_hook(self._capture_aux)
        renderer = DifferentiableRenderer(str(self.root / 'mappings'), bg_color=[97/255, 2/255, 101/255])
        self.raster = FastRaster(renderer)
        self.policy = self.pipeline['torso_geometry']

    def _capture_torso(self, module, inputs, output):
        self.torso_features = output[:, self.torso_query].detach()

    def _capture_aux(self, module, inputs, output):
        self.torso_aux = output[1]

    @torch.no_grad()
    def _evidence(self, a, full=False):
        # Preserve the validated CPU normalization order before transfer to CUDA.
        image = (torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255)[None].cuda()
        if full:
            evidence = self.extract(image)
            prediction = evidence['uv_v2']
        else:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, _ = self.head(image, encoder=self.encoder)
        aux = self.torso_aux
        uv = prediction.flatten(2)[:, :, self.head.ids[self.torso_query]].transpose(1, 2)
        view = aux['view_visibility_logits'][:, :, self.torso_query].sigmoid().transpose(1, 2)
        features = torch.cat([self.torso_features.float(), uv.float(), view.float()], -1)[0].cpu().half()
        logits = aux['alpha_logits'][0, self.torso_query][torch.as_tensor(OUTER, device=self.torso_query.device)].float().cpu()
        result = {'features': features, 'logits': logits}
        if full:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                head_logits, _ = self.refiner(evidence)
            skin = assemble(evidence, self.refiner, head_logits)
            skin.flatten(2)[:, 3, self.refiner.ids[self.refiner.outer]] = (
                head_logits.sigmoid() >= self.pipeline['head_outer_threshold']).float()
            result.update(uv=torch.from_numpy(quantize(skin)), body_rgb=evidence['uv_v1'][0, :3].float().cpu())
        return result

    @torch.no_grad()
    def infer(self, input_path, output_path):
        started = time.monotonic()
        with Image.open(input_path) as im:
            if im.size != (512, 512):
                raise ValueError('SKING_DDJ_v131 expects a 512x512 front_left + back_left edited render')
            bg = Image.new('RGBA', im.size, (97, 2, 101, 255))
            bg.alpha_composite(im.convert('RGBA'))
            a = np.array(bg.convert('RGB')).copy()
        normal, normal_mask, _ = self.normalizer(a)
        raw, safe = self._evidence(a, full=True), self._evidence(normal)
        f = torch.stack([raw['features'], safe['features']])
        z = torch.stack([raw['logits'], safe['logits']])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            adapted, _ = self.torso_adapter(f.cuda(), z.cuda())
        torso_logits = adapted[-1].cpu()
        parent_uv = fuse(raw['uv'], raw['body_rgb'], torso_logits)
        face_features = f[:, OUTER][:, FB_MASK][None].cuda()
        face_logits = torch.cat([z[:, FB_MASK], torso_logits[FB_MASK][None]], 0)[None].cuda()
        images = crop_image(normal)[None].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, empty_logits = self.face_judge((face_features, images), face_logits)
        empty = empty_logits.sigmoid().cpu()[0].numpy()
        active = torso_logits[FB_MASK].numpy() >= 0
        whole_face = active & (empty[FACE_IDS] >= self.pipeline['torso_empty_threshold'])
        removed = set(FB_IDS[whole_face].tolist())
        components = analyze(self.raster, a, parent_uv)
        removed |= remove_components(components, self.policy)
        pixels = parent_uv.copy()
        if removed:
            pixels.reshape(-1, 4)[list(removed)] = 0
        assert not ((pixels[..., 3] > 127) & (parent_uv[..., 3] < 128)).any()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        self.calls += 1
        record = dict(
            complete=True, model_version='SKING_DDJ_v131', input_sha256=sha(input_path),
            output_sha256=sha(path), release_sha256=sha(self.root / 'release.json'),
            checkpoint_sha256=self.release['sha256']['parser.pt'], pipeline=self.pipeline,
            worker_pid=os.getpid(), generation_index=self.calls, seconds=time.monotonic() - started,
            normalized_background_pixels=int(normal_mask.sum()),
            removed_torso_front_back_pixels=len(removed), input_reference_uv_used=False,
        )
        path.with_suffix(path.suffix + '.json').write_text(json.dumps(record, indent=2) + '\n')
        return record
