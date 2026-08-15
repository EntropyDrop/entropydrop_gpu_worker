import io
import sys
from pathlib import Path


class DenseUVInferenceRuntime:
    """Persistent production runtime for the current Dense UV pipeline."""

    def __init__(
        self,
        toolkit_root: str,
        checkpoint_path: str,
        mappings_dir: str,
        device: str = "cuda",
    ):
        toolkit_path = Path(toolkit_root).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        mappings = Path(mappings_dir).resolve()
        if not toolkit_path.is_dir():
            raise FileNotFoundError(
                f"SkingToolkit directory does not exist: {toolkit_path}"
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Dense UV checkpoint does not exist: {checkpoint}"
            )
        if not mappings.is_dir():
            raise FileNotFoundError(
                f"Dense UV mappings directory does not exist: {mappings}"
            )

        toolkit_parent = str(toolkit_path.parent)
        if toolkit_parent not in sys.path:
            sys.path.insert(0, toolkit_parent)

        import torch
        from SkingToolkit.dense_uv_parser.foreground import (
            build_parser_input,
        )
        from SkingToolkit.dense_uv_parser.infer import (
            image_to_render_tensor,
            load_parser,
            simple_inpaint_uv,
        )
        from SkingToolkit.dense_uv_parser.inference_config import (
            production_preprocessing_defaults,
            production_splat_defaults,
        )
        from SkingToolkit.dense_uv_parser.runtime import get_device
        from SkingToolkit.dense_uv_parser.uv_layout import (
            tensor_to_rgba_image,
            view_native_size,
        )
        from SkingToolkit.dense_uv_parser.semantic_targets import (
            head_outer_face_values_to_uv,
        )
        from SkingToolkit.dense_uv_parser.utils import (
            attach_projected_head_outer_structure,
            attach_projected_outer_uv_occupancy,
            estimate_top_left_flood_foreground,
            parse_views,
            splat_parser_predictions_to_uv_conditioning,
            surface_class_count,
        )
        from SkingToolkit.renderer import DifferentiableRenderer

        self.torch = torch
        self.build_parser_input = build_parser_input
        self.image_to_render_tensor = image_to_render_tensor
        self.simple_inpaint_uv = simple_inpaint_uv
        self.tensor_to_rgba_image = tensor_to_rgba_image
        self.head_outer_face_values_to_uv = head_outer_face_values_to_uv
        self.view_native_size = view_native_size
        self.estimate_foreground = estimate_top_left_flood_foreground
        self.attach_projected_outer_uv_occupancy = (
            attach_projected_outer_uv_occupancy
        )
        self.attach_projected_head_outer_structure = (
            attach_projected_head_outer_structure
        )
        self.splat = splat_parser_predictions_to_uv_conditioning
        self.preprocessing = production_preprocessing_defaults()
        self.splat_kwargs = production_splat_defaults()
        self.inpaint_kwargs = {
            "head_outer_threshold": self.splat_kwargs.pop(
                "head_outer_completion_threshold"
            ),
            "head_outer_min_component_seeds": self.splat_kwargs.pop(
                "head_outer_completion_min_component_seeds"
            ),
            "head_outer_symmetry_threshold": self.splat_kwargs.pop(
                "head_outer_symmetry_completion_threshold"
            ),
            "head_outer_symmetry_candidate_threshold": self.splat_kwargs.pop(
                "head_outer_symmetry_candidate_threshold"
            ),
            "head_outer_closed_ring_threshold": self.splat_kwargs.pop(
                "head_outer_closed_ring_completion_threshold"
            ),
            "head_outer_open_top_threshold": self.splat_kwargs.pop(
                "head_outer_open_top_completion_threshold"
            ),
            "head_outer_open_top_max_gap": self.splat_kwargs.pop(
                "head_outer_open_top_max_gap"
            ),
        }

        self.device = get_device(device)
        self.model, self.parser_args = load_parser(
            str(checkpoint),
            self.device,
        )
        head_precision = self.parser_args.get(
            "_checkpoint_head_outer_occupancy_precision"
        )
        head_recall = self.parser_args.get(
            "_checkpoint_head_outer_occupancy_recall"
        )
        head_reliable = (
            head_precision is not None
            and head_recall is not None
            and head_precision
            >= self.preprocessing["head_outer_topology_min_precision"]
            and head_recall
            >= self.preprocessing["head_outer_topology_min_recall"]
        )
        if self.preprocessing["head_outer_topology_auto_reliability"]:
            self.splat_kwargs["head_outer_topology_rescue"] = bool(
                self.splat_kwargs["head_outer_topology_rescue"]
                and head_reliable
            )
        self.views = parse_views(
            self.parser_args.get(
                "views",
                (
                    "front_left,"
                    "back_left"
                ),
            )
        )
        if self.model.view_classes not in (0, len(self.views)):
            raise ValueError(
                "Dense UV checkpoint view metadata does not match its "
                f"model: classes={self.model.view_classes}, "
                f"views={self.views}"
            )

        self.renderer = DifferentiableRenderer(
            mappings_dir=str(mappings)
        ).to(self.device)
        missing_views = [
            view for view in self.views if view not in self.renderer.views
        ]
        if missing_views:
            raise ValueError(
                "Dense UV mappings are missing checkpoint views: "
                + ", ".join(missing_views)
            )
        if self.model.predict_affine and self.model.surface_classes > 0:
            mapping_classes = surface_class_count(
                self.renderer,
                self.views,
            )
            if self.model.surface_classes != mapping_classes:
                raise ValueError(
                    "Dense UV checkpoint/mappings surface mismatch: "
                    f"checkpoint={self.model.surface_classes}, "
                    f"mappings={mapping_classes}"
                )

        self.bg_color = self.parser_args.get(
            "bg_color",
            (128, 128, 128),
        )
        self.view_ids = torch.arange(
            len(self.views),
            device=self.device,
        )

    def _load_combined_render(self, content: bytes):
        from PIL import Image

        with Image.open(io.BytesIO(content)) as source:
            combined = source.convert("RGB")
        width, height = combined.size
        if width % len(self.views) != 0:
            raise ValueError(
                f"Combined render width {width} is not divisible by "
                f"{len(self.views)} views"
            )
        view_width = width // len(self.views)
        images = [
            combined.crop(
                (
                    index * view_width,
                    0,
                    (index + 1) * view_width,
                    height,
                )
            )
            for index in range(len(self.views))
        ]
        tensors = [
            self.image_to_render_tensor(
                image,
                self.view_native_size(self.renderer, view),
                bg_color=self.bg_color,
            )
            for image, view in zip(images, self.views)
        ]
        return self.torch.stack(tensors, dim=0).to(self.device)

    def infer_png(self, combined_render: bytes) -> bytes:
        rendered = self._load_combined_render(combined_render)
        observed_foreground = self.estimate_foreground(
            rendered,
            color_tolerance=self.preprocessing[
                "foreground_flood_tolerance"
            ],
        )
        parser_rendered = self.build_parser_input(
            rendered,
            observed_foreground,
            bg_color=self.bg_color,
            background_mode=self.preprocessing[
                "foreground_parser_background"
            ],
        )

        with self.torch.inference_mode():
            outputs = self.model(
                parser_rendered,
                view_ids=self.view_ids,
                semantic_foreground=observed_foreground,
            )
            # Keep the production path identical to infer.py for checkpoints
            # that include the projected UV occupancy branch.
            outputs = self.attach_projected_outer_uv_occupancy(
                self.model,
                outputs,
                self.renderer,
                self.views,
                observed_foreground=observed_foreground,
                center_power=float(
                    self.parser_args.get("route_texel_center_power", 2.0)
                ),
            )
            outputs = self.attach_projected_head_outer_structure(
                self.model,
                outputs,
                self.renderer,
                self.views,
                observed_foreground=observed_foreground,
                source_images=parser_rendered,
                center_power=float(
                    self.parser_args.get("route_texel_center_power", 2.0)
                ),
            )
            conditioning, _ = self.splat(
                rendered,
                outputs,
                renderer=self.renderer,
                views=self.views,
                group_size=len(self.views),
                bg_color=self.bg_color,
                observed_foreground=observed_foreground,
                **self.splat_kwargs,
                return_details=True,
            )

        head_outer_probability = None
        head_outer_symmetry_probability = None
        head_outer_closed_ring_probability = None
        head_outer_open_top_probability = None
        if (
            int(
                getattr(
                    self.model,
                    "head_outer_projected_input_version",
                    1,
                )
            )
            >= 2
            and self.splat_kwargs["head_outer_topology_rescue"]
            and "head_outer_face_occupancy_logits" in outputs
        ):
            head_outer_probability = self.head_outer_face_values_to_uv(
                self.torch.sigmoid(
                    outputs["head_outer_face_occupancy_logits"].float()
                )
            )[0, 0].detach().cpu()
            if "head_outer_symmetry_logit" in outputs:
                head_outer_symmetry_probability = self.torch.sigmoid(
                    outputs["head_outer_symmetry_logit"].float()
                )[0].detach().cpu()
            if "head_outer_accessory_logits" in outputs:
                accessory_probability = self.torch.sigmoid(
                    outputs["head_outer_accessory_logits"].float()
                )[0].detach().cpu()
                head_outer_closed_ring_probability = accessory_probability[0]
                head_outer_open_top_probability = accessory_probability[1]

        repaired, _ = self.simple_inpaint_uv(
            conditioning.detach().cpu(),
            alpha_threshold=self.preprocessing["alpha_threshold"],
            head_outer_probability=head_outer_probability,
            head_outer_symmetry_probability=(
                head_outer_symmetry_probability
            ),
            head_outer_closed_ring_probability=(
                head_outer_closed_ring_probability
            ),
            head_outer_open_top_probability=(
                head_outer_open_top_probability
            ),
            **self.inpaint_kwargs,
        )
        output = io.BytesIO()
        self.tensor_to_rgba_image(repaired).save(
            output,
            format="PNG",
        )
        return output.getvalue()
