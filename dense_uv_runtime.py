import io
import sys
from pathlib import Path


class DenseUVInferenceRuntime:
    """Persistent production runtime for the SkingToolkit v61 pipeline."""

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
        from SkingToolkit.dense_uv_parser.runtime import get_device
        from SkingToolkit.dense_uv_parser.uv_layout import (
            tensor_to_rgba_image,
            view_native_size,
        )
        from SkingToolkit.dense_uv_parser.utils import (
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
        self.view_native_size = view_native_size
        self.estimate_foreground = estimate_top_left_flood_foreground
        self.splat = splat_parser_predictions_to_uv_conditioning

        self.device = get_device(device)
        self.model, self.parser_args = load_parser(
            str(checkpoint),
            self.device,
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
            color_tolerance=0.03,
        )
        parser_rendered = self.build_parser_input(
            rendered,
            observed_foreground,
            bg_color=self.bg_color,
            background_mode="adaptive",
        )

        with self.torch.inference_mode():
            outputs = self.model(
                parser_rendered,
                view_ids=self.view_ids,
                semantic_foreground=observed_foreground,
            )
            conditioning, _ = self.splat(
                rendered,
                outputs,
                renderer=self.renderer,
                views=self.views,
                group_size=len(self.views),
                fg_threshold=0.5,
                bg_color=self.bg_color,
                semantic_gate=True,
                affine_refine=False,
                affine_refine_translation_px=0.0,
                affine_refine_scale=0.0,
                route_confidence_threshold=0.0,
                route_margin_threshold=0.0,
                outer_route_confidence_threshold=0.80,
                outer_route_margin_threshold=0.55,
                outer_uv_min_coverage=0.25,
                outer_uv_min_source_pixels=33,
                outer_silhouette_consistency=True,
                outer_silhouette_min_coverage=0.50,
                outer_silhouette_dilation=0,
                outer_silhouette_min_pixels=4,
                outer_geometry_rescue=True,
                outer_semantic_rescue=True,
                outer_semantic_presence_threshold=0.80,
                outer_semantic_coverage_threshold=0.20,
                outer_rescue_confidence_threshold=0.60,
                outer_rescue_margin_threshold=0.25,
                outer_rescue_min_coverage=0.10,
                color_aggregation="grid_mode",
                geometry_route_texel_consensus=True,
                geometry_route_texel_consensus_weight=0.60,
                geometry_route_preserve_outer_confidence=0.80,
                geometry_route_preserve_outer_margin=0.35,
                geometry_route_consensus_outer_confidence=0.70,
                geometry_route_consensus_outer_margin=0.20,
                geometry_cross_view_outer_consistency=False,
                geometry_cross_view_outer_weight=0.50,
                geometry_cross_view_outer_positive_confidence=0.70,
                geometry_cross_view_outer_positive_margin=0.20,
                geometry_cross_view_outer_negative_confidence=0.70,
                geometry_cross_view_outer_negative_margin=0.20,
                geometry_cross_view_outer_background_max_coverage=0.25,
                geometry_cross_view_outer_min_views=2,
                outer_uv_occupancy=False,
                outer_uv_occupancy_blend_weight=0.0,
                outer_uv_occupancy_gate_threshold=0.15,
                outer_uv_occupancy_rescue_threshold=0.70,
                outer_uv_occupancy_rescue_route_threshold=0.30,
                outer_uv_component_routing=False,
                outer_uv_component_seed_threshold=0.80,
                outer_uv_component_grow_threshold=0.50,
                outer_uv_component_min_size=2,
                observed_foreground=observed_foreground,
                background_color_tolerance=0.25,
                color_background_tolerance=0.031372549,
                color_foreground_inset=1,
                reject_semantic_fallback=True,
                include_rejected_context=False,
                include_confidence=False,
                return_details=True,
            )

        repaired, _ = self.simple_inpaint_uv(
            conditioning.detach().cpu(),
            alpha_threshold=0.5,
        )
        output = io.BytesIO()
        self.tensor_to_rgba_image(repaired).save(
            output,
            format="PNG",
        )
        return output.getvalue()
