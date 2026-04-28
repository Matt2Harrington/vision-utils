import logging
import shlex
import subprocess
import numpy as np
import scipy
import cv2
import os
from transformers import pipeline, Pipeline
from PIL import Image, ImageChops
from moviepy import ImageSequenceClip, VideoFileClip
from torch.multiprocessing import Pool, Process, set_start_method, cpu_count
from collections import namedtuple
from typing import List, Optional
from spatialconverter.file_mixin import FileMixin
from spatialconverter.timer import timing

set_start_method("spawn", force=True)


FrameData = namedtuple("FrameData", ["index", "frame"])


class VideoHandler(FileMixin):
    MODEL_MAP = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(
        self,
        filename,
        model_size: str = "large",
        target_fps: float | None = None,
        shift_left: int = 10,
        shift_right: int = 50,
        hfov: float = 63.4,
        cdist: float = 19.24,
        hadjust: float = 0.02,
        projection: str = "rect",
        spatial_extra: str = "",
        zoom: float = 1.0,
        stereo_format: str = "ou",
        spatial_enabled: bool = True,
    ):
        self.filename = filename
        self.directory = None
        self.pipe = None
        self.model_size = model_size
        self.target_fps = target_fps
        self.shift_left = shift_left
        self.shift_right = shift_right
        self.hfov = hfov
        self.cdist = cdist
        self.hadjust = hadjust
        self.projection = projection
        self.spatial_extra = spatial_extra
        self.zoom = zoom
        self.stereo_format = stereo_format if stereo_format in ("ou", "sbs") else "ou"
        self.spatial_enabled = spatial_enabled

    def over_under_video_filename(self):
        return f"{self.get_directory_name()}/over_under.mp4"

    def spatial_video_filename(self):
        return f"{self.get_directory_name()}/spatial_video.mov"

    def spatial_audio_filename(self):
        return f"{self.get_directory_name()}/temp-audio.m4a"

    def get_pipe(self) -> Pipeline:
        """
        Depth-Anything-V2 model from https://github.com/DepthAnything/Depth-Anything-V2.
        Small is fastest, Large is highest quality. Default Large preserves prior behavior.
        """
        if self.pipe is None:
            model = self.MODEL_MAP.get(self.model_size, self.MODEL_MAP["large"])
            self.pipe = pipeline(task="depth-estimation", model=model)
        return self.pipe

    def _apply_zoom(self, frame):
        """Scale the frame around its center by self.zoom while preserving frame size.

        zoom > 1.0 crops the center and scales it back up (zoom in).
        zoom < 1.0 scales the frame down and pads the rest with black (zoom out).
        zoom == 1.0 returns the frame unchanged.
        """
        if self.zoom is None or abs(self.zoom - 1.0) < 1e-6:
            return frame
        h, w = frame.shape[:2]
        if self.zoom > 1.0:
            new_h = max(1, int(h / self.zoom))
            new_w = max(1, int(w / self.zoom))
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            cropped = frame[y0:y0 + new_h, x0:x0 + new_w]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            new_h = max(1, int(h * self.zoom))
            new_w = max(1, int(w * self.zoom))
            scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = np.zeros_like(frame)
            y0 = (h - new_h) // 2
            x0 = (w - new_w) // 2
            result[y0:y0 + new_h, x0:x0 + new_w] = scaled
            return result

    @timing
    def produce_frames(self):
        """Return a list of frames; if target_fps is below source fps, sample at a stride."""
        capture = cv2.VideoCapture(self.filename)
        src_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        if self.target_fps and src_fps > 0 and self.target_fps < src_fps:
            stride = src_fps / self.target_fps
        else:
            stride = 1.0
        frame_idx = 0
        kept_idx = 0
        next_target = 0.0
        result = []
        while True:
            return_code, frame = capture.read()
            if not return_code:
                break
            if frame_idx >= next_target:
                kept_idx += 1
                result.append((self._apply_zoom(frame), kept_idx))
                next_target += stride
            frame_idx += 1
        capture.release()
        return result

    def shift_image(self, data, shift_amount=10):
        # Ensure depth image is grayscale (for single value)
        data_with_alpha = np.dstack(
            [data, np.full(data.shape[:2], 255, dtype=np.uint8)]
        )
        frame = Image.fromarray(data_with_alpha, "RGBA")
        frame_depth = self.get_pipe()(frame)["depth"]
        depth_image = frame_depth.convert("L")
        depth_data = np.array(depth_image)
        deltas = np.array((depth_data / 255.0) * float(shift_amount), dtype=int)

        shifted_data = np.zeros_like(data_with_alpha)

        width = frame.width

        for y, row in enumerate(deltas):
            width = len(row)
            x = 0
            while x < width:
                dx = row[x]
                if x + dx >= width:
                    break
                if x - dx < 0:
                    shifted_data[y][x - dx] = [0, 0, 0, 0]
                else:
                    shifted_data[y][x - dx] = data_with_alpha[y][x]
                x += 1

        # Convert the pixel data to an image.
        shifted_image = Image.fromarray(shifted_data)

        alphas_image = Image.fromarray(
            scipy.ndimage.binary_fill_holes(
                ImageChops.invert(shifted_image.getchannel("A"))
            )
        ).convert("1")
        shifted_image.putalpha(ImageChops.invert(alphas_image))
        return shifted_image

    def inpaint(self, image):
        org_image = np.array(image)
        damaged_image = np.array(image)

        # Convert all pixels greater than zero to black while black becomes white
        # Assuming damaged_image is a NumPy array of shape (height, width, 3)
        mask = damaged_image.sum(axis=2) > 0

        # Initialize a new array with the same shape as damaged_image, filled with white pixels
        new_image = np.ones_like(damaged_image) * 255
        new_image[mask] = [0, 0, 0]

        # saving the mask
        mask = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        inpainted = cv2.inpaint(org_image, mask, 7, cv2.INPAINT_TELEA)
        return inpainted

    @timing
    def make_preview(self, output_path: Optional[str] = None, timestamp_sec: Optional[float] = None) -> str:
        """Single-frame preview: extract one frame, run depth-shift + stereo
        stack with current settings, write a PNG, skip encoding and spatial
        tagging entirely. Used by the UI for fast iteration on settings."""
        capture = cv2.VideoCapture(self.filename)
        src_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if timestamp_sec is not None:
            target_idx = max(0, int(timestamp_sec * src_fps))
        else:
            target_idx = max(0, total_frames // 2)
        if total_frames > 0:
            target_idx = min(target_idx, total_frames - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(
                f"Could not read preview frame at index {target_idx} from {self.filename}"
            )

        frame = self._apply_zoom(frame)
        # Reuse the single-frame stereo builder. It logs `frame: 1` like a
        # normal run; benign for preview.
        result = self.create_over_under_video_frame((frame, 1))
        # result.frame is RGB after the cvtColor in that method; cv2.imwrite
        # expects BGR.
        bgr = cv2.cvtColor(result.frame, cv2.COLOR_RGB2BGR)

        if output_path is None:
            output_path = f"{self.get_directory_name()}/preview.png"
        cv2.imwrite(output_path, bgr)
        logging.info(f"Preview frame {target_idx} of {total_frames} -> {output_path}")
        logging.info(f"OUTPUT: {output_path}")
        return output_path

    def create_over_under_video_frame(self, frame) -> List[FrameData]:
        """Build the stereo frame. Stacking depends on self.stereo_format:
            "ou"  → left on top, right on bottom (vconcat)
            "sbs" → left on left, right on right (hconcat)
        """
        image, index = frame
        logging.info(f"frame: {index}")

        # Shift and inpaint images
        shifted_left_image = self.shift_image(image, self.shift_left).convert("RGB")
        inpainted_left_image = self.inpaint(shifted_left_image)
        shifted_right_image = self.shift_image(image, self.shift_right).convert("RGB")
        inpainted_right_image = self.inpaint(shifted_right_image)

        if self.stereo_format == "sbs":
            stacked_image = cv2.hconcat([inpainted_left_image, inpainted_right_image])
        else:
            stacked_image = cv2.vconcat([inpainted_left_image, inpainted_right_image])
        stacked_image = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB)
        return FrameData(index, stacked_image)

    @timing
    def make_video(self):
        frames = self.produce_frames()
        logging.info(f"Processed {len(frames)} frames")

        # Use the number of cpus that your computer has. This doesn't work on all systems
        # but we're using this as an approximation to parallelize running on each frame
        multi_pool = Pool(processes=cpu_count())
        output = multi_pool.map(self.create_over_under_video_frame, frames)
        multi_pool.close()
        multi_pool.join()

        # Since we parallelized this, let's re-sort the frames by the index
        sorted_frames = sorted(output, key=lambda x: x.index)
        video_clip = VideoFileClip(self.filename)
        # If we down-sampled, render at the target fps so duration matches the source audio.
        out_fps = self.target_fps if (self.target_fps and self.target_fps < video_clip.fps) else video_clip.fps
        clip = ImageSequenceClip([obj.frame for obj in sorted_frames], fps=out_fps)
        clip = clip.with_audio(video_clip.audio)
        clip.write_videofile(
            self.over_under_video_filename(),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=self.spatial_audio_filename(),
            remove_temp=True,
        )

        if not self.spatial_enabled:
            logging.info(
                f"Spatial tagging disabled; raw stereo output left at "
                f"{self.over_under_video_filename()}"
            )
            logging.info(f"OUTPUT: {self.over_under_video_filename()}")
            return

        logging.info("Running spatial tagger")
        spatial_cmd = [
            "./spatial", "make",
            "-i", self.over_under_video_filename(),
            "-f", self.stereo_format,
            "-o", self.spatial_video_filename(),
            "--cdist", str(self.cdist),
            "--hfov", str(self.hfov),
            "--hadjust", str(self.hadjust),
            "--projection", self.projection,
        ]
        if self.spatial_extra:
            # Append free-form extra args (e.g. extra `spatial` flags). Later
            # flags override earlier ones for `spatial`, so this is the override.
            spatial_cmd += shlex.split(self.spatial_extra)
        logging.info(f"spatial cmd: {' '.join(shlex.quote(a) for a in spatial_cmd)}")
        result = subprocess.run(spatial_cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"./spatial exited with code {result.returncode}")
        logging.info(f"OUTPUT: {self.spatial_video_filename()}")
