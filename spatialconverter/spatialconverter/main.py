import argparse
import logging
import sys
from spatialconverter.image_handler import ImageHandler
from spatialconverter.video_handler import VideoHandler

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 2D Photos & Videos")
    parser.add_argument("--photo", type=str, help="a file path to a photo")
    parser.add_argument("--video", type=str, help="a file path to a video")
    parser.add_argument(
        "--model-size",
        choices=["small", "base", "large"],
        default="large",
        help="Depth-Anything-V2 model size (smaller = faster, lower quality)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="downsample to this fps before processing (default: source fps)",
    )
    parser.add_argument(
        "--shift-left",
        type=int,
        default=10,
        help="left-eye depth shift amount (default 10)",
    )
    parser.add_argument(
        "--shift-right",
        type=int,
        default=50,
        help="right-eye depth shift amount (default 50)",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=63.4,
        help="horizontal field of view in degrees passed to ./spatial (default 63.4 = iPhone 15 Pro main lens; raise for wider lenses to reduce zoom on Vision Pro)",
    )
    parser.add_argument(
        "--cdist",
        type=float,
        default=19.24,
        help="camera/eye baseline distance in mm passed to ./spatial (default 19.24)",
    )
    parser.add_argument(
        "--hadjust",
        type=float,
        default=0.02,
        help="horizontal alignment adjust passed to ./spatial (default 0.02)",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default="rect",
        help="projection passed to ./spatial: rect | fisheye | half_equirect (default rect)",
    )
    parser.add_argument(
        "--spatial-extra",
        type=str,
        default="",
        help="free-form extra args appended verbatim to the ./spatial make command (e.g. \"--primary right\")",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="pre-process zoom applied to source frames before depth shift; >1.0 zooms in (crops), <1.0 zooms out (letterboxes), 1.0 unchanged",
    )
    parser.add_argument(
        "--stereo-format",
        choices=["ou", "sbs"],
        default="ou",
        help="output stereo layout: ou=over-under (top/bottom, default), sbs=side-by-side (left/right)",
    )

    args = parser.parse_args()

    if args.photo:
        image_handler = ImageHandler(args.photo)
        image_handler.make_3d_image()
    elif args.video:
        video_handler = VideoHandler(
            args.video,
            model_size=args.model_size,
            target_fps=args.target_fps,
            shift_left=args.shift_left,
            shift_right=args.shift_right,
            hfov=args.hfov,
            cdist=args.cdist,
            hadjust=args.hadjust,
            projection=args.projection,
            spatial_extra=args.spatial_extra,
            zoom=args.zoom,
            stereo_format=args.stereo_format,
        )
        video_handler.make_video()
    else:
        logging.info("Please add a photo or video if you want to see anything happen!")
